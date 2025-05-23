import sys
import os
import os.path as osp
import json
import tempfile
import shutil

os.environ['PYOPENGL_PLATFORM'] = 'egl'

from threadpoolctl import threadpool_limits
import trimesh
import torch
import time
from collections import defaultdict
from loguru import logger
import numpy as np
from omegaconf import OmegaConf, DictConfig
import cv2
from PIL import Image

import resource

from human_shape.config.defaults import conf as default_conf
from human_shape.models.build import build_model
from human_shape.data import build_all_data_loaders
from human_shape.data.structures.image_list import to_image_list
from human_shape.utils import Checkpointer


rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (rlimit[1], rlimit[1]))


class SingleImageProcessor:
    """
    A class-based processor for converting a single image with keypoints to PLY mesh.
    """
    
    def __init__(self, 
                 config_path: str,
                 model_path: str,
                 focal_length: float = 5000,
                 sensor_width: float = 36,
                 device: str = 'cuda'):
        """
        Initialize the processor.
        
        Args:
            config_path: Path to the experiment config YAML file
            model_path: Path to the trained model folder
            focal_length: Camera focal length
            sensor_width: Camera sensor width
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.config_path = config_path
        self.model_path = model_path
        self.focal_length = focal_length
        self.sensor_width = sensor_width
        self.device = torch.device(device)
        
        if device == 'cuda' and not torch.cuda.is_available():
            logger.error('CUDA is not available!')
            raise RuntimeError('CUDA is not available!')
        
        # Setup logger
        logger.remove()
        logger.add(sys.stderr, level="INFO", colorize=True)
        
        # Load configuration
        self.cfg = self._load_config()
        
        # Build and load model
        self.model = self._build_model()
        
        logger.info("SingleImageProcessor initialized successfully")
    
    def _load_config(self) -> DictConfig:
        """Load and setup configuration."""
        cfg = default_conf.copy()
        
        # Load experiment config
        if self.config_path and osp.exists(self.config_path):
            cfg.merge_with(OmegaConf.load(self.config_path))
        
        # Set basic configuration
        cfg.is_training = False
        cfg.output_folder = self.model_path
        cfg.part_key = 'pose'
        cfg.datasets.batch_size = 1
        cfg.datasets.pose_shape_ratio = 1.0
        
        return cfg
    
    def _build_model(self):
        """Build and load the model."""
        model_dict = build_model(self.cfg)
        model = model_dict['network']
        
        try:
            model = model.to(device=self.device)
        except RuntimeError as e:
            logger.error(f"Failed to move model to device: {e}")
            raise
        
        # Load checkpoint
        checkpoint_folder = osp.join(self.model_path, self.cfg.checkpoint_folder)
        if not osp.exists(checkpoint_folder):
            raise FileNotFoundError(f"Checkpoint folder not found: {checkpoint_folder}")
        
        checkpointer = Checkpointer(model, save_dir=checkpoint_folder,
                                    pretrained=self.cfg.pretrained)
        
        extra_checkpoint_data = checkpointer.load_checkpoint()
        model = model.eval()
        
        return model
    
    def _create_temp_dataset(self, image_path: str, keypoints_path: str) -> str:
        """
        Create a temporary dataset structure for a single image.
        
        Args:
            image_path: Path to the input image
            keypoints_path: Path to the keypoints JSON file
            
        Returns:
            Path to temporary dataset folder
        """
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Create dataset structure
        data_folder = osp.join(temp_dir, 'single_image_data')
        img_folder = osp.join(data_folder, 'images')
        keyp_folder = osp.join(data_folder, 'openpose')
        
        os.makedirs(img_folder, exist_ok=True)
        os.makedirs(keyp_folder, exist_ok=True)
        
        # Copy image
        img_name = osp.basename(image_path)
        img_dest = osp.join(img_folder, img_name)
        shutil.copy2(image_path, img_dest)
        
        # Copy keypoints
        keyp_name = osp.basename(keypoints_path)
        keyp_dest = osp.join(keyp_folder, keyp_name)
        shutil.copy2(keypoints_path, keyp_dest)
        
        return data_folder
    
    def _setup_single_image_config(self, data_folder: str):
        """Setup configuration for single image processing."""
        # Update dataset configuration
        self.cfg.datasets.pose.openpose.data_folder = data_folder
        self.cfg.datasets.pose.openpose.img_folder = 'images'
        self.cfg.datasets.pose.openpose.keyp_folder = 'openpose'
        
        # Clear splits and set test split
        for part_key in ['pose', 'shape']:
            splits = self.cfg.datasets.get(part_key, {}).get('splits', {})
            if splits:
                splits['train'] = []
                splits['val'] = []
                splits['test'] = []
        
        self.cfg.datasets.pose.splits.test = ['openpose']
    
    def _weak_persp_to_blender(self, targets, camera_scale, camera_transl, H, W):
        """Convert weak-perspective camera to perspective camera parameters."""
        if torch.is_tensor(camera_scale):
            camera_scale = camera_scale.detach().cpu().numpy()
        if torch.is_tensor(camera_transl):
            camera_transl = camera_transl.detach().cpu().numpy()

        output = defaultdict(lambda: [])
        for ii, target in enumerate(targets):
            orig_bbox_size = target.get_field('orig_bbox_size')
            bbox_center = target.get_field('orig_center')
            z = 2 * self.focal_length / (camera_scale[ii] * orig_bbox_size)

            transl = [
                camera_transl[ii, 0].item(), 
                camera_transl[ii, 1].item(),
                z.item()
            ]
            shift_x = - (bbox_center[0] / W - 0.5)
            shift_y = (bbox_center[1] - 0.5 * H) / W
            focal_length_in_mm = self.focal_length / W * self.sensor_width
            
            output['shift_x'].append(shift_x)
            output['shift_y'].append(shift_y)
            output['transl'].append(transl)
            output['focal_length_in_mm'].append(focal_length_in_mm)
            output['focal_length_in_px'].append(self.focal_length)
            output['center'].append(bbox_center)
            output['sensor_width'].append(self.sensor_width)

        for key in output:
            output[key] = np.array(output[key])
        return output
    
    @torch.no_grad()
    def process_image(self, image_path: str, keypoints_path: str, output_path: str = None) -> str:
        """
        Process a single image and generate PLY mesh.
        
        Args:
            image_path: Path to the input image
            keypoints_path: Path to the keypoints JSON file
            output_path: Path where to save the PLY file (optional)
            
        Returns:
            Path to the generated PLY file
        """
        if not osp.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not osp.exists(keypoints_path):
            raise FileNotFoundError(f"Keypoints file not found: {keypoints_path}")
        
        logger.info(f"Processing image: {image_path}")
        logger.info(f"Using keypoints: {keypoints_path}")
        
        # Create temporary dataset
        temp_data_folder = self._create_temp_dataset(image_path, keypoints_path)
        
        try:
            # Setup configuration for single image
            self._setup_single_image_config(temp_data_folder)
            
            # Build data loader
            with threadpool_limits(limits=1):
                dataloaders = build_all_data_loaders(
                    self.cfg, split='test', shuffle=False, enable_augment=False,
                    return_full_imgs=True,
                )
            
            part_key = self.cfg.get('part_key', 'pose')
            if isinstance(dataloaders[part_key], (list,)):
                body_dloader = dataloaders[part_key][0]
            else:
                body_dloader = dataloaders[part_key]
            
            # Process the single image
            for batch in body_dloader:
                full_imgs_list, body_imgs, body_targets = batch
                
                if body_imgs is None:
                    raise RuntimeError("Failed to load image data")
                
                full_imgs = to_image_list(full_imgs_list)
                body_imgs = body_imgs.to(device=self.device)
                body_targets = [target.to(self.device) for target in body_targets]
                if full_imgs is not None:
                    full_imgs = full_imgs.to(device=self.device)
                
                # Model inference
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                start = time.perf_counter()
                model_output = self.model(body_imgs, body_targets, full_imgs=full_imgs,
                                        device=self.device)
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                elapsed = time.perf_counter() - start
                
                logger.info(f"Inference time: {elapsed:.4f}s")
                
                _, _, H, W = full_imgs.shape
                
                # Extract mesh from final stage
                stage_n_out = model_output['stage_02']
                model_vertices = stage_n_out.get('vertices', None)
                
                if model_vertices is None:
                    raise RuntimeError("No vertices generated by the model")
                
                faces = stage_n_out['faces']
                model_vertices = model_vertices.detach().cpu().numpy()
                camera_parameters = model_output.get('camera_parameters', {})
                camera_scale = camera_parameters['scale'].detach()
                camera_transl = camera_parameters['translation'].detach()
                
                hd_params = self._weak_persp_to_blender(
                    body_targets,
                    camera_scale=camera_scale,
                    camera_transl=camera_transl,
                    H=H, W=W,
                )
                
                # Create output filename if not provided
                if output_path is None:
                    img_name = osp.splitext(osp.basename(image_path))[0]
                    output_path = f"{img_name}_mesh.ply"
                
                # Ensure output directory exists
                os.makedirs(osp.dirname(osp.abspath(output_path)), exist_ok=True)
                
                # Create and save mesh
                mesh = trimesh.Trimesh(
                    model_vertices[0] + hd_params['transl'][0], 
                    faces,
                    process=False
                )
                mesh.export(output_path)
                
                logger.info(f"PLY mesh saved to: {output_path}")
                
                return output_path
                
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_data_folder, ignore_errors=True)
    
    def process_batch(self, image_keypoint_pairs: list, output_folder: str = 'output') -> list:
        """
        Process multiple images in batch.
        
        Args:
            image_keypoint_pairs: List of tuples (image_path, keypoints_path)
            output_folder: Folder to save all PLY files
            
        Returns:
            List of paths to generated PLY files
        """
        os.makedirs(output_folder, exist_ok=True)
        results = []
        
        for i, (img_path, keyp_path) in enumerate(image_keypoint_pairs):
            img_name = osp.splitext(osp.basename(img_path))[0]
            output_path = osp.join(output_folder, f"{img_name}_mesh.ply")
            
            try:
                result_path = self.process_image(img_path, keyp_path, output_path)
                results.append(result_path)
                logger.info(f"Processed {i+1}/{len(image_keypoint_pairs)}: {img_path}")
            except Exception as e:
                logger.error(f"Failed to process {img_path}: {e}")
                results.append(None)
        
        return results


def main():
    """Example usage of the SingleImageProcessor class."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Single Image to PLY Processor')
    parser.add_argument('--config', required=True, help='Path to config YAML file')
    parser.add_argument('--model-path', required=True, help='Path to trained model folder')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--keypoints', required=True, help='Path to keypoints JSON file')
    parser.add_argument('--output', help='Output PLY file path')
    parser.add_argument('--focal-length', type=float, default=5000, help='Focal length')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = SingleImageProcessor(
        config_path=args.config,
        model_path=args.model_path,
        focal_length=args.focal_length,
        device=args.device
    )
    
    # Process single image
    output_path = processor.process_image(
        image_path=args.image,
        keypoints_path=args.keypoints,
        output_path=args.output
    )
    
    print(f"Generated PLY file: {output_path}")


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    main()