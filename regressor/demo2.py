import sys
import os
import os.path as osp

os.environ['PYOPENGL_PLATFORM'] = 'egl'

from threadpoolctl import threadpool_limits
from tqdm import tqdm
import trimesh
import torch
import time
import argparse
from collections import defaultdict
from loguru import logger
import numpy as np
from omegaconf import OmegaConf, DictConfig

import resource

from human_shape.config.defaults import conf as default_conf
from human_shape.models.build import build_model
from human_shape.data import build_all_data_loaders
from human_shape.data.structures.image_list import to_image_list
from human_shape.utils import Checkpointer


rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (rlimit[1], rlimit[1]))


def weak_persp_to_blender(
        targets,
        camera_scale,
        camera_transl,
        H, W,
        sensor_width=36,
        focal_length=5000):
    ''' Converts weak-perspective camera to a perspective camera
    '''
    if torch.is_tensor(camera_scale):
        camera_scale = camera_scale.detach().cpu().numpy()
    if torch.is_tensor(camera_transl):
        camera_transl = camera_transl.detach().cpu().numpy()

    output = defaultdict(lambda: [])
    for ii, target in enumerate(targets):
        orig_bbox_size = target.get_field('orig_bbox_size')
        bbox_center = target.get_field('orig_center')
        z = 2 * focal_length / (camera_scale[ii] * orig_bbox_size)

        transl = [
            camera_transl[ii, 0].item(), camera_transl[ii, 1].item(),
            z.item()]
        shift_x = - (bbox_center[0] / W - 0.5)
        shift_y = (bbox_center[1] - 0.5 * H) / W
        focal_length_in_mm = focal_length / W * sensor_width
        output['shift_x'].append(shift_x)
        output['shift_y'].append(shift_y)
        output['transl'].append(transl)
        output['focal_length_in_mm'].append(focal_length_in_mm)
        output['focal_length_in_px'].append(focal_length)
        output['center'].append(bbox_center)
        output['sensor_width'].append(sensor_width)

    for key in output:
        output[key] = np.array(output[key])
    return output


@torch.no_grad()
def main(
    exp_cfg: DictConfig,
    demo_output_folder: os.PathLike = 'demo_output',
    focal_length: float = 5000,
    sensor_width: float = 36,
    split: str = 'test',
) -> None:

    device = torch.device('cuda')
    if not torch.cuda.is_available():
        logger.error('CUDA is not available!')
        sys.exit(3)

    logger.remove()
    logger.add(lambda x: tqdm.write(x, end=''),
               level=exp_cfg.logger_level.upper(),
               colorize=True)

    output_folder = osp.expandvars(exp_cfg.output_folder)
    os.makedirs(demo_output_folder, exist_ok=True)

    log_file = osp.join(output_folder, 'info.log')
    logger.add(log_file, level=exp_cfg.logger_level.upper(), colorize=True)

    # Build and load model
    model_dict = build_model(exp_cfg)
    model = model_dict['network']
    try:
        model = model.to(device=device)
    except RuntimeError:
        sys.exit(3)

    checkpoint_folder = osp.join(output_folder, exp_cfg.checkpoint_folder)
    if not osp.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    checkpointer = Checkpointer(model, save_dir=checkpoint_folder,
                                pretrained=exp_cfg.pretrained)

    arguments = {'iteration': 0, 'epoch_number': 0}
    extra_checkpoint_data = checkpointer.load_checkpoint()
    for key in arguments:
        if key in extra_checkpoint_data:
            arguments[key] = extra_checkpoint_data[key]

    model = model.eval()

    # Build data loaders
    dataloaders = build_all_data_loaders(
        exp_cfg, split=split, shuffle=False, enable_augment=False,
        return_full_imgs=True,
    )

    part_key = exp_cfg.get('part_key', 'pose')

    if isinstance(dataloaders[part_key], (list,)):
        assert len(dataloaders[part_key]) == 1
        body_dloader = dataloaders[part_key][0]
    else:
        body_dloader = dataloaders[part_key]

    total_time = 0
    cnt = 0

    for bidx, batch in enumerate(tqdm(body_dloader, dynamic_ncols=True)):
        full_imgs_list, body_imgs, body_targets = batch

        if body_imgs is None:
            continue

        full_imgs = to_image_list(full_imgs_list)
        body_imgs = body_imgs.to(device=device)
        body_targets = [target.to(device) for target in body_targets]
        if full_imgs is not None:
            full_imgs = full_imgs.to(device=device)

        # Model inference
        torch.cuda.synchronize()
        start = time.perf_counter()
        model_output = model(body_imgs, body_targets, full_imgs=full_imgs,
                             device=device)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        cnt += 1
        total_time += elapsed

        _, _, H, W = full_imgs.shape

        stage_keys = model_output.get('stage_keys')

        # Process each stage to extract meshes
        for stage_key in tqdm(stage_keys, leave=False):
            stage_n_out = model_output[stage_key]
            model_vertices = stage_n_out.get('vertices', None)
            if model_vertices is None:
                continue

            faces = stage_n_out['faces']
            model_vertices = model_vertices.detach().cpu().numpy()
            camera_parameters = model_output.get('camera_parameters', {})
            camera_scale = camera_parameters['scale'].detach()
            camera_transl = camera_parameters['translation'].detach()

            hd_params = weak_persp_to_blender(
                body_targets,
                camera_scale=camera_scale,
                camera_transl=camera_transl,
                H=H, W=W,
                sensor_width=sensor_width,
                focal_length=focal_length,
            )

            # Save PLY files
            for idx in tqdm(range(len(body_targets)), 'Saving PLY files...'):
                fname = body_targets[idx].get_field('fname')
                filename = body_targets[idx].get_field('filename', '')
                
                if filename != '':
                    f1, f2 = filename.split('/')[-3:-1]
                    curr_out_path = osp.join(demo_output_folder, f1, f2)
                else:
                    curr_out_path = demo_output_folder

                imgfname = fname.split('.')[0]
                os.makedirs(curr_out_path, exist_ok=True)

                # Store the body mesh
                mesh = trimesh.Trimesh(model_vertices[idx] +
                                       hd_params['transl'][idx], faces,
                                       process=False)
                mesh_fname = osp.join(curr_out_path,
                                      f'{imgfname}_{stage_key}.ply')
                mesh.export(mesh_fname)
                logger.info(f'Saved mesh: {mesh_fname}')

    logger.info(f'Average inference time: {total_time / cnt}')


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    arg_formatter = argparse.ArgumentDefaultsHelpFormatter
    description = 'PyTorch SMPL-X Regressor Demo - PLY Only'
    parser = argparse.ArgumentParser(formatter_class=arg_formatter,
                                     description=description)

    parser.add_argument('--exp-cfg', type=str, dest='exp_cfgs',
                        nargs='+',
                        help='The configuration of the experiment')
    parser.add_argument('--output-folder', dest='output_folder',
                        default='demo_output', type=str,
                        help='The folder where the PLY files will be saved')
    parser.add_argument('--datasets', nargs='+',
                        default=['openpose'], type=str,
                        help='Datasets to process')
    parser.add_argument('--exp-opts', default=[], dest='exp_opts',
                        nargs='*',
                        help='The configuration of the Detector')
    parser.add_argument('--focal-length', dest='focal_length', type=float,
                        default=5000,
                        help='Focal length')
    parser.add_argument('--split', default='test', type=str,
                        choices=['train', 'test', 'val'],
                        help='Which split to use')

    cmd_args = parser.parse_args()

    output_folder = cmd_args.output_folder
    focal_length = cmd_args.focal_length
    split = cmd_args.split

    cfg = default_conf.copy()

    for exp_cfg in cmd_args.exp_cfgs:
        if exp_cfg:
            cfg.merge_with(OmegaConf.load(exp_cfg))
    if cmd_args.exp_opts:
        cfg.merge_with(OmegaConf.from_cli(cmd_args.exp_opts))

    cfg.is_training = False
    
    # Clear splits and set the desired dataset
    for part_key in ['pose', 'shape']:
        splits = cfg.datasets.get(part_key, {}).get('splits', {})
        if splits:
            splits['train'] = []
            splits['val'] = []
            splits['test'] = []
    part_key = cfg.get('part_key', 'pose')
    cfg.datasets[part_key].splits[split] = cmd_args.datasets

    with threadpool_limits(limits=1):
        main(cfg, demo_output_folder=output_folder,
             focal_length=focal_length,
             split=split)