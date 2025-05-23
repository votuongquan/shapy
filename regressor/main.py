import os
import sys
import argparse
from demo3 import SingleImageProcessor

def main():
    parser = argparse.ArgumentParser(description='Process image to generate 3D PLY mesh')
    
    # Required arguments
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--keypoints', required=True, help='Path to keypoints JSON file')
    parser.add_argument('--config', required=True, help='Path to config YAML file')
    parser.add_argument('--model-path', required=True, help='Path to model directory')
    
    # Optional arguments
    parser.add_argument('--output', help='Output PLY file path (default: auto-generated)')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)
    
    if not os.path.exists(args.keypoints):
        print(f"Error: Keypoints file not found: {args.keypoints}")
        sys.exit(1)
    
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model directory not found: {args.model_path}")
        sys.exit(1)
    
    print("Initializing processor...")
    
    try:
        # Initialize processor
        processor = SingleImageProcessor(
            config_path=args.config,
            model_path=args.model_path,
            device=args.device
        )
        
        # Process image
        print("Processing image...")
        output_path = processor.process_image(
            image_path=args.image,
            keypoints_path=args.keypoints,
            output_path=args.output
        )
        
        print(f"Success! Generated PLY mesh: {output_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()