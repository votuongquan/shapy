import numpy as np
import PIL.Image as pil_img
import PIL.ImageOps
from loguru import logger
import cv2
import PIL.ExifTags
import warnings
import os

# Suppress any lingering jpeg4py warnings
warnings.filterwarnings("ignore", message=".*decompressor.*", category=AttributeError)
warnings.filterwarnings("ignore", message=".*'JPEG' object has no attribute.*", category=AttributeError)
warnings.filterwarnings("ignore", category=AttributeError, module="jpeg4py")

# Set environment variables to reduce verbosity
os.environ['JPEG4PY_VERBOSE'] = '0'

# We'll use Pillow and OpenCV instead of jpeg4py for better reliability


def read_img(img_fn, dtype=np.float32):
    """
    Read image from file using Pillow and OpenCV with automatic fallback.
    Handles EXIF orientation for JPEG files.
    
    Args:
        img_fn (str): Path to image file
        dtype: Target data type (np.float32 or np.uint8)
        
    Returns:
        numpy.ndarray: Image array in RGB format
    """
    img = None
    
    # Try Pillow first (handles EXIF automatically and supports more formats)
    try:
        with pil_img.open(img_fn) as pil_image:
            # Handle EXIF orientation automatically
            pil_image = _apply_exif_orientation_pil(pil_image)
            
            # Convert to RGB if needed
            if pil_image.mode != 'RGB':
                if pil_image.mode == 'RGBA':
                    # Handle alpha channel by compositing over white background
                    background = pil_img.new('RGB', pil_image.size, (255, 255, 255))
                    background.paste(pil_image, mask=pil_image.split()[-1])
                    pil_image = background
                else:
                    pil_image = pil_image.convert('RGB')
            
            img = np.array(pil_image)
            logger.debug(f"Successfully read {img_fn} with Pillow")
            
    except Exception as e:
        logger.warning(f'Failed to read {img_fn} with Pillow: {e}. Falling back to OpenCV.')
        
    # Fallback to OpenCV if Pillow failed
    if img is None:
        try:
            img = cv2.imread(img_fn)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Handle EXIF orientation for OpenCV (manual approach)
                if img_fn.lower().endswith(('jpeg', 'jpg')):
                    img = _handle_exif_orientation_cv2(img_fn, img)
                
                logger.debug(f"Successfully read {img_fn} with OpenCV")
            else:
                raise ValueError(f"OpenCV imread returned None for {img_fn}")
                
        except Exception as e:
            logger.error(f"Failed to read {img_fn} with OpenCV: {e}")
            raise ValueError(f"Could not load image: {img_fn}")
    
    # Final check if image was loaded successfully
    if img is None:
        raise ValueError(f"Could not load image: {img_fn}")
    
    # Convert dtype if needed
    if dtype == np.float32:
        if img.dtype == np.uint8:
            img = img.astype(dtype) / 255.0
            img = np.clip(img, 0, 1)
    elif dtype == np.uint8:
        if img.dtype != np.uint8:
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    
    return img


def _apply_exif_orientation_pil(pil_image):
    """
    Apply EXIF orientation transformation to PIL image.
    Uses Pillow's built-in orientation handling.
    
    Args:
        pil_image (PIL.Image): Input PIL image
        
    Returns:
        PIL.Image: Oriented image
    """
    try:
        # Use Pillow's built-in EXIF orientation handling
        pil_image = pil_img.ImageOps.exif_transpose(pil_image)
    except Exception as e:
        logger.warning(f"Could not apply EXIF orientation: {e}")
    
    return pil_image


def _handle_exif_orientation_cv2(img_fn, img):
    """
    Handle EXIF orientation for OpenCV-loaded images.
    
    Args:
        img_fn (str): Image filename
        img (numpy.ndarray): OpenCV-loaded image
        
    Returns:
        numpy.ndarray: Oriented image
    """
    try:
        with pil_img.open(img_fn) as pil_image:
            exif_raw_dict = pil_image._getexif()
            if exif_raw_dict is not None:
                exif_data = {
                    PIL.ExifTags.TAGS[k]: v
                    for k, v in exif_raw_dict.items()
                    if k in PIL.ExifTags.TAGS
                }
                orientation = exif_data.get('Orientation', 1)
                if orientation != 1:
                    img = _apply_exif_orientation_numpy(img, orientation)
                    logger.debug(f"Applied EXIF orientation {orientation} to {img_fn}")
    except Exception as e:
        logger.warning(f"Could not read EXIF data from {img_fn}: {e}")
    
    return img


def _apply_exif_orientation_numpy(img, orientation):
    """
    Apply EXIF orientation transformation to numpy array.
    
    Args:
        img (numpy.ndarray): Input image
        orientation (int): EXIF orientation value
        
    Returns:
        numpy.ndarray: Oriented image
    """
    if orientation == 1 or orientation == 0:
        # Normal image - nothing to do!
        return img
    elif orientation == 2:
        # Mirrored left to right
        return np.fliplr(img)
    elif orientation == 3:
        # Rotated 180 degrees
        return np.rot90(img, k=2)
    elif orientation == 4:
        # Mirrored top to bottom
        return np.fliplr(np.rot90(img, k=2))
    elif orientation == 5:
        # Mirrored along top-left diagonal
        return np.fliplr(np.rot90(img, axes=(1, 0)))
    elif orientation == 6:
        # Rotated 90 degrees
        return np.rot90(img, axes=(1, 0))
    elif orientation == 7:
        # Mirrored along top-right diagonal
        return np.fliplr(np.rot90(img))
    elif orientation == 8:
        # Rotated 270 degrees
        return np.rot90(img)
    else:
        logger.warning(f"Unknown EXIF orientation: {orientation}")
        return img


def read_img_fast(img_fn, dtype=np.float32):
    """
    Fast image reading function optimized for performance.
    Uses OpenCV for speed, Pillow for fallback.
    
    Args:
        img_fn (str): Path to image file
        dtype: Target data type (np.float32 or np.uint8)
        
    Returns:
        numpy.ndarray: Image array in RGB format
    """
    # Try OpenCV first for speed
    try:
        img = cv2.imread(img_fn)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Handle EXIF orientation if JPEG
            if img_fn.lower().endswith(('jpeg', 'jpg')):
                img = _handle_exif_orientation_cv2(img_fn, img)
            
            # Convert dtype
            if dtype == np.float32 and img.dtype == np.uint8:
                img = img.astype(dtype) / 255.0
            elif dtype == np.uint8 and img.dtype != np.uint8:
                img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
                
            return img
    except Exception as e:
        logger.warning(f"OpenCV failed for {img_fn}: {e}")
    
    # Fallback to regular read_img
    return read_img(img_fn, dtype)


def save_img(img, img_fn, quality=95):
    """
    Save image to file with proper format handling.
    
    Args:
        img (numpy.ndarray): Image array
        img_fn (str): Output filename
        quality (int): JPEG quality (if saving as JPEG)
    """
    try:
        # Ensure proper data type
        if img.dtype != np.uint8:
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        
        # Convert to PIL and save
        pil_image = pil_img.fromarray(img)
        
        if img_fn.lower().endswith(('jpeg', 'jpg')):
            pil_image.save(img_fn, 'JPEG', quality=quality, optimize=True)
        else:
            pil_image.save(img_fn)
            
        logger.debug(f"Successfully saved image to {img_fn}")
        
    except Exception as e:
        logger.error(f"Failed to save image to {img_fn}: {e}")
        raise