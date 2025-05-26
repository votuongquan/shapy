import numpy as np
import PIL.Image as pil_img
from loguru import logger
import cv2
import PIL.ExifTags

# Try to import jpeg4py, but fallback to cv2 if not available
try:
    import jpeg4py as jpeg
    JPEG4PY_AVAILABLE = True
    logger.info("jpeg4py loaded successfully")
except (ImportError, OSError) as e:
    logger.warning(f"jpeg4py not available: {e}. Using cv2 for all image reading.")
    JPEG4PY_AVAILABLE = False


def read_img(img_fn, dtype=np.float32):
    """
    Read image from file with automatic fallback from jpeg4py to cv2.
    Handles EXIF orientation for JPEG files.
    
    Args:
        img_fn (str): Path to image file
        dtype: Target data type (np.float32 or np.uint8)
        
    Returns:
        numpy.ndarray: Image array in RGB format
    """
    img = None
    
    if img_fn.endswith(('jpeg', 'jpg', 'JPEG', 'JPG')):
        # Try jpeg4py first if available (faster for JPEG)
        if JPEG4PY_AVAILABLE:
            try:
                with open(img_fn, 'rb') as f:
                    img = jpeg.JPEG(f).decode()
                logger.debug(f"Successfully read {img_fn} with jpeg4py")
            except (jpeg.JPEGRuntimeError, OSError) as e:
                logger.warning(f'{img_fn} failed with jpeg4py: {e}. Falling back to cv2.')
                img = None
            except Exception as e:
                logger.warning(f'Unexpected error with jpeg4py for {img_fn}: {e}. Falling back to cv2.')
                img = None
        
        # Fallback to cv2 if jpeg4py failed or not available
        if img is None:
            try:
                img = cv2.imread(img_fn)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    logger.debug(f"Successfully read {img_fn} with cv2")
                else:
                    raise ValueError(f"cv2.imread returned None for {img_fn}")
            except Exception as e:
                logger.error(f"Failed to read {img_fn} with cv2: {e}")
                raise ValueError(f"Could not load image: {img_fn}")
        
        # Handle EXIF orientation for JPEG files
        try:
            with pil_img.open(img_fn) as pil_image:
                exif_raw_dict = pil_image._getexif()
                if exif_raw_dict is not None:
                    exif_data = {
                        PIL.ExifTags.TAGS[k]: v
                        for k, v in exif_raw_dict.items()
                        if k in PIL.ExifTags.TAGS
                    }
                    orientation = exif_data.get('Orientation', None)
                    if orientation is not None and orientation != 1:
                        img = _apply_exif_orientation(img, orientation)
                        logger.debug(f"Applied EXIF orientation {orientation} to {img_fn}")
        except Exception as e:
            logger.warning(f"Could not read EXIF data from {img_fn}: {e}")
    
    else:
        # For non-JPEG files (PNG, BMP, TIFF, etc.), use cv2
        try:
            img = cv2.imread(img_fn)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                logger.debug(f"Successfully read {img_fn} with cv2")
            else:
                raise ValueError(f"cv2.imread returned None for {img_fn}")
        except Exception as e:
            logger.error(f"Failed to read {img_fn}: {e}")
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


def _apply_exif_orientation(img, orientation):
    """
    Apply EXIF orientation transformation to image.
    
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