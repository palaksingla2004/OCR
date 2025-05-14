import cv2
import numpy as np
import torch
from PIL import Image
from typing import Tuple, Union, List

def resize_image(image: Union[np.ndarray, Image.Image], target_size: Tuple[int, int] = (384, 384)) -> Image.Image:
    """
    Resize an image to the target size required by the TrOCR model.
    
    Args:
        image: Input image (numpy array or PIL Image)
        target_size: Target size as (width, height)
    
    Returns:
        Resized PIL Image
    """
    if isinstance(image, np.ndarray):
        # Convert OpenCV BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
    
    return image.resize(target_size, Image.BILINEAR)

def preprocess_image(image_path: str, target_size: Tuple[int, int] = (384, 384), 
                    denoise: bool = True, binarize: bool = True) -> Image.Image:
    """
    Preprocess an image for the OCR model.
    
    Args:
        image_path: Path to the image file
        target_size: Target size for resizing
        denoise: Whether to apply denoising
        binarize: Whether to binarize the image
    
    Returns:
        Preprocessed PIL Image
    """
    # Read the image
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
    else:
        image = image_path
        
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # Apply denoising if requested
    if denoise:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
    # Apply binarization if requested
    if binarize:
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
    
    # Convert to PIL Image and resize
    pil_image = Image.fromarray(gray)
    return resize_image(pil_image, target_size)

def augment_image(image: np.ndarray) -> np.ndarray:
    """
    Apply augmentations to the image to improve model robustness.
    
    Args:
        image: Input image as numpy array
    
    Returns:
        Augmented image
    """
    # Random rotation (±10 degrees)
    angle = np.random.uniform(-10, 10)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    # Random scaling
    scale = np.random.uniform(0.9, 1.1)
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    
    # Adjust back to original size
    h_new, w_new = image.shape[:2]
    if h_new != h or w_new != w:
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Random brightness/contrast adjustment
    alpha = np.random.uniform(0.8, 1.2)  # Contrast
    beta = np.random.uniform(-10, 10)    # Brightness
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    return image

def batch_preprocess(image_paths: List[str], target_size: Tuple[int, int] = (384, 384), 
                    augment: bool = False) -> List[Image.Image]:
    """
    Preprocess a batch of images.
    
    Args:
        image_paths: List of paths to image files
        target_size: Target size for resizing
        augment: Whether to apply augmentation
        
    Returns:
        List of preprocessed PIL Images
    """
    processed_images = []
    
    for path in image_paths:
        img = cv2.imread(path)
        
        if img is None:
            print(f"Warning: Could not read image {path}")
            continue
            
        if augment:
            img = augment_image(img)
            
        processed = preprocess_image(img, target_size)
        processed_images.append(processed)
        
    return processed_images 