import os
import json
import random
from typing import Dict, List, Tuple, Optional, Union
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import TrOCRProcessor
from PIL import Image
import numpy as np

class HandwritingDataset(Dataset):
    """
    Dataset class for handwriting OCR tasks, compatible with TrOCR.
    """
    def __init__(
        self,
        image_paths: List[str],
        texts: List[str],
        processor: TrOCRProcessor,
        max_target_length: int = 128,
        augment: bool = False
    ):
        """
        Initialize the dataset.
        
        Args:
            image_paths: List of paths to image files
            texts: List of corresponding ground truth texts
            processor: TrOCR processor for tokenization and image processing
            max_target_length: Maximum length of target sequences
            augment: Whether to apply data augmentation
        """
        self.image_paths = image_paths
        self.texts = texts
        self.processor = processor
        self.max_target_length = max_target_length
        self.augment = augment
        
    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing pixel_values and labels
        """
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        
        # Get text
        text = self.texts[idx]
        
        # Preprocess image and text using the TrOCR processor
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
        
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        
        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "text": text  # Include the original text for evaluation
        }

def load_iam_dataset(
    base_path: str,
    processor: TrOCRProcessor,
    max_samples: Optional[int] = None,
    split: str = "train",
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[HandwritingDataset, Optional[HandwritingDataset]]:
    """
    Load the IAM handwriting dataset.
    
    Args:
        base_path: Path to the IAM dataset
        processor: TrOCR processor for tokenization and image processing
        max_samples: Maximum number of samples to load (None for all)
        split: Dataset split to use ('train', 'val', or 'test')
        val_ratio: Validation ratio (only used when split='train')
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of datasets (train, val) if split='train', else single dataset
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Path to images and annotations
    lines_path = os.path.join(base_path, "lines")
    annotation_file = os.path.join(base_path, "lines.txt")
    
    # Read annotation file
    image_paths = []
    texts = []
    
    with open(annotation_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):  # Skip header lines
                continue
            
            parts = line.strip().split(" ")
            image_id = parts[0]
            text = " ".join(parts[8:])  # Text content starts from 9th field
            
            image_path = os.path.join(lines_path, f"{image_id}.png")
            if os.path.exists(image_path):
                image_paths.append(image_path)
                texts.append(text)
    
    # Limit the number of samples if specified
    if max_samples and max_samples < len(image_paths):
        combined = list(zip(image_paths, texts))
        random.shuffle(combined)
        image_paths, texts = zip(*combined[:max_samples])
    
    # Split the data if needed
    if split == "train":
        # Split into train and validation sets
        combined = list(zip(image_paths, texts))
        random.shuffle(combined)
        
        val_size = int(len(combined) * val_ratio)
        train_data = combined[val_size:]
        val_data = combined[:val_size]
        
        train_images, train_texts = zip(*train_data)
        val_images, val_texts = zip(*val_data)
        
        train_dataset = HandwritingDataset(
            list(train_images), list(train_texts), processor, augment=True
        )
        val_dataset = HandwritingDataset(
            list(val_images), list(val_texts), processor, augment=False
        )
        
        return train_dataset, val_dataset
    else:
        # Return the full dataset for testing
        return HandwritingDataset(image_paths, texts, processor, augment=False), None

def load_imgur5k_dataset(
    base_path: str,
    processor: TrOCRProcessor,
    max_samples: Optional[int] = None,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[HandwritingDataset, HandwritingDataset]:
    """
    Load the Imgur5K dataset for handwriting recognition.
    
    Args:
        base_path: Path to the Imgur5K dataset
        processor: TrOCR processor for tokenization and image processing
        max_samples: Maximum number of samples to load (None for all)
        val_ratio: Validation ratio
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of datasets (train, val)
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Path to images and annotations
    images_dir = os.path.join(base_path, "images")
    annotation_file = os.path.join(base_path, "imgur5k_annotations.json")
    
    # Read annotation file
    with open(annotation_file, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    
    image_paths = []
    texts = []
    
    for item in annotations:
        image_path = os.path.join(images_dir, item["image_file"])
        text = item["text"]
        
        if os.path.exists(image_path):
            image_paths.append(image_path)
            texts.append(text)
    
    # Limit the number of samples if specified
    if max_samples and max_samples < len(image_paths):
        combined = list(zip(image_paths, texts))
        random.shuffle(combined)
        image_paths, texts = zip(*combined[:max_samples])
    
    # Split into train and validation sets
    combined = list(zip(image_paths, texts))
    random.shuffle(combined)
    
    val_size = int(len(combined) * val_ratio)
    train_data = combined[val_size:]
    val_data = combined[:val_size]
    
    train_images, train_texts = zip(*train_data)
    val_images, val_texts = zip(*val_data)
    
    train_dataset = HandwritingDataset(
        list(train_images), list(train_texts), processor, augment=True
    )
    val_dataset = HandwritingDataset(
        list(val_images), list(val_texts), processor, augment=False
    )
    
    return train_dataset, val_dataset

def create_dataloaders(
    train_dataset: HandwritingDataset,
    val_dataset: HandwritingDataset,
    batch_size: int = 4,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoader instances for training and validation.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        
    Returns:
        Tuple of DataLoaders (train_loader, val_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def generate_synthetic_data(
    processor: TrOCRProcessor,
    output_dir: str,
    num_samples: int = 1000,
    random_seed: int = 42
) -> HandwritingDataset:
    """
    Generate synthetic handwritten text data using TextRecognitionDataGenerator.
    This requires that TextRecognitionDataGenerator is installed.
    
    Args:
        processor: TrOCR processor for tokenization and image processing
        output_dir: Directory to save generated images
        num_samples: Number of synthetic samples to generate
        random_seed: Random seed for reproducibility
        
    Returns:
        Dataset containing synthetic samples
    """
    try:
        from trdg.generators import (
            GeneratorFromStrings,
            GeneratorFromWikipedia,
            GeneratorFromRandom
        )
    except ImportError:
        print("TextRecognitionDataGenerator not installed. Run:")
        print("pip install trdg")
        return None
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate random text data
    generator = GeneratorFromRandom(
        count=num_samples,
        random_seed=random_seed,
        language='en',
        size=32,
        skewing_angle=2,
        random_skew=True,
        blur=1,
        random_blur=True,
        handwritten=True,
    )
    
    image_paths = []
    texts = []
    
    for i, (img, text) in enumerate(generator):
        image_path = os.path.join(output_dir, f"synthetic_{i:05d}.png")
        img.save(image_path)
        
        image_paths.append(image_path)
        texts.append(text)
    
    # Create dataset from synthetic data
    synthetic_dataset = HandwritingDataset(
        image_paths, texts, processor, augment=False
    )
    
    return synthetic_dataset 