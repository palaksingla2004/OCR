import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from typing import List, Dict, Optional, Tuple, Union
import os

def plot_sample(image: Union[str, Image.Image], text: str, prediction: Optional[str] = None,
               figsize: Tuple[int, int] = (10, 5)):
    """
    Plot an image with its ground truth text and optionally the predicted text.
    
    Args:
        image: Path to image or PIL Image
        text: Ground truth text
        prediction: Predicted text (optional)
        figsize: Figure size as (width, height)
    """
    plt.figure(figsize=figsize)
    
    # Load image if path is provided
    if isinstance(image, str):
        img = Image.open(image)
    else:
        img = image
    
    # Show image
    plt.imshow(img)
    plt.axis('off')
    
    # Set title
    if prediction is not None:
        plt.title(f"Ground Truth: \"{text}\"\nPrediction: \"{prediction}\"")
    else:
        plt.title(f"Text: \"{text}\"")
    
    plt.tight_layout()
    plt.show()

def plot_multiple_samples(images: List[Union[str, Image.Image]], texts: List[str], 
                        predictions: Optional[List[str]] = None, 
                        num_samples: int = 5, figsize: Tuple[int, int] = (15, 10)):
    """
    Plot multiple images with their texts and predictions.
    
    Args:
        images: List of image paths or PIL Images
        texts: List of ground truth texts
        predictions: List of predicted texts (optional)
        num_samples: Number of samples to plot
        figsize: Figure size as (width, height)
    """
    num_samples = min(num_samples, len(images))
    
    plt.figure(figsize=figsize)
    
    for i in range(num_samples):
        plt.subplot(num_samples, 1, i+1)
        
        # Load image if path is provided
        if isinstance(images[i], str):
            img = Image.open(images[i])
        else:
            img = images[i]
        
        # Show image
        plt.imshow(img)
        plt.axis('off')
        
        # Set title
        if predictions is not None:
            plt.title(f"GT: \"{texts[i]}\"\nPred: \"{predictions[i]}\"")
        else:
            plt.title(f"Text: \"{texts[i]}\"")
    
    plt.tight_layout()
    plt.show()

def plot_training_history(training_history: Dict[str, List[float]], figsize: Tuple[int, int] = (12, 8)):
    """
    Plot training history metrics.
    
    Args:
        training_history: Dictionary with metric names as keys and lists of values as values
        figsize: Figure size as (width, height)
    """
    plt.figure(figsize=figsize)
    
    # Number of metrics to plot
    num_metrics = len(training_history)
    
    for i, (metric_name, values) in enumerate(training_history.items()):
        plt.subplot(num_metrics, 1, i+1)
        plt.plot(values)
        plt.title(f"{metric_name}")
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(predictions: List[str], references: List[str], 
                         num_chars: int = 10, figsize: Tuple[int, int] = (10, 8)):
    """
    Plot a confusion matrix for the most common character errors.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        num_chars: Number of most common characters to include
        figsize: Figure size as (width, height)
    """
    # Collect all chars from the references
    all_chars = "".join(references)
    char_counts = {}
    
    for char in all_chars:
        if char.isalnum() or char in " ,.?!;:-'\"()":
            char_counts[char] = char_counts.get(char, 0) + 1
    
    # Get most common chars
    most_common = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)[:num_chars]
    common_chars = [char for char, _ in most_common]
    
    # Initialize confusion matrix
    confusion = np.zeros((len(common_chars), len(common_chars)))
    
    # Fill confusion matrix
    for pred, ref in zip(predictions, references):
        min_len = min(len(pred), len(ref))
        
        for i in range(min_len):
            if ref[i] in common_chars and pred[i] in common_chars:
                ref_idx = common_chars.index(ref[i])
                pred_idx = common_chars.index(pred[i])
                confusion[ref_idx, pred_idx] += 1
    
    # Plot confusion matrix
    plt.figure(figsize=figsize)
    plt.imshow(confusion, cmap='Blues')
    plt.colorbar()
    
    # Add labels
    plt.xlabel('Predicted')
    plt.ylabel('Reference')
    plt.title('Character Confusion Matrix')
    
    # Add ticks
    plt.xticks(range(len(common_chars)), common_chars)
    plt.yticks(range(len(common_chars)), common_chars)
    
    # Add values to cells
    for i in range(len(common_chars)):
        for j in range(len(common_chars)):
            plt.text(j, i, int(confusion[i, j]), ha='center', va='center',
                    color='white' if confusion[i, j] > confusion.max() / 2 else 'black')
    
    plt.tight_layout()
    plt.show()

def plot_error_distribution(predictions: List[str], references: List[str], 
                          figsize: Tuple[int, int] = (12, 6)):
    """
    Plot the distribution of CER and WER across samples.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        figsize: Figure size as (width, height)
    """
    from jiwer import wer, cer
    
    # Calculate individual error rates
    individual_wer = []
    individual_cer = []
    
    for pred, ref in zip(predictions, references):
        if not ref:  # Skip empty references
            continue
            
        try:
            individual_wer.append(wer(ref, pred))
            individual_cer.append(cer(ref, pred))
        except Exception:
            # Skip samples that cause errors in WER/CER calculation
            continue
    
    plt.figure(figsize=figsize)
    
    # Plot WER distribution
    plt.subplot(1, 2, 1)
    plt.hist(individual_wer, bins=20, alpha=0.7)
    plt.axvline(x=np.mean(individual_wer), color='r', linestyle='--', 
                label=f'Mean: {np.mean(individual_wer):.4f}')
    plt.axvline(x=np.median(individual_wer), color='g', linestyle='--', 
                label=f'Median: {np.median(individual_wer):.4f}')
    plt.xlabel('WER')
    plt.ylabel('Count')
    plt.title('Word Error Rate Distribution')
    plt.legend()
    
    # Plot CER distribution
    plt.subplot(1, 2, 2)
    plt.hist(individual_cer, bins=20, alpha=0.7)
    plt.axvline(x=np.mean(individual_cer), color='r', linestyle='--', 
                label=f'Mean: {np.mean(individual_cer):.4f}')
    plt.axvline(x=np.median(individual_cer), color='g', linestyle='--', 
                label=f'Median: {np.median(individual_cer):.4f}')
    plt.xlabel('CER')
    plt.ylabel('Count')
    plt.title('Character Error Rate Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def save_sample_predictions(output_dir: str, images: List[Union[str, Image.Image]], 
                          texts: List[str], predictions: List[str], 
                          max_samples: int = 10):
    """
    Save sample predictions as images with annotations.
    
    Args:
        output_dir: Directory to save visualization images
        images: List of image paths or PIL Images
        texts: List of ground truth texts
        predictions: List of predicted texts
        max_samples: Maximum number of samples to save
    """
    os.makedirs(output_dir, exist_ok=True)
    
    num_samples = min(max_samples, len(images))
    
    for i in range(num_samples):
        plt.figure(figsize=(10, 5))
        
        # Load image if path is provided
        if isinstance(images[i], str):
            img = Image.open(images[i])
            img_path = images[i]
        else:
            img = images[i]
            img_path = f"image_{i}"
        
        # Get filename without path
        if isinstance(img_path, str):
            filename = os.path.basename(img_path)
        else:
            filename = f"image_{i}.png"
        
        # Show image
        plt.imshow(img)
        plt.axis('off')
        
        # Set title and match status
        correct = texts[i] == predictions[i]
        status = "✓" if correct else "✗"
        plt.title(f"{status} GT: \"{texts[i]}\"\nPred: \"{predictions[i]}\"")
        
        # Save figure
        output_path = os.path.join(output_dir, f"pred_{i}_{filename}")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

def visualize_attention(image: Union[str, Image.Image], text: str, attention_weights: np.ndarray,
                      figsize: Tuple[int, int] = (15, 10)):
    """
    Visualize attention weights for a prediction.
    Note: This only works with transformer models that expose attention weights.
    
    Args:
        image: Image path or PIL Image
        text: Predicted text
        attention_weights: Attention weights from the model
        figsize: Figure size as (width, height)
    """
    # Load image if path is provided
    if isinstance(image, str):
        img = np.array(Image.open(image))
    else:
        img = np.array(image)
    
    # Get image dimensions
    H, W = img.shape[:2]
    
    # Reshape attention weights
    attn_h, attn_w = attention_weights.shape
    
    plt.figure(figsize=figsize)
    
    # Display original image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')
    
    # Display attention heatmap overlaid on image
    plt.subplot(1, 2, 2)
    plt.imshow(img)
    
    # Resize attention to image dimensions
    attention_resized = np.resize(attention_weights, (H, W))
    plt.imshow(attention_resized, alpha=0.5, cmap='jet')
    plt.title(f"Attention Heatmap\nPrediction: \"{text}\"")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show() 