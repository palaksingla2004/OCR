#!/usr/bin/env python
# coding: utf-8

"""
# Handwriting Recognition with TrOCR

This script demonstrates how to fine-tune a TrOCR model for handwritten text recognition. 
The project aims to achieve high accuracy in recognizing diverse handwriting styles by 
leveraging modern transformer-based architectures.

## Overview

- Model: TrOCR (Transformer-based OCR) by Microsoft, which combines a Vision Transformer (ViT) encoder with a text Transformer decoder
- Datasets: IAM Handwriting Database and Imgur5K
- Target Metrics: CER <= 7% and WER <= 15%

Run this script in Google Colab for optimal performance.
"""

# Setup and Dependencies
# --------------------------

# Check if we're running in Colab
import sys
import subprocess
import os

try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Clone our repository if in Colab
if IN_COLAB:
    # Clone the repository
    subprocess.run(["git", "clone", "https://github.com/palaksingla2004/OCR.git"], check=True)
    os.chdir("OCR")
    
    # Install dependencies
    subprocess.run(["pip", "install", "-q", "-r", "requirements.txt"], check=True)
    
    # Clone TextRecognitionDataGenerator for synthetic data (optional)
    subprocess.run(["git", "clone", "https://github.com/Belval/TextRecognitionDataGenerator.git"], check=True)
    os.chdir("TextRecognitionDataGenerator")
    subprocess.run(["pip", "install", "-e", "."], check=True)
    os.chdir("..")

# Import key libraries
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from tqdm.auto import tqdm
import random
import json
import cv2
from IPython.display import display

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Import our project modules
from utils.preprocessing import preprocess_image, batch_preprocess
from utils.metrics import calculate_cer, calculate_wer, print_evaluation_report
from utils.trainer import OCRTrainer
from utils.visualization import plot_sample, plot_multiple_samples, plot_error_distribution
from data.data_loader import HandwritingDataset, create_dataloaders
from data.synthetic_generator import SyntheticTextGenerator

# Download and Prepare Datasets
# ----------------------------

# Create directories for datasets
os.makedirs("datasets", exist_ok=True)
os.makedirs("datasets/iam", exist_ok=True)
os.makedirs("datasets/imgur5k", exist_ok=True)
os.makedirs("datasets/synthetic", exist_ok=True)

# Download IAM dataset
if not os.path.exists("datasets/iam/lines.txt"):
    print("Downloading IAM dataset...")
    
    # Check if using Colab to mount Google Drive
    if IN_COLAB:
        from google.colab import drive
        drive.mount('/content/drive')
        
        # Alternative: Download using Hugging Face datasets
        from datasets import load_dataset
        iam = load_dataset("iamdb", "lines")
        
        # Save samples
        os.makedirs("datasets/iam/lines", exist_ok=True)
        
        with open("datasets/iam/lines.txt", "w") as f:
            for i, sample in enumerate(iam["train"]):
                # Save image
                image_path = f"datasets/iam/lines/{sample['id']}.png"
                sample["image"].save(image_path)
                
                # Write annotation
                f.write(f"{sample['id']} {sample['text']}\n")
                
                if i % 100 == 0:
                    print(f"Processed {i}/{len(iam['train'])} IAM samples")
    else:
        print("Please download the IAM dataset manually and place it in datasets/iam/")
else:
    print("IAM dataset already exists")

# Generate synthetic data for augmentation
if not os.path.exists("datasets/synthetic/synthetic_00000.png"):
    print("Generating synthetic data...")
    generator = SyntheticTextGenerator(output_dir="datasets/synthetic")
    synthetic_data = generator.generate_dataset(num_samples=1000, width=384, height=96)
    
    # Save annotations
    with open("datasets/synthetic/annotations.json", "w") as f:
        json.dump(synthetic_data, f)
else:
    print("Synthetic data already exists")

# Load TrOCR Model and Processor
# ------------------------------

# Initialize the TrOCR model and processor
model_name = "microsoft/trocr-large-handwritten"
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

# Set special tokens
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"Model loaded successfully on {device}")

# Prepare Training and Validation Datasets
# ---------------------------------------

# Load annotation files
def load_iam_annotations(file_path):
    image_paths = []
    texts = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):  # Skip header lines
                continue
                
            parts = line.strip().split(" ")
            if len(parts) < 9:  # Skip malformed lines
                continue
                
            image_id = parts[0]
            text = " ".join(parts[8:])  # Text content starts from 9th field
            
            image_path = os.path.join(os.path.dirname(file_path), "lines", f"{image_id}.png")
            if os.path.exists(image_path):
                image_paths.append(image_path)
                texts.append(text)
    
    return image_paths, texts

def load_synthetic_annotations(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    
    image_paths = [item["image_path"] for item in data]
    texts = [item["text"] for item in data]
    
    return image_paths, texts

# Load dataset annotations
iam_image_paths, iam_texts = load_iam_annotations("datasets/iam/lines.txt")
synthetic_image_paths, synthetic_texts = load_synthetic_annotations("datasets/synthetic/annotations.json")

print(f"Loaded {len(iam_image_paths)} IAM samples")
print(f"Loaded {len(synthetic_image_paths)} synthetic samples")

# Combine datasets
all_image_paths = iam_image_paths + synthetic_image_paths
all_texts = iam_texts + synthetic_texts

# Shuffle the data
combined = list(zip(all_image_paths, all_texts))
random.shuffle(combined)
all_image_paths, all_texts = zip(*combined)

# Split into train and validation sets
val_size = int(0.1 * len(all_image_paths))  # 10% for validation
train_image_paths = all_image_paths[val_size:]
train_texts = all_texts[val_size:]
val_image_paths = all_image_paths[:val_size]
val_texts = all_texts[:val_size]

print(f"Training set: {len(train_image_paths)} samples")
print(f"Validation set: {len(val_image_paths)} samples")

# Create datasets
train_dataset = HandwritingDataset(
    train_image_paths, train_texts, processor, max_target_length=128, augment=True
)
val_dataset = HandwritingDataset(
    val_image_paths, val_texts, processor, max_target_length=128, augment=False
)

# Create data loaders
train_loader, val_loader = create_dataloaders(
    train_dataset, val_dataset, batch_size=4, num_workers=2
)

# Visualize Some Training Samples
# ------------------------------

# Display some training samples
sample_indices = random.sample(range(len(train_dataset)), 5)
sample_images = [train_image_paths[i] for i in sample_indices]
sample_texts = [train_texts[i] for i in sample_indices]

plot_multiple_samples(sample_images, sample_texts)

# Fine-tune TrOCR Model
# --------------------

# Initialize the trainer
trainer = OCRTrainer(
    model_name_or_path=model_name,
    output_dir="results",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,  # Simulate larger batch size
    num_train_epochs=10,
    warmup_steps=500,
    fp16=torch.cuda.is_available(),  # Use mixed precision if available
    save_steps=500,
    eval_steps=500,
    logging_steps=100
)

# Train the model
model, processor = trainer.train(train_dataset, val_dataset)

# Save the fine-tuned model and processor
model.save_pretrained("results/best_model")
processor.save_pretrained("results/best_model")

# Evaluate the Fine-tuned Model
# ---------------------------

# Evaluate on the validation set
evaluation_results = trainer.evaluate(val_loader)

# Print detailed evaluation report
print_evaluation_report(
    evaluation_results["predictions"],
    evaluation_results["references"],
    num_samples=5
)

# Visualize error distributions
plot_error_distribution(
    evaluation_results["predictions"],
    evaluation_results["references"]
)

# Analyze Performance on Different Handwriting Styles
# -------------------------------------------------

# Get worst-performing samples
from utils.metrics import get_error_analysis

error_analysis = get_error_analysis(
    evaluation_results["predictions"],
    evaluation_results["references"],
    max_errors=10
)

# Display worst samples
worst_indices = [item["index"] for item in error_analysis]
worst_images = [val_image_paths[i] for i in worst_indices]
worst_texts = [val_texts[i] for i in worst_indices]
worst_predictions = [evaluation_results["predictions"][i] for i in worst_indices]

plot_multiple_samples(worst_images, worst_texts, worst_predictions, num_samples=5)

# Test on New Images
# ----------------

# Test prediction on new images
def predict_text(image_path):
    # Load and preprocess image
    image = preprocess_image(image_path)
    
    # Prepare for model
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    
    # Generate prediction
    generated_ids = model.generate(
        pixel_values=pixel_values,
        max_length=128,
        num_beams=4
    )
    
    # Decode prediction
    predicted_text = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return predicted_text

# Test on a few samples from the test set
test_indices = random.sample(range(len(val_dataset)), 3)
for i in test_indices:
    image_path = val_image_paths[i]
    ground_truth = val_texts[i]
    
    # Make prediction
    prediction = predict_text(image_path)
    
    # Display result
    plot_sample(image_path, ground_truth, prediction)

# Prepare for GitHub Repository
# ---------------------------

# Save a summary report
report_text = f"""
# OCR Model Fine-Tuning Report

## Model Information
- Base model: {model_name}
- Dataset size: {len(train_dataset)} training samples, {len(val_dataset)} validation samples

## Results
- Character Error Rate (CER): {evaluation_results['cer']:.4f}
- Word Error Rate (WER): {evaluation_results['wer']:.4f}

## Target Achievement
- CER target (<= 7%): {'Achieved' if evaluation_results['cer'] <= 0.07 else 'Not achieved'}
- WER target (<= 15%): {'Achieved' if evaluation_results['wer'] <= 0.15 else 'Not achieved'}

## Training Details
- Learning rate: 5e-5
- Batch size: 4 (effective batch size: 8 with gradient accumulation)
- Number of epochs: 10
- Hardware: {device}
"""

with open("report.md", "w") as f:
    f.write(report_text)

print("Report saved to report.md")

if IN_COLAB:
    # Setup Git in Colab
    subprocess.run(["git", "config", "--global", "user.email", "your.email@example.com"], check=True)
    subprocess.run(["git", "config", "--global", "user.name", "Your Name"], check=True)
    
    # Add new files to git
    subprocess.run(["git", "add", "."], check=True)
    
    # Commit changes
    subprocess.run(["git", "commit", "-m", "Add OCR model fine-tuning implementation"], check=True)
    
    # Push to GitHub
    # Note: You'll need to provide your GitHub credentials
    print("To push to GitHub, use this command in Colab:")
    print("!git push origin main")

# Conclusion
# ---------
print("""
In this script, we have:
1. Set up the TrOCR model for handwriting recognition
2. Prepared training and validation datasets
3. Fine-tuned the model on the combined dataset
4. Evaluated the model against our target metrics
5. Analyzed the performance on different handwriting styles

The fine-tuned model can now be used as part of a document digitization pipeline for recognizing handwritten text with high accuracy.
""") 