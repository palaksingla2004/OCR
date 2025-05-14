#!/usr/bin/env python
# coding: utf-8

"""
Handwriting OCR Prediction Script

This script loads a fine-tuned TrOCR model and predicts text from handwritten images.
"""

import os
import argparse
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from utils.preprocessing import preprocess_image
from utils.visualization import plot_sample

def predict_text(image_path, model_path="results/best_model", device=None):
    """
    Predict text from a handwritten image.
    
    Args:
        image_path: Path to the image file
        model_path: Path to the saved model
        device: Device to run inference on (None for auto)
        
    Returns:
        Predicted text
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load processor and model
    try:
        processor = TrOCRProcessor.from_pretrained(model_path)
        model = VisionEncoderDecoderModel.from_pretrained(model_path)
        model.to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to pretrained model")
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
        model.to(device)
    
    # Preprocess image
    image = preprocess_image(image_path)
    
    # Predict text
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(
        pixel_values=pixel_values,
        max_length=128,
        num_beams=4
    )
    
    # Decode prediction
    prediction = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return prediction

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Predict text from handwritten images")
    parser.add_argument("--image", "-i", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", "-m", type=str, default="results/best_model", help="Path to model directory")
    parser.add_argument("--visualize", "-v", action="store_true", help="Visualize results")
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image {args.image} not found")
        return
    
    # Make prediction
    prediction = predict_text(args.image, args.model)
    
    # Display results
    print("\nPrediction:", prediction)
    
    # Visualize if requested
    if args.visualize:
        plot_sample(args.image, prediction)

if __name__ == "__main__":
    main() 