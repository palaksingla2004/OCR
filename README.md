# Handwriting Recognition OCR

Fine-tuned Transformer-based OCR model for handwritten text recognition.

## Overview
This project focuses on fine-tuning a state-of-the-art Optical Character Recognition (OCR) model to achieve high accuracy in recognizing handwritten text. The solution leverages modern transformer-based architecture and is trained on diverse handwriting datasets.

## Model
- TrOCR (Transformer-based OCR): A model that combines a Vision Transformer (ViT) encoder with a text Transformer decoder
- Base model: microsoft/trocr-large-handwritten

## Datasets
- IAM Handwriting Database: Contains 13,353 handwritten English text lines from 657 writers
- Imgur5K: A diverse dataset with ~135K handwritten English words across 5K images
- Synthetic data generation for augmentation

## Evaluation Metrics
- Character Error Rate (CER): Target ≤ 7%
- Word Error Rate (WER): Target ≤ 15%

## Project Structure
- `ocr_handwriting_recognition.ipynb`: Main notebook containing the complete pipeline
- `utils/`: Helper functions for preprocessing and evaluation
- `data/`: Data handling and preparation scripts
- `report.pdf`: Detailed report on methodology and results

## Requirements
- PyTorch
- Transformers (Hugging Face)
- OpenCV
- Pillow
- jiwer (for CER/WER calculation)
- datasets (Hugging Face) 