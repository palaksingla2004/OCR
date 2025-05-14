#!/usr/bin/env python
# coding: utf-8

"""
# Run OCR Model Fine-Tuning in Colab

This script clones the OCR repository and runs the training process in Google Colab.
"""

# Setup and Dependencies
import os
import subprocess
from IPython.display import display, Markdown

print("Setting up the OCR environment...")

# Clone the repository
subprocess.run(["git", "clone", "https://github.com/palaksingla2004/OCR.git"], check=True)
os.chdir("OCR")

# Install dependencies
print("Installing dependencies...")
subprocess.run(["pip", "install", "-q", "-r", "requirements.txt"], check=True)

# Clone TextRecognitionDataGenerator for synthetic data (optional)
print("Setting up TextRecognitionDataGenerator...")
subprocess.run(["git", "clone", "https://github.com/Belval/TextRecognitionDataGenerator.git"], check=True)
os.chdir("TextRecognitionDataGenerator")
subprocess.run(["pip", "install", "-e", "."], check=True)
os.chdir("..")

# Display success message
display(Markdown("""
## Environment Setup Complete!

You can now run the OCR training script by executing the following command:

```python
%run colab_notebook.py
```

Or you can follow the steps in `COLAB_GUIDE.md` to run the training process step by step.
"""))

# Option to mount Google Drive
display(Markdown("""
## Mount Google Drive (Optional)

If you have datasets stored in Google Drive, you can mount it by running:

```python
from google.colab import drive
drive.mount('/content/drive')
```
"""))

# Display available commands
display(Markdown("""
## Available Commands

Here are some useful commands to run after setup:

1. **Run the full training pipeline**:
   ```python
   %run colab_notebook.py
   ```

2. **Manual dataset download using Hugging Face**:
   ```python
   from datasets import load_dataset
   iam = load_dataset("iamdb", "lines")
   # Create directories
   os.makedirs("datasets/iam/lines", exist_ok=True)
   # Save samples
   with open("datasets/iam/lines.txt", "w") as f:
       for i, sample in enumerate(iam["train"]):
           image_path = f"datasets/iam/lines/{sample['id']}.png"
           sample["image"].save(image_path)
           f.write(f"{sample['id']} {sample['text']}\\n")
   ```

3. **Generate synthetic data**:
   ```python
   from data.synthetic_generator import SyntheticTextGenerator
   generator = SyntheticTextGenerator(output_dir="datasets/synthetic")
   synthetic_data = generator.generate_dataset(num_samples=1000)
   ```

4. **Test the model after training**:
   ```python
   from utils.preprocessing import preprocess_image
   from utils.visualization import plot_sample
   from transformers import TrOCRProcessor, VisionEncoderDecoderModel
   import torch
   
   # Load model and processor
   model_path = "results/best_model"
   processor = TrOCRProcessor.from_pretrained(model_path)
   model = VisionEncoderDecoderModel.from_pretrained(model_path)
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model.to(device)
   
   # Predict text from image
   def predict_text(image_path):
       image = preprocess_image(image_path)
       pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
       generated_ids = model.generate(pixel_values=pixel_values, max_length=128, num_beams=4)
       predicted_text = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
       return predicted_text
   
   # Example usage
   image_path = "path/to/image.png"  # Replace with your image path
   prediction = predict_text(image_path)
   plot_sample(image_path, "Unknown", prediction)
   ```
"""))

print("\nSetup complete! You can now run the training script with %run colab_notebook.py") 