# Google Colab Guide for OCR Fine-Tuning

This guide provides step-by-step instructions for running our handwriting OCR model fine-tuning in Google Colab.

## 1. Set Up Google Colab

1. Open Google Colab: https://colab.research.google.com/
2. Create a new notebook
3. Make sure you have GPU acceleration enabled:
   - Go to `Runtime` > `Change runtime type`
   - Set Hardware accelerator to `GPU`
   - Click `Save`

## 2. Clone the Repository

Run the following code in a Colab cell:

```python
# Clone the repository
!git clone https://github.com/palaksingla2004/OCR.git
%cd OCR

# Install dependencies
!pip install -q -r requirements.txt

# Clone TextRecognitionDataGenerator for synthetic data (optional)
!git clone https://github.com/Belval/TextRecognitionDataGenerator.git
!cd TextRecognitionDataGenerator && pip install -e .
```

## 3. Download Datasets

You have several options for obtaining the datasets:

### Option 1: Use Hugging Face Datasets

```python
# Install datasets library if not already installed
!pip install -q datasets

# Download IAM dataset
from datasets import load_dataset
iam = load_dataset("iamdb", "lines")

# Save to disk
import os
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
```

### Option 2: Use Google Drive

If you have the datasets saved in your Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')

# Copy IAM dataset from your Drive
!mkdir -p datasets/iam
!cp -r /content/drive/MyDrive/PATH_TO_YOUR_IAM_DATASET/* datasets/iam/
```

## 4. Generate Synthetic Data

```python
from data.synthetic_generator import SyntheticTextGenerator

# Create synthetic data
os.makedirs("datasets/synthetic", exist_ok=True)
generator = SyntheticTextGenerator(output_dir="datasets/synthetic")
synthetic_data = generator.generate_dataset(num_samples=1000, width=384, height=96)

# Save annotations
import json
with open("datasets/synthetic/annotations.json", "w") as f:
    json.dump(synthetic_data, f)
```

## 5. Run the Training Script

You can run the entire training script:

```python
# Run the training script
%run colab_notebook.py
```

Or run individual steps:

```python
# Import needed modules
from utils.trainer import OCRTrainer
from data.data_loader import HandwritingDataset, create_dataloaders
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

# Load model and processor
model_name = "microsoft/trocr-large-handwritten"
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

# Customize the training procedure
trainer = OCRTrainer(
    model_name_or_path=model_name,
    output_dir="results",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=10,
    fp16=True
)

# Start training
model, processor = trainer.train(train_dataset, val_dataset)
```

## 6. Saving Your Fine-Tuned Model

After training, you can save the model to Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')

# Create a directory in your Drive
!mkdir -p /content/drive/MyDrive/ocr_model

# Copy the model files
!cp -r results/best_model/* /content/drive/MyDrive/ocr_model/
```

## 7. Testing Your Model

Test your model on individual images:

```python
from utils.preprocessing import preprocess_image
from utils.visualization import plot_sample

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

# Test on an image
image_path = "path/to/your/test/image.png"
prediction = predict_text(image_path)
plot_sample(image_path, "Unknown", prediction)
```

## 8. Troubleshooting

### CUDA Out of Memory
If you encounter CUDA out of memory errors:
- Reduce the batch size (`per_device_train_batch_size`)
- Increase gradient accumulation steps (`gradient_accumulation_steps`)
- Use a smaller model like `microsoft/trocr-base-handwritten`

### Slow Training
For faster training:
- Enable mixed precision training (`fp16=True`)
- Reduce the number of training epochs for testing
- Use a subset of the dataset for initial experiments

### Image Loading Issues
If you encounter issues loading images:
- Make sure the path format is correct
- Check if the images are in a supported format (PNG, JPEG)
- Verify that the images exist at the specified path 