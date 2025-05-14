import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    EarlyStoppingCallback
)
from typing import Dict, List, Optional, Union
import logging
import numpy as np
from tqdm.auto import tqdm
from .metrics import calculate_cer, calculate_wer

class OCRTrainer:
    """
    Trainer class for OCR model fine-tuning.
    """
    def __init__(
        self,
        model_name_or_path: str = "microsoft/trocr-large-handwritten",
        output_dir: str = "./results",
        learning_rate: float = 5e-5,
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 2,
        num_train_epochs: int = 10,
        warmup_steps: int = 500,
        fp16: bool = True,
        seed: int = 42,
        save_steps: int = 500,
        eval_steps: int = 500,
        logging_steps: int = 100,
        max_length: int = 128
    ):
        """
        Initialize the trainer.
        
        Args:
            model_name_or_path: Name or path of the pretrained model
            output_dir: Directory to save the fine-tuned model
            learning_rate: Learning rate for training
            per_device_train_batch_size: Batch size per device for training
            gradient_accumulation_steps: Number of steps to accumulate gradients
            num_train_epochs: Number of training epochs
            warmup_steps: Number of warmup steps for learning rate scheduler
            fp16: Whether to use mixed precision training
            seed: Random seed for reproducibility
            save_steps: Number of steps to save the model
            eval_steps: Number of steps to evaluate the model
            logging_steps: Number of steps to log training progress
            max_length: Maximum length of the target sequence
        """
        self.model_name_or_path = model_name_or_path
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_train_epochs = num_train_epochs
        self.warmup_steps = warmup_steps
        self.fp16 = fp16
        self.seed = seed
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps
        self.max_length = max_length
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO
        )
        
        # Load model and processor
        self.load_model_and_processor()
        
    def load_model_and_processor(self):
        """Load the pretrained model and processor."""
        self.processor = TrOCRProcessor.from_pretrained(self.model_name_or_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name_or_path)
        
        # Set special tokens
        self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model.config.vocab_size = self.model.config.decoder.vocab_size
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        print(f"Model loaded and moved to {self.device}")
    
    def train(self, train_dataset, val_dataset):
        """
        Fine-tune the model using the Hugging Face Seq2SeqTrainer.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            
        Returns:
            Fine-tuned model
        """
        # Define training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            num_train_epochs=self.num_train_epochs,
            warmup_steps=self.warmup_steps,
            fp16=self.fp16,
            logging_dir=os.path.join(self.output_dir, "logs"),
            logging_steps=self.logging_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=self.eval_steps,
            save_steps=self.save_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="cer",
            greater_is_better=False,
            seed=self.seed,
            predict_with_generate=True,
            generation_max_length=self.max_length,
            generation_num_beams=4
        )
        
        # Define compute metrics function
        def compute_metrics(pred):
            labels_ids = pred.label_ids
            pred_ids = pred.predictions
            
            # Replace -100 with pad token id
            labels_ids[labels_ids == -100] = self.processor.tokenizer.pad_token_id
            
            # Decode predictions and labels
            pred_str = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            labels_ids[labels_ids == self.processor.tokenizer.pad_token_id] = -100
            label_str = self.processor.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
            
            # Calculate metrics
            cer = calculate_cer(pred_str, label_str)
            wer = calculate_wer(pred_str, label_str)
            
            return {
                "cer": cer,
                "wer": wer
            }
        
        # Create trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            data_collator=default_data_collator,
            tokenizer=self.processor.tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
        )
        
        # Train model
        trainer.train()
        
        # Save the fine-tuned model and processor
        trainer.save_model(os.path.join(self.output_dir, "best_model"))
        self.processor.save_pretrained(os.path.join(self.output_dir, "best_model"))
        
        return self.model, self.processor
    
    def manual_train(self, train_loader, val_loader, num_epochs=None):
        """
        Manually train the model using PyTorch for more control.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs (overrides self.num_train_epochs if provided)
            
        Returns:
            Fine-tuned model
        """
        if num_epochs is None:
            num_epochs = self.num_train_epochs
            
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        
        # Setup scaler for mixed precision training
        scaler = torch.cuda.amp.GradScaler() if self.fp16 else None
        
        # Training loop
        best_cer = float('inf')
        best_model_state = None
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_iterator = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}")
            
            for batch in train_iterator:
                # Move batch to device
                pixel_values = batch["pixel_values"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Forward pass
                if self.fp16:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(pixel_values=pixel_values, labels=labels)
                        loss = outputs.loss
                else:
                    outputs = self.model(pixel_values=pixel_values, labels=labels)
                    loss = outputs.loss
                
                # Scale loss
                if self.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Update weights
                if self.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad()
                
                train_loss += loss.item()
                train_iterator.set_postfix({"loss": loss.item()})
            
            avg_train_loss = train_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {avg_train_loss:.4f}")
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            predictions = []
            references = []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"):
                    # Move batch to device
                    pixel_values = batch["pixel_values"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    texts = batch["text"]
                    
                    # Forward pass
                    outputs = self.model(pixel_values=pixel_values, labels=labels)
                    loss = outputs.loss
                    
                    # Generate predictions
                    generated_ids = self.model.generate(
                        pixel_values=pixel_values,
                        max_length=self.max_length,
                        num_beams=4
                    )
                    
                    # Decode predictions
                    pred_texts = self.processor.tokenizer.batch_decode(
                        generated_ids, skip_special_tokens=True
                    )
                    
                    predictions.extend(pred_texts)
                    references.extend(texts)
                    
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            cer = calculate_cer(predictions, references)
            wer = calculate_wer(predictions, references)
            
            print(f"Epoch {epoch+1}/{num_epochs} - Validation Loss: {avg_val_loss:.4f}, CER: {cer:.4f}, WER: {wer:.4f}")
            
            # Save the best model
            if cer < best_cer:
                best_cer = cer
                best_model_state = self.model.state_dict().copy()
                
                # Save the best model
                os.makedirs(os.path.join(self.output_dir, "best_model"), exist_ok=True)
                torch.save(best_model_state, os.path.join(self.output_dir, "best_model", "pytorch_model.bin"))
                self.processor.save_pretrained(os.path.join(self.output_dir, "best_model"))
                
                print(f"New best model saved with CER: {best_cer:.4f}")
        
        # Load the best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            
        return self.model, self.processor
    
    def evaluate(self, test_loader):
        """
        Evaluate the model on a test set.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()
        predictions = []
        references = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                # Move batch to device
                pixel_values = batch["pixel_values"].to(self.device)
                texts = batch["text"]
                
                # Generate predictions
                generated_ids = self.model.generate(
                    pixel_values=pixel_values,
                    max_length=self.max_length,
                    num_beams=4
                )
                
                # Decode predictions
                pred_texts = self.processor.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                
                predictions.extend(pred_texts)
                references.extend(texts)
        
        # Calculate metrics
        cer = calculate_cer(predictions, references)
        wer = calculate_wer(predictions, references)
        
        # Print metrics
        print(f"Test CER: {cer:.4f}")
        print(f"Test WER: {wer:.4f}")
        
        return {
            "cer": cer,
            "wer": wer,
            "predictions": predictions,
            "references": references
        }
    
    def predict(self, image):
        """
        Predict text from a single image.
        
        Args:
            image: PIL Image object
            
        Returns:
            Predicted text
        """
        self.model.eval()
        
        with torch.no_grad():
            # Preprocess image
            pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
            
            # Generate prediction
            generated_ids = self.model.generate(
                pixel_values=pixel_values,
                max_length=self.max_length,
                num_beams=4
            )
            
            # Decode prediction
            pred_text = self.processor.tokenizer.decode(
                generated_ids[0], skip_special_tokens=True
            )
            
        return pred_text 