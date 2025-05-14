import jiwer
from typing import List, Dict, Union, Tuple
import numpy as np

def calculate_wer(predictions: List[str], references: List[str]) -> float:
    """
    Calculate Word Error Rate (WER) between predictions and references.
    
    Args:
        predictions: List of predicted text strings
        references: List of reference (ground truth) text strings
    
    Returns:
        WER value (lower is better)
    """
    return jiwer.wer(references, predictions)

def calculate_cer(predictions: List[str], references: List[str]) -> float:
    """
    Calculate Character Error Rate (CER) between predictions and references.
    
    Args:
        predictions: List of predicted text strings
        references: List of reference (ground truth) text strings
    
    Returns:
        CER value (lower is better)
    """
    return jiwer.cer(references, predictions)

def evaluate_batch(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Calculate multiple metrics for a batch of predictions.
    
    Args:
        predictions: List of predicted text strings
        references: List of reference (ground truth) text strings
    
    Returns:
        Dictionary containing WER and CER values
    """
    results = {
        "wer": calculate_wer(predictions, references),
        "cer": calculate_cer(predictions, references)
    }
    return results

def print_evaluation_report(predictions: List[str], references: List[str], 
                           num_samples: int = 5) -> None:
    """
    Print a detailed evaluation report including metrics and example comparisons.
    
    Args:
        predictions: List of predicted text strings
        references: List of reference (ground truth) text strings
        num_samples: Number of example comparisons to show
    """
    metrics = evaluate_batch(predictions, references)
    
    print(f"=== Evaluation Report ===")
    print(f"WER: {metrics['wer']:.4f}")
    print(f"CER: {metrics['cer']:.4f}")
    print(f"Number of samples: {len(predictions)}")
    
    if metrics['wer'] <= 0.15:
        print("✅ WER target achieved (≤ 15%)")
    else:
        print("❌ WER target not achieved (> 15%)")
        
    if metrics['cer'] <= 0.07:
        print("✅ CER target achieved (≤ 7%)")
    else:
        print("❌ CER target not achieved (> 7%)")
    
    print("\n=== Example Comparisons ===")
    num_samples = min(num_samples, len(predictions))
    indices = np.random.choice(len(predictions), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        print(f"Example {i+1}:")
        print(f"Reference: \"{references[idx]}\"")
        print(f"Prediction: \"{predictions[idx]}\"")
        
        # Calculate individual metrics
        individual_wer = calculate_wer([predictions[idx]], [references[idx]])
        individual_cer = calculate_cer([predictions[idx]], [references[idx]])
        print(f"Individual WER: {individual_wer:.4f}, CER: {individual_cer:.4f}")
        print("-" * 50)

def get_error_analysis(predictions: List[str], references: List[str], 
                     max_errors: int = 10) -> List[Dict[str, Union[str, float]]]:
    """
    Get detailed error analysis for the worst performing samples.
    
    Args:
        predictions: List of predicted text strings
        references: List of reference (ground truth) text strings
        max_errors: Maximum number of worst samples to analyze
    
    Returns:
        List of dictionaries containing error analysis
    """
    individual_errors = []
    
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        wer = calculate_wer([pred], [ref])
        cer = calculate_cer([pred], [ref])
        
        individual_errors.append({
            "index": i,
            "reference": ref,
            "prediction": pred,
            "wer": wer,
            "cer": cer
        })
    
    # Sort by CER in descending order
    individual_errors.sort(key=lambda x: x["cer"], reverse=True)
    
    return individual_errors[:max_errors] 