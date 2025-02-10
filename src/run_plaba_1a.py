import torch
from transformers import RobertaTokenizerFast, RobertaForTokenClassification
from typing import List, Tuple, Dict
import json
from sklearn.metrics import precision_recall_fscore_support
from inference_jargon import JargonDetector

def load_abstracts(abstracts_dir: str, ground_truth_path: str) -> Dict[str, str]:
    """
    Load all abstract texts
    """
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)
    
    abstracts = {}
    for abstract_id in ground_truth.keys():
        with open(f"{abstracts_dir}/{abstract_id}.src.txt", 'r') as f:
            abstracts[abstract_id] = f.read()
    
    return abstracts

def calculate_overlap(start1, end1, start2, end2):
    """Calculate the overlap percentage between two spans"""
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    
    if overlap_end <= overlap_start:
        return 0.0
    
    overlap_length = overlap_end - overlap_start
    span1_length = end1 - start1
    span2_length = end2 - start2
    
    # Calculate overlap percentage relative to both spans
    overlap_pct1 = overlap_length / span1_length
    overlap_pct2 = overlap_length / span2_length
    
    return min(overlap_pct1, overlap_pct2)

def evaluate_jargon_detection(model_path: str, abstracts_dir: str, ground_truth_path: str):
    """
    Evaluate jargon detection model on the dataset using 75% overlap criterion
    """
    detector = JargonDetector(model_path)
    
    # Load data
    abstracts = load_abstracts(abstracts_dir, ground_truth_path)
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)
    
    # Track metrics
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # Process each abstract
    for abstract_id, text in abstracts.items():
        print(f"\nProcessing abstract: {abstract_id}")
        
        # Get ground truth spans
        true_jargons = []
        for jargon in ground_truth[abstract_id].keys():
            pos = 0
            while True:
                idx = text.lower().find(jargon.lower(), pos)
                if idx == -1:
                    break
                true_jargons.append((jargon, idx, idx + len(jargon)))
                pos = idx + 1
        
        # Get predicted spans
        predictions = detector.detect_jargons(text)
        
        # Match predictions with ground truth using 75% overlap
        matched_pred = set()
        matched_true = set()
        
        for i, (pred_term, pred_start, pred_end) in enumerate(predictions):
            for j, (true_term, true_start, true_end) in enumerate(true_jargons):
                if j not in matched_true:
                    overlap = calculate_overlap(
                        pred_start, pred_end,
                        true_start, true_end
                    )
                    if overlap >= 0.75:
                        true_positives += 1
                        matched_pred.add(i)
                        matched_true.add(j)
                        break
        
        # Count unmatched as false positives/negatives
        false_positives += len(predictions) - len(matched_pred)
        false_negatives += len(true_jargons) - len(matched_true)
    
    # Calculate final metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'total_abstracts': len(abstracts)
    }

def main():
    # Configuration
    model_path = "output/jargon_model_finetuned.pt"
    abstracts_dir = "data/PLABA_2024-Task_1/abstracts"
    ground_truth_path = "data/PLABA_2024-Task_1/task_1_testing.json"
    
    print("Starting evaluation...")
    print(f"Model: {model_path}")
    print(f"Data directory: {abstracts_dir}")
    
    # Run evaluation
    metrics = evaluate_jargon_detection(model_path, abstracts_dir, ground_truth_path)
    
    # Print results
    print("\n=== Evaluation Results ===")
    print(f"Total abstracts processed: {metrics['total_abstracts']}")
    print(f"Precision: {metrics['precision']:.2f}%")
    print(f"Recall: {metrics['recall']:.2f}%")
    print(f"F1 Score: {metrics['f1']:.2f}%")

if __name__ == "__main__":
    main()