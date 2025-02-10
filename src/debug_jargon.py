import torch
from transformers import RobertaTokenizerFast, RobertaForTokenClassification
import json
from typing import List, Dict, Set, Tuple

def analyze_predictions(text: str, true_jargons: Set[str], predicted_jargons: List[Tuple[str, int, int]]):
    """Analyze predictions vs ground truth for one abstract"""
    pred_terms = set(term.lower() for term, _, _ in predicted_jargons)
    
    # Find matches and mismatches
    correct = true_jargons.intersection(pred_terms)
    missed = true_jargons - pred_terms
    extra = pred_terms - true_jargons
    
    print("\n=== Analysis ===")
    print(f"Text snippet: {text[:200]}...")
    print(f"\nTrue Jargons ({len(true_jargons)}):")
    for j in true_jargons:
        print(f"  - {j}")
    
    print(f"\nCorrectly Identified ({len(correct)}):")
    for j in correct:
        print(f"  ✓ {j}")
    
    print(f"\nMissed Jargons ({len(missed)}):")
    for j in missed:
        print(f"  × {j}")
    
    print(f"\nFalse Positives ({len(extra)}):")
    for j in extra:
        print(f"  ! {j}")
    
    return len(correct), len(missed), len(extra)

def debug_model(model_path: str, abstracts_dir: str, annotations_path: str, num_samples: int = 5):
    """Debug model predictions on a few samples"""
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', add_prefix_space=True)
    model = RobertaForTokenClassification.from_pretrained('roberta-base', num_labels=2)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    # Load data
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    total_correct = 0
    total_missed = 0
    total_extra = 0
    
    # Process a few samples
    for i, (abstract_id, jargons) in enumerate(list(annotations.items())[:num_samples]):
        print(f"\n\n{'='*50}")
        print(f"Abstract {i+1}: {abstract_id}")
        
        # Read abstract
        with open(f"{abstracts_dir}/{abstract_id}.src.txt", 'r') as f:
            text = f.read()
        
        # Get ground truth jargons
        true_jargons = set(j.lower() for j in jargons.keys())
        
        # Get predictions
        inputs = tokenizer(
            text,
            add_special_tokens=True,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device)
            )
            predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().tolist()
        
        # Extract predicted jargons
        predicted_jargons = []
        current_jargon = []
        current_start = None
        offset_mapping = inputs["offset_mapping"][0].tolist()
        
        for idx, (pred, (start, end)) in enumerate(zip(predictions[1:-1], offset_mapping[1:-1])):
            if pred == 1:  # Jargon token
                if not current_jargon:
                    current_start = start
                current_jargon.append(text[start:end])
            elif current_jargon:  # End of jargon
                jargon_text = ' '.join(current_jargon)
                predicted_jargons.append((jargon_text, current_start, end))
                current_jargon = []
        
        # Analyze this sample
        correct, missed, extra = analyze_predictions(text, true_jargons, predicted_jargons)
        total_correct += correct
        total_missed += missed
        total_extra += extra
    
    print("\n=== Overall Statistics ===")
    print(f"Total Correct: {total_correct}")
    print(f"Total Missed: {total_missed}")
    print(f"Total False Positives: {total_extra}")
    precision = total_correct / (total_correct + total_extra) if (total_correct + total_extra) > 0 else 0
    recall = total_correct / (total_correct + total_missed) if (total_correct + total_missed) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print(f"\nPrecision: {precision*100:.2f}%")
    print(f"Recall: {recall*100:.2f}%")
    print(f"F1 Score: {f1*100:.2f}%")

if __name__ == "__main__":
    debug_model(
        model_path="output/RoBERTa_base_binary.pt",
        abstracts_dir="data/PLABA_2024-Task_1/abstracts",
        annotations_path="data/PLABA_2024-Task_1/train.json",
        num_samples=5
    ) 