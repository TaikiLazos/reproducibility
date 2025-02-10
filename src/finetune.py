import torch
from transformers import RobertaTokenizerFast, RobertaForTokenClassification
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import json
from typing import List, Dict, Tuple

class JargonDataset(Dataset):
    def __init__(self, abstracts_dir: str, annotations_path: str, tokenizer):
        self.tokenizer = tokenizer
        self.examples = self._prepare_data(abstracts_dir, annotations_path)
    
    def _prepare_data(self, abstracts_dir: str, annotations_path: str) -> List[Dict]:
        examples = []
        max_length = 512  # Maximum sequence length
        
        # Load annotations
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)
        
        # Process each abstract
        for abstract_id, jargons in annotations.items():
            try:
                with open(f"{abstracts_dir}/{abstract_id}.src.txt", 'r') as f:
                    text = f.read()
            except FileNotFoundError:
                continue
            
            # Get jargon terms and their positions
            jargon_terms = list(jargons.keys())
            
            # Tokenize the full text first with padding
            tokenized = self.tokenizer(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_offsets_mapping=True,
                return_tensors='pt'
            )
            
            # Create label array (same length as input_ids)
            labels = torch.zeros_like(tokenized['input_ids'][0])
            offset_mapping = tokenized['offset_mapping'][0].tolist()
            
            # Mark jargons using character offsets
            for jargon in jargon_terms:
                jargon_lower = jargon.lower()
                text_lower = text.lower()
                start_idx = 0
                
                while True:
                    pos = text_lower.find(jargon_lower, start_idx)
                    if pos == -1:
                        break
                        
                    # Find tokens that overlap with this jargon span
                    jargon_end = pos + len(jargon)
                    for idx, (token_start, token_end) in enumerate(offset_mapping):
                        # Skip special tokens and padding
                        if token_start == token_end:
                            continue
                            
                        # Check if token overlaps with jargon
                        if (token_start < jargon_end and token_end > pos):
                            labels[idx] = 1
                    
                    start_idx = pos + 1
            
            # Set labels for padding tokens to -100 (ignored in loss calculation)
            attention_mask = tokenized['attention_mask'][0]
            labels[attention_mask == 0] = -100
            
            examples.append({
                'input_ids': tokenized['input_ids'][0],
                'attention_mask': tokenized['attention_mask'][0],
                'labels': labels,
                'offset_mapping': offset_mapping
            })
            
            # Debug info for first example
            if len(examples) == 1:
                print("\nFirst example processed:")
                print("Text:", text[:100], "...")
                print("Jargons:", jargon_terms[:3], "...")
                print("Number of tokens:", len(labels))
                print("Number of jargon tokens:", torch.sum(labels == 1).item())
                print("Sequence length:", len(tokenized['input_ids'][0]))
        
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def train(model, train_dataloader, val_dataloader, device, num_epochs=20):
    # Use a learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )
    best_f1 = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        
        for batch in train_dataloader:
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                labels=batch['labels'].to(device)
            )
            
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Add gradient clipping
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Validation
        val_metrics = evaluate(model, val_dataloader, device)
        print(f"Validation F1: {val_metrics['f1']:.2f}")
        
        # Update learning rate
        scheduler.step(val_metrics['f1'])
        
        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save(model.state_dict(), 'output/jargon_model_finetuned.pt')
            print("New best model saved!")

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

def evaluate(model, dataloader, device):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device)
            )
            
            # Get predictions
            predictions = torch.argmax(outputs.logits, dim=2)
            labels = batch['labels']
            
            # Convert predictions to spans
            for pred, label, offsets in zip(predictions, labels, batch['offset_mapping']):
                pred_spans = []  # List of (start, end) tuples
                true_spans = []
                
                # Extract predicted spans
                current_start = None
                for idx, (p, (start, end)) in enumerate(zip(pred, offsets)):
                    if p == 1 and current_start is None:
                        current_start = start
                    elif p == 0 and current_start is not None:
                        pred_spans.append((current_start, end))
                        current_start = None
                
                # Extract true spans
                current_start = None
                for idx, (l, (start, end)) in enumerate(zip(label, offsets)):
                    if l == 1 and current_start is None:
                        current_start = start
                    elif l == 0 and current_start is not None:
                        true_spans.append((current_start, end))
                        current_start = None
                
                # Match spans with ≥75% overlap
                matched_pred = set()
                matched_true = set()
                
                for i, pred_span in enumerate(pred_spans):
                    for j, true_span in enumerate(true_spans):
                        if j not in matched_true:
                            overlap = calculate_overlap(
                                pred_span[0], pred_span[1],
                                true_span[0], true_span[1]
                            )
                            if overlap >= 0.75:
                                true_positives += 1
                                matched_pred.add(i)
                                matched_true.add(j)
                                break
                
                # Count unmatched predictions as false positives
                false_positives += len(pred_spans) - len(matched_pred)
                
                # Count unmatched true spans as false negatives
                false_negatives += len(true_spans) - len(matched_true)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {'precision': precision * 100, 'recall': recall * 100, 'f1': f1 * 100}

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "output/RoBERTa_base_binary.pt"
    abstracts_dir = "data/PLABA_2024-Task_1/abstracts"
    annotations_path = "data/PLABA_2024-Task_1/train.json"
    
    # Initialize model and tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', add_prefix_space=True)
    model = RobertaForTokenClassification.from_pretrained(
        'roberta-base',
        num_labels=2,
        hidden_dropout_prob=0.2,  # Add dropout
        attention_probs_dropout_prob=0.2
    )
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    
    # Create dataset and dataloaders
    dataset = JargonDataset(abstracts_dir, annotations_path, tokenizer)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8)
    
    train(model, train_dataloader, val_dataloader, device)

if __name__ == "__main__":
    main()