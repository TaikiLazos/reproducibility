import torch
from transformers import RobertaTokenizerFast, RobertaForTokenClassification
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import json
from typing import List, Dict, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
from inference_jargon import JargonDetector
import os
import random

class JargonDataset(Dataset):
    def __init__(self, abstracts_dir: str, annotations_path: str, tokenizer):
        self.tokenizer = tokenizer
        self.examples = self._prepare_data(abstracts_dir, annotations_path)
        
        # Debug: Print label distribution
        label_counts = {-100: 0, 0: 0, 1: 0}
        for example in self.examples:
            for label in example['labels'].tolist():
                if label in label_counts:
                    label_counts[label] += 1
                else:
                    label_counts[label] = 1
        print(f"DEBUG - Label distribution in dataset: {label_counts}")
        
        # Debug: Print a sample example
        if len(self.examples) > 0:
            sample = self.examples[0]
            print("\nDEBUG - Sample example:")
            print(f"Input shape: {sample['input_ids'].shape}")
            print(f"Labels shape: {sample['labels'].shape}")
            print(f"Label values: {set(sample['labels'].tolist())}")
            
            # Print a few tokens with their labels
            tokens = tokenizer.convert_ids_to_tokens(sample['input_ids'][:20])
            labels = sample['labels'][:20].tolist()
            print("\nSample tokens and labels:")
            for t, l in zip(tokens, labels):
                print(f"Token: {t}, Label: {l}")
    
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
            # Using -100 for non-jargon tokens (ignored in loss calculation)
            # and 1 for jargon tokens
            labels = torch.ones_like(tokenized['input_ids'][0]) * -100
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
            
            # Set non-special tokens that aren't jargon to 0
            for idx, (token_start, token_end) in enumerate(offset_mapping):
                if token_start != token_end and labels[idx] == -100:
                    labels[idx] = 0  # Non-jargon token
            
            examples.append({
                'input_ids': tokenized['input_ids'][0],
                'attention_mask': tokenized['attention_mask'][0],
                'labels': labels
            })
        
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

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

def evaluate_jargon_detection(model_path, abstracts_dir, annotations_path):
    """
    Evaluate jargon detection on test set
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model and tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large', add_prefix_space=True)
    model = RobertaForTokenClassification.from_pretrained(
        'roberta-large',
        num_labels=2
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Define punctuation and special characters to exclude
    punctuation_chars = set('.,;:!?()[]{}"-')
    
    # Load test data
    with open(annotations_path, 'r') as f:
        ground_truth = json.load(f)
    
    # Process each abstract
    all_predictions = {}
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    # Print some examples for debugging
    print("\nDEBUG - Sample predictions vs ground truth:")
    sample_count = 0
    
    for abstract_id, true_jargons in tqdm(ground_truth.items(), desc="Evaluating"):
        try:
            with open(f"{abstracts_dir}/{abstract_id}.src.txt", 'r') as f:
                text = f.read()
        except FileNotFoundError:
            continue
        
        # Tokenize text
        inputs = tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Get predictions
        with torch.no_grad():
            outputs = model(
                input_ids=inputs['input_ids'].to(device),
                attention_mask=inputs['attention_mask'].to(device)
            )
        
        # Get predicted jargon spans
        predictions = outputs.logits.argmax(dim=-1)[0].cpu().numpy()
        offset_mapping = inputs['offset_mapping'][0].cpu().numpy()
        attention_mask = inputs['attention_mask'][0].cpu().numpy()
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Extract jargon spans with improved punctuation handling
        predicted_jargons = {}
        i = 0
        while i < len(predictions):
            if predictions[i] == 1 and attention_mask[i] == 1:  # Jargon token and not padding
                # Skip if token is just punctuation
                token_text = tokens[i].replace('Ġ', '')
                if token_text and all(c in punctuation_chars for c in token_text):
                    i += 1
                    continue
                
                start_char = offset_mapping[i][0]
                # Skip special tokens with no character mapping
                if start_char == 0 and offset_mapping[i][1] == 0:
                    i += 1
                    continue
                
                # Find the end of this jargon span
                j = i
                while j < len(predictions) and predictions[j] == 1 and attention_mask[j] == 1:
                    j += 1
                
                # Get the raw span
                if j-1 < len(offset_mapping):
                    end_char = offset_mapping[j-1][1]
                    
                    # Extract the jargon term
                    if start_char < end_char and end_char <= len(text):
                        jargon_term = text[start_char:end_char]
                        
                        # Clean up jargon term
                        jargon_term = jargon_term.strip()
                        
                        # Remove trailing punctuation
                        while jargon_term and jargon_term[-1] in punctuation_chars:
                            jargon_term = jargon_term[:-1].strip()
                        
                        # Remove leading punctuation
                        while jargon_term and jargon_term[0] in punctuation_chars:
                            jargon_term = jargon_term[1:].strip()
                        
                        # Only add if not empty after cleaning and at least 2 chars
                        if jargon_term and len(jargon_term) > 1:
                            predicted_jargons[jargon_term] = 1
                
                i = j
            else:
                i += 1
        
        # Store predictions
        all_predictions[abstract_id] = predicted_jargons
        
        # Calculate metrics for this abstract
        tp = 0
        fp = 0
        fn = 0
        
        # Print some examples for debugging
        if sample_count < 5:
            print(f"\nAbstract ID: {abstract_id}")
            print(f"Text snippet: {text[:200]}...")
            print("Ground truth jargons:", list(true_jargons.keys()))
            print("Predicted jargons:", list(predicted_jargons.keys()))
            sample_count += 1
        
        # Check for true positives and false positives
        for pred_jargon in predicted_jargons:
            found = False
            for true_jargon in true_jargons:
                # Check for 75% overlap
                if _calculate_overlap(pred_jargon, true_jargon) >= 0.75:
                    tp += 1
                    found = True
                    break
            
            if not found:
                fp += 1
        
        # Check for false negatives
        for true_jargon in true_jargons:
            found = False
            for pred_jargon in predicted_jargons:
                if _calculate_overlap(true_jargon, pred_jargon) >= 0.75:
                    found = True
                    break
            
            if not found:
                fn += 1
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    # Calculate overall metrics
    precision = total_tp / (total_tp + total_fp) * 100 if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) * 100 if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'total_abstracts': len(all_predictions),
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def _calculate_overlap(str1, str2):
    """Calculate character-level overlap between two strings"""
    # Simple character-level overlap
    str1 = str1.lower()
    str2 = str2.lower()
    
    # Find the longer and shorter strings
    if len(str1) < len(str2):
        shorter, longer = str1, str2
    else:
        shorter, longer = str2, str1
    
    # Check if shorter is substring of longer
    if shorter in longer:
        return len(shorter) / len(longer)
    
    # Otherwise calculate character overlap
    common_chars = sum(1 for c in shorter if c in longer)
    return common_chars / max(len(shorter), len(longer))

def evaluate(model, dataloader, device, tokenizer):
    """
    Evaluate model performance using 75% overlap criterion
    """
    model.eval()
    all_predictions = []  # List to store predicted spans
    all_labels = []      # List to store true spans
    
    # Debug counters
    debug_total_tokens = 0
    debug_predicted_jargon_tokens = 0
    debug_actual_jargon_tokens = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Debug: Print shapes of first batch
            if batch_idx == 0:
                print(f"\nDEBUG - Batch shapes:")
                print(f"input_ids: {batch['input_ids'].shape}")
                print(f"attention_mask: {batch['attention_mask'].shape}")
                print(f"labels: {batch['labels'].shape}")
                print(f"Label values in batch: {set(batch['labels'].flatten().tolist())}")
            
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device)
            )
            
            # Debug: Print model output shape
            if batch_idx == 0:
                print(f"Model output logits shape: {outputs.logits.shape}")
                print(f"Model output logits range: [{outputs.logits.min().item():.4f}, {outputs.logits.max().item():.4f}]")
            
            # Get predictions based on model output format
            if outputs.logits.shape[-1] == 1:
                # Binary case with single output
                predictions = (torch.sigmoid(outputs.logits) > 0.3).squeeze(-1)
                print("DEBUG - Using sigmoid for single output dimension")
            else:
                # Multi-class case
                predictions = outputs.logits.argmax(dim=-1)
                print("DEBUG - Using argmax for multiple output dimensions")
            
            # Debug: Print prediction shape and values
            if batch_idx == 0:
                print(f"Predictions shape: {predictions.shape}")
                print(f"Prediction values: {set(predictions.flatten().cpu().tolist())}")
            
            # Process each sequence in batch
            for seq_idx, (pred_seq, label_seq, mask, input_ids) in enumerate(zip(
                predictions, batch['labels'], batch['attention_mask'], batch['input_ids']
            )):
                length = mask.sum().item()
                
                # Convert predictions to list based on model output format
                if outputs.logits.shape[-1] == 1:
                    pred_seq = pred_seq[:length].cpu().tolist()  # Binary case
                else:
                    pred_seq = pred_seq[:length].cpu().tolist()  # Multi-class case
                
                label_seq = label_seq[:length].cpu().tolist()
                
                # Debug counters
                debug_total_tokens += length
                debug_predicted_jargon_tokens += sum(1 for p in pred_seq if p == 1 or p == True)
                debug_actual_jargon_tokens += sum(1 for l in label_seq if l == 1)
                
                # Print detailed debug for first sequence of first batch
                if batch_idx == 0 and seq_idx == 0:
                    print("\nDEBUG - First sequence predictions vs labels:")
                    tokens = tokenizer.convert_ids_to_tokens(input_ids[:length])
                    for i, (t, p, l) in enumerate(zip(tokens[:30], pred_seq[:30], label_seq[:30])):
                        if l != -100:  # Skip special tokens
                            print(f"{i}: Token: {t}, Pred: {p}, Label: {l}")
                
                # Extract spans for predictions
                pred_spans = []
                start_idx = None
                for idx, pred in enumerate(pred_seq):
                    if (pred == 1 or pred == True) and start_idx is None:  # Start of jargon
                        start_idx = idx
                    elif (pred != 1 and pred != True or idx == len(pred_seq) - 1) and start_idx is not None:  # End of jargon
                        end_idx = idx if (pred != 1 and pred != True) else idx + 1
                        pred_spans.append((start_idx, end_idx))
                        start_idx = None
                
                # Extract spans for true labels
                true_spans = []
                start_idx = None
                for idx, true in enumerate(label_seq):
                    if true == 1 and start_idx is None:  # Start of jargon
                        start_idx = idx
                    elif (true != 1 or idx == len(label_seq) - 1) and start_idx is not None:  # End of jargon
                        end_idx = idx if true != 1 else idx + 1
                        true_spans.append((start_idx, end_idx))
                        start_idx = None
                
                all_predictions.append(pred_spans)
                all_labels.append(true_spans)
    
    # Debug: Print token statistics
    print(f"\nDEBUG - Token statistics:")
    print(f"Total tokens: {debug_total_tokens}")
    print(f"Predicted jargon tokens: {debug_predicted_jargon_tokens} ({debug_predicted_jargon_tokens/debug_total_tokens*100:.2f}%)")
    print(f"Actual jargon tokens: {debug_actual_jargon_tokens} ({debug_actual_jargon_tokens/debug_total_tokens*100:.2f}%)")
    
    # Debug: Print span statistics
    total_pred_spans = sum(len(spans) for spans in all_predictions)
    total_true_spans = sum(len(spans) for spans in all_labels)
    print(f"Total predicted spans: {total_pred_spans}")
    print(f"Total true spans: {total_true_spans}")
    
    # Calculate metrics using 75% overlap criterion
    tp = 0  # True positives
    fp = 0  # False positives
    fn = 0  # False negatives
    
    for pred_spans, true_spans in zip(all_predictions, all_labels):
        matched_true = set()
        
        for pred_span in pred_spans:
            found_match = False
            for i, true_span in enumerate(true_spans):
                if i not in matched_true:
                    # Calculate overlap
                    overlap_start = max(pred_span[0], true_span[0])
                    overlap_end = min(pred_span[1], true_span[1])
                    
                    if overlap_end > overlap_start:  # There is overlap
                        overlap_size = overlap_end - overlap_start
                        pred_size = pred_span[1] - pred_span[0]
                        true_size = true_span[1] - true_span[0]
                        
                        overlap_ratio = overlap_size / max(pred_size, true_size)
                        if overlap_ratio >= 0.75:  # 75% overlap threshold
                            tp += 1
                            matched_true.add(i)
                            found_match = True
                            break
            
            if not found_match:
                fp += 1
        
        # Count unmatched true spans as false negatives
        fn += len(true_spans) - len(matched_true)
    
    # Calculate final metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nDEBUG - Detailed metrics:")
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return {
        'f1': f1 * 100,
        'precision': precision * 100,
        'recall': recall * 100
    }

def train(model, train_dataloader, val_dataloader, device, tokenizer, num_epochs=10):
    # Using a lower learning rate for RoBERTa-large
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    # Early stopping and more aggressive LR reduction
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.2, patience=2, verbose=True
    )
    best_f1 = 0
    patience_counter = 0
    max_patience = 5  # Increased patience for early stopping
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Track metrics
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_f1': [],
        'val_precision': [],
        'val_recall': []
    }
    
    # Debug: Print model architecture
    print("\nDEBUG - Model architecture:")
    print(f"Model type: {type(model).__name__}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Main training loop with tqdm
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Debug: Print first batch shapes and values
            if epoch == 0 and batch_idx == 0:
                print("\nDEBUG - First batch:")
                print(f"input_ids shape: {batch['input_ids'].shape}")
                print(f"attention_mask shape: {batch['attention_mask'].shape}")
                print(f"labels shape: {batch['labels'].shape}")
                print(f"Label values: {set(batch['labels'].flatten().tolist())}")
            
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                labels=batch['labels'].to(device)
            )
            
            # Debug: Print loss details for first batch
            if epoch == 0 and batch_idx == 0:
                print(f"Loss: {outputs.loss.item():.4f}")
                if hasattr(outputs, 'logits'):
                    print(f"Logits shape: {outputs.logits.shape}")
                    print(f"Logits range: [{outputs.logits.min().item():.4f}, {outputs.logits.max().item():.4f}]")
            
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_loss / len(train_dataloader)
        history['train_loss'].append(avg_train_loss)
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        
        # Validation
        print("Running validation...")
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                outputs = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    labels=batch['labels'].to(device)
                )
                val_loss += outputs.loss.item()
        
        avg_val_loss = val_loss / len(val_dataloader)
        history['val_loss'].append(avg_val_loss)
        
        val_metrics = evaluate(model, val_dataloader, device, tokenizer)
        history['val_f1'].append(val_metrics['f1'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation F1: {val_metrics['f1']:.2f}")
        print(f"Validation Precision: {val_metrics['precision']:.2f}")
        print(f"Validation Recall: {val_metrics['recall']:.2f}")
        
        # Update learning rate
        scheduler.step(val_metrics['f1'])
        
        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save(model.state_dict(), 'output/jargon_model_roberta_large.pt')
            print("New best model saved!")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Always save the latest model
        torch.save(model.state_dict(), 'output/jargon_model_latest.pt')
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss over time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot metrics
    plt.subplot(1, 2, 2)
    plt.plot(history['val_f1'], label='F1')
    plt.plot(history['val_precision'], label='Precision')
    plt.plot(history['val_recall'], label='Recall')
    plt.title('Validation Metrics over time')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('output/training_history_roberta_large.png')
    plt.close()
    
    return history

def main():
    # Import needed at the top level
    import os
    import random
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configuration
    model_path = "output/jargon_model_roberta_large.pt"
    abstracts_dir = "data/PLABA_2024-Task_1/abstracts"
    train_annotations = "data/PLABA_2024-Task_1/train.json"
    test_annotations = "data/PLABA_2024-Task_1/task_1_testing.json"
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    print("Starting training and evaluation...")
    print(f"Using device: {device}")
    
    # Initialize model and tokenizer with RoBERTa-large
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large', add_prefix_space=True)
    
    # Debug: Print tokenizer info
    print("\nDEBUG - Tokenizer info:")
    print(f"Tokenizer type: {type(tokenizer).__name__}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Model max length: {tokenizer.model_max_length}")
    
    # Create datasets with the correct label format
    print("\nDEBUG - Creating datasets with num_labels=2")
    model = RobertaForTokenClassification.from_pretrained(
        'roberta-large',
        num_labels=2,  # Binary classification: 0=non-jargon, 1=jargon
        hidden_dropout_prob=0.2,  # Increased dropout to reduce overfitting
        attention_probs_dropout_prob=0.2  # Increased dropout to reduce overfitting
    )
    model.to(device)
    
    # Create datasets
    train_dataset = JargonDataset(abstracts_dir, train_annotations, tokenizer)
    
    # Use stratified split to ensure balanced validation set
    train_size = int(0.85 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # Smaller batch size for RoBERTa-large to fit in memory
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)
    
    # Train the model
    print("\nStarting training...")
    history = train(model, train_loader, val_loader, device, tokenizer, num_epochs=20)  # Reduced epochs to prevent overfitting
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    try:
        metrics = evaluate_jargon_detection(model_path, abstracts_dir, test_annotations)
        
        # Print final results
        print("\n=== Final Results ===")
        print(f"Total abstracts processed: {metrics['total_abstracts']}")
        print(f"Precision: {metrics['precision']:.2f}%")
        print(f"Recall: {metrics['recall']:.2f}%")
        print(f"F1 Score: {metrics['f1']:.2f}%")
    except FileNotFoundError:
        print("Model file not found. Using latest model instead.")
        # Try using the latest model instead
        metrics = evaluate_jargon_detection("output/jargon_model_latest.pt", abstracts_dir, test_annotations)
        
        # Print final results
        print("\n=== Final Results (using latest model) ===")
        print(f"Total abstracts processed: {metrics['total_abstracts']}")
        print(f"Precision: {metrics['precision']:.2f}%")
        print(f"Recall: {metrics['recall']:.2f}%")
        print(f"F1 Score: {metrics['f1']:.2f}%")
    
    # Training history was saved to 'output/training_history_roberta_large.png'
    print("\nTraining history plot saved to 'output/training_history_roberta_large.png'")

if __name__ == "__main__":
    main()