import torch
from transformers import RobertaTokenizerFast, RobertaForTokenClassification
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List
import random
from transformers import get_linear_schedule_with_warmup

class JargonDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, is_plaba=True):
        self.tokenizer = tokenizer
        self.is_plaba = is_plaba
        self.examples = self._prepare_data(data_path)
    
    def _prepare_data(self, data_path):
        examples = []
        
        if self.is_plaba:
            # Load PLABA data
            with open(data_path, 'r') as f:
                data = json.load(f)
            
            for abstract_id, jargons in data.items():
                try:
                    with open(f"data/PLABA_2024-Task_1/abstracts/{abstract_id}.src.txt", 'r') as f:
                        text = f.read()
                except FileNotFoundError:
                    continue
                
                # Tokenize text
                tokenized = self._tokenize_and_align_labels(text, list(jargons.keys()))
                if tokenized:
                    examples.append(tokenized)
        else:
            # New MedREADME data handling (similar to run_entity_experiments.py)
            with open(data_path, 'r') as f:
                data = json.load(f)
            
            for item in data:
                tokens = item['tokens']
                entities = item['entities']
                
                # Create BIO labels for the sentence
                labels = ['O'] * len(tokens)  # Initialize all as 'O'
                
                # Mark entity positions with BIO tags
                for start, end, _, _ in entities:
                    # Mark beginning of entity
                    labels[start] = 'B-JARGON'
                    # Mark inside of entity
                    for i in range(start + 1, end):
                        labels[i] = 'I-JARGON'
                
                # Tokenize and align labels
                tokenized = self.tokenizer(
                    tokens,
                    is_split_into_words=True,
                    add_special_tokens=True,
                    truncation=True,
                    padding='max_length',
                    max_length=512,
                    return_tensors='pt'
                )
                
                # Align labels with subwords
                word_ids = tokenized.word_ids()
                previous_word_idx = None
                label_ids = []
                
                for word_idx in word_ids:
                    if word_idx is None:
                        # Special tokens get -100
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        # First subword of a word gets the original label
                        label = labels[word_idx]
                        if label.startswith('B-') or label.startswith('I-'):
                            label_ids.append(1)  # Jargon
                        else:
                            label_ids.append(0)  # Not jargon
                    else:
                        # Subsequent subwords get the same label
                        if labels[word_idx].startswith('B-') or labels[word_idx].startswith('I-'):
                            label_ids.append(1)
                        else:
                            label_ids.append(0)
                    previous_word_idx = word_idx
                
                # Convert to tensor
                label_ids = torch.tensor(label_ids, dtype=torch.long)
                
                examples.append({
                    'input_ids': tokenized['input_ids'][0],
                    'attention_mask': tokenized['attention_mask'][0],
                    'labels': label_ids
                })
        
        return examples
    
    def _tokenize_and_align_labels(self, text, jargons):
        # Tokenize the text
        tokenized = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=512,
            padding='max_length',
            return_offsets_mapping=True,
            return_tensors='pt'
        )
        
        # Create labels
        labels = torch.zeros_like(tokenized['input_ids'][0])
        offset_mapping = tokenized['offset_mapping'][0].tolist()
        
        # Mark jargons using character offsets
        text_lower = text.lower()
        for jargon in jargons:
            jargon_lower = jargon.lower()
            start_idx = 0
            
            while True:
                pos = text_lower.find(jargon_lower, start_idx)
                if pos == -1:
                    break
                
                jargon_end = pos + len(jargon)
                first_token = True
                for idx, (token_start, token_end) in enumerate(offset_mapping):
                    if token_start == token_end:  # Skip special tokens
                        continue
                    if token_start < jargon_end and token_end > pos:
                        labels[idx] = 1
                    first_token = False
                
                start_idx = pos + 1
        
        # Set padding labels to -100
        attention_mask = tokenized['attention_mask'][0]
        labels[attention_mask == 0] = -100
        
        return {
            'input_ids': tokenized['input_ids'][0],
            'attention_mask': tokenized['attention_mask'][0],
            'labels': labels
        }
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def plot_curves(train_losses: List[float], val_metrics: List[Dict], save_path: str = 'output/training_curves.png'):
    """
    Plot training loss and validation metrics.
    
    Args:
        train_losses: List of training losses per epoch
        val_metrics: List of dictionaries containing validation metrics per epoch
        save_path: Path to save the plot
    """
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    # Plot training loss
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training Loss', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Plot validation metrics
    val_f1 = [m['f1'] for m in val_metrics]
    val_precision = [m['precision'] for m in val_metrics]
    val_recall = [m['recall'] for m in val_metrics]
    
    ax2.plot(epochs, val_f1, 'r-', label='Val F1')
    ax2.plot(epochs, val_precision, 'g-', label='Val Precision')
    ax2.plot(epochs, val_recall, 'y-', label='Val Recall')
    ax2.set_ylabel('Validation Metrics (%)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    plt.title('Training Loss and Validation Metrics')
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def train_model(model, train_loader, val_loader, device, num_epochs=20):
    # Use a smaller learning rate
    optimizer = AdamW(model.parameters(), lr=1e-5)  # Changed from 2e-5
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )
    best_f1 = 0
    patience = 5  # Increased patience for early stopping
    patience_counter = 0
    
    # Lists to store metrics for plotting
    train_losses = []
    val_metrics_history = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                labels=batch['labels'].to(device)
            )
            
            loss = outputs.loss
            loss.backward()
            # Slightly reduced clip value
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Changed from 1.0
            optimizer.step()
            total_loss += loss.item()
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate average loss for this epoch
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation
        val_metrics = evaluate(model, val_loader, device)
        val_metrics_history.append(val_metrics)
        
        print(f"\nEpoch {epoch+1}:")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Val F1: {val_metrics['f1']:.2f}")
        print(f"Val Precision: {val_metrics['precision']:.2f}")
        print(f"Val Recall: {val_metrics['recall']:.2f}")
        
        scheduler.step(val_metrics['f1'])
        
        # Early stopping with increased patience
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            # Save full checkpoint for training
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'val_f1': best_f1,
                'train_losses': train_losses,
                'val_metrics': val_metrics_history
            }, 'output/roberta_large_plaba_checkpoint.pt')
            
            # Save just the model state dict for inference
            torch.save(model.state_dict(), 'output/roberta_large_plaba.pt')
            print("New best model saved!")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Plot training curves
    plot_curves(train_losses, val_metrics_history)
    return best_f1

def get_entities_from_labels(tokens, labels):
    """Extract entity spans from labels."""
    entities = []
    current_entity = []
    
    for i, (token, label) in enumerate(zip(tokens, labels)):
        if label == 1:
            current_entity.append(i)
        elif current_entity:
            entities.append((min(current_entity), max(current_entity) + 1))
            current_entity = []
    
    if current_entity:  # Don't forget last entity
        entities.append((min(current_entity), max(current_entity) + 1))
    
    return entities

def evaluate(model, dataloader, device):
    model.eval()
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device)
            )
            predictions = torch.argmax(outputs.logits, dim=2)
            
            # Process each sequence in the batch
            for pred_seq, label_seq, mask, input_ids in zip(predictions, batch['labels'], 
                                                          batch['attention_mask'], batch['input_ids']):
                # Get actual sequence length
                seq_len = mask.sum().item()
                
                # Get predictions and labels for non-padding tokens
                pred_seq = pred_seq[:seq_len].cpu()
                label_seq = label_seq[:seq_len].cpu()
                
                # Get token sequence
                tokens = input_ids[:seq_len]
                
                # Extract predicted and gold entities
                pred_entities = get_entities_from_labels(tokens, pred_seq)
                gold_entities = get_entities_from_labels(tokens, label_seq)
                
                # Exact match comparison
                pred_set = set(map(tuple, pred_entities))
                gold_set = set(map(tuple, gold_entities))
                
                # Calculate metrics
                true_positives += len(pred_set & gold_set)
                false_positives += len(pred_set - gold_set)
                false_negatives += len(gold_set - pred_set)
    
    # Calculate final metrics
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    
    return {
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize tokenizer and model
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large', add_prefix_space=True)
    model = RobertaForTokenClassification.from_pretrained(
        'roberta-large',
        num_labels=2,
        hidden_dropout_prob=0.3,
        attention_probs_dropout_prob=0.3,
        classifier_dropout=0.3
    )
    model.to(device)
    
    # Load PLABA datasets
    print("Loading PLABA datasets...")
    plaba_train = JargonDataset('data/PLABA_2024-Task_1/train.json', tokenizer, is_plaba=True)
    plaba_test = JargonDataset('data/PLABA_2024-Task_1/task_1_testing.json', tokenizer, is_plaba=True)
    
    # Split PLABA test set into validation and test
    test_size = len(plaba_test)
    val_size = int(0.3 * test_size)  # Use 30% of test set as validation
    test_size = test_size - val_size
    
    plaba_val, plaba_final_test = torch.utils.data.random_split(
        plaba_test, 
        [val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create PLABA dataloaders
    train_loader = DataLoader(plaba_train, batch_size=8, shuffle=True)
    val_loader = DataLoader(plaba_val, batch_size=8)
    test_loader = DataLoader(plaba_final_test, batch_size=8)
    
    # Print dataset sizes
    print(f"\nPLABA Dataset sizes:")
    print(f"Train size: {len(plaba_train)}")
    print(f"Validation size: {val_size}")
    print(f"Test size: {test_size}")
    
    # Train on PLABA
    print("\nTraining on PLABA dataset...")
    best_f1 = train_model(model, train_loader, val_loader, device)
    
    # Evaluate on PLABA test set
    print("\nEvaluating on PLABA test set...")
    plaba_metrics = evaluate(model, test_loader, device)
    print(f"PLABA Test Results:")
    print(f"F1: {plaba_metrics['f1']:.2f}")
    print(f"Precision: {plaba_metrics['precision']:.2f}")
    print(f"Recall: {plaba_metrics['recall']:.2f}")
    
    # Load and split MedREADME data
    print("\nLoading and evaluating on MedREADME dataset...")
    medreadme_data = JargonDataset('data/medreadme/jargon.json', tokenizer, is_plaba=False)
    
    # Split MedREADME data into validation and test
    total_size = len(medreadme_data)
    medreadme_test_size = int(0.3 * total_size)
    medreadme_val_size = total_size - medreadme_test_size
    
    medreadme_val, medreadme_test = torch.utils.data.random_split(
        medreadme_data,
        [medreadme_val_size, medreadme_test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    medreadme_test_loader = DataLoader(medreadme_test, batch_size=8)
    
    print(f"\nMedREADME Dataset sizes:")
    print(f"Validation size: {medreadme_val_size}")
    print(f"Test size: {medreadme_test_size}")
    
    # Evaluate on MedREADME test set
    medreadme_metrics = evaluate(model, medreadme_test_loader, device)
    print("\nMedREADME Test Results:")
    print(f"F1: {medreadme_metrics['f1']:.2f}")
    print(f"Precision: {medreadme_metrics['precision']:.2f}")
    print(f"Recall: {medreadme_metrics['recall']:.2f}")

    # When loading the best model for evaluation, also load the metrics
    checkpoint = torch.load('output/roberta_large_plaba_checkpoint.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("\nBest model was saved at:")
    print(f"Epoch: {checkpoint['epoch'] + 1}")
    print(f"Training Loss: {checkpoint['loss']:.4f}")
    print(f"Validation F1: {checkpoint['val_f1']:.2f}")

if __name__ == "__main__":
    main() 