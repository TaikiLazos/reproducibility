import json
import os
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup
)
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

class PLABADataset:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.abstracts_dir = self.data_dir / "abstracts"
        
        # Load training data
        with open(self.data_dir / "train.json", 'r') as f:
            self.train_data = json.load(f)
            
        # Load test data
        with open(self.data_dir / "task_1_testing.json", 'r') as f:
            self.test_data = json.load(f)
            
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')
            
    def get_abstract_text(self, abstract_id):
        """Load abstract text from file."""
        abstract_path = self.abstracts_dir / f"{abstract_id}.src.txt"
        with open(abstract_path, 'r') as f:
            return f.read().strip()
    
    def create_token_labels(self, text, jargon_terms):
        """Create BIO labels for each token in the text."""
        # First, get word-level tokenization
        words = text.split()
        word_labels = ['O'] * len(words)
        
        # Find jargon terms in words
        for i, word in enumerate(words):
            for term in jargon_terms:
                term_words = term.split()
                if i + len(term_words) <= len(words):
                    if ' '.join(words[i:i+len(term_words)]) == term:
                        # Mark the first word as B-JARGON
                        word_labels[i] = 'B-JARGON'
                        # Mark remaining words as I-JARGON
                        for j in range(1, len(term_words)):
                            word_labels[i+j] = 'I-JARGON'
        
        # Now tokenize each word and propagate labels
        all_tokens = []
        all_labels = []
        
        for word, word_label in zip(words, word_labels):
            # Tokenize without special tokens
            word_tokens = self.tokenizer.tokenize(word)
            
            # First subword gets the word's label
            if word_tokens:
                all_tokens.extend(word_tokens)
                all_labels.append(word_label)
                # Any remaining subwords get I-JARGON if part of jargon, O otherwise
                for _ in range(len(word_tokens) - 1):
                    all_labels.append('I-JARGON' if word_label != 'O' else 'O')
                    
        return all_tokens, all_labels
            
    def get_training_examples(self):
        """Create training examples by pairing abstracts with their jargon terms."""
        examples = []
        
        for abstract_id, terms in self.train_data.items():
            abstract_text = self.get_abstract_text(abstract_id)
            jargon_terms = list(terms.keys())
            
            # Create token-level labels
            tokens, labels = self.create_token_labels(abstract_text, jargon_terms)
            
            examples.append({
                'id': abstract_id,
                'text': abstract_text,
                'jargon_terms': jargon_terms,
                'tokens': tokens,
                'labels': labels
            })
            
        return examples

    def get_test_examples(self):
        """Create test examples by pairing abstracts with their jargon terms."""
        examples = []
        
        for abstract_id, terms in self.test_data.items():
            abstract_text = self.get_abstract_text(abstract_id)
            jargon_terms = list(terms.keys())
            
            # Create token-level labels
            tokens, labels = self.create_token_labels(abstract_text, jargon_terms)
            
            examples.append({
                'id': abstract_id,
                'text': abstract_text,
                'jargon_terms': jargon_terms,
                'tokens': tokens,
                'labels': labels
            })
            
        return examples

class PLABATokenDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create label to id mapping
        self.label2id = {'O': 0, 'B-JARGON': 1, 'I-JARGON': 2}
        self.id2label = {v: k for k, v in self.label2id.items()}
        
    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Add special tokens and pad
        tokens = ['[CLS]'] + example['tokens'][:self.max_length-2] + ['[SEP]']
        labels = ['O'] + example['labels'][:self.max_length-2] + ['O']
        
        # Convert tokens to ids
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        label_ids = [self.label2id[label] for label in labels]
        
        # Pad sequences
        padding_length = self.max_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        label_ids += [self.label2id['O']] * padding_length
        
        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'labels': torch.tensor(label_ids)
        }

def calculate_class_weights(examples):
    """Calculate class weights based on label distribution."""
    label_counts = {'O': 0, 'B-JARGON': 0, 'I-JARGON': 0}
    total_labels = 0
    
    for example in examples:
        for label in example['labels']:
            label_counts[label] += 1
            total_labels += 1
    
    # Calculate weights (inverse of frequency)
    weights = {
        label: total_labels / (count * len(label_counts)) 
        for label, count in label_counts.items()
    }
    
    print("\nLabel distribution:")
    for label, count in label_counts.items():
        print(f"{label}: {count} ({count/total_labels:.2%})")
    
    print("\nClass weights:")
    for label, weight in weights.items():
        print(f"{label}: {weight:.4f}")
        
    return weights

def train_model(model, train_dataloader, class_weights, num_epochs=100, learning_rate=3e-5, warmup_steps=0.1, patience=7):
    """Train the model with early stopping."""
    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Calculate total training steps
    total_steps = len(train_dataloader) * num_epochs
    
    # Create scheduler with warmup
    if isinstance(warmup_steps, float):
        warmup_steps = int(total_steps * warmup_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Create class weight tensor
    class_weight_tensor = torch.tensor([
        class_weights['O'],
        class_weights['B-JARGON'],
        class_weights['I-JARGON']
    ]).to(device)
    
    # Training loop
    print(f"\nTraining on {device}")
    print(f"Total steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")
    
    best_loss = float('inf')
    no_improve_epochs = 0
    best_model_state = None
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Zero gradients
            model.zero_grad()
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Apply class weights to loss
            logits = outputs.logits
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, 3),
                labels.view(-1),
                weight=class_weight_tensor,
                ignore_index=-100
            )
            
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            scheduler.step()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
        
        # Print epoch statistics
        avg_loss = total_loss / len(train_dataloader)
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"Average loss: {avg_loss:.4f}")
        
        # Check for improvement
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve_epochs = 0
            # Save best model
            best_model_path = "models/plaba_jargon_detector_best"
            os.makedirs(best_model_path, exist_ok=True)
            model.save_pretrained(best_model_path)
            print(f"Saved best model with loss: {best_loss:.4f}")
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs} epochs")
        
        # Early stopping
        if no_improve_epochs >= patience:
            print(f"\nEarly stopping after {epoch + 1} epochs")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)
    
    return model

def evaluate_model(model, test_dataloader, id2label):
    """Evaluate the model on test data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_true_labels = []
    
    print("\nEvaluating model...")
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Get predictions
            predictions = torch.argmax(outputs.logits, dim=2)
            
            # Convert predictions and labels to lists
            for pred, label, mask in zip(predictions, labels, attention_mask):
                pred = pred.cpu().numpy()
                label = label.cpu().numpy()
                mask = mask.cpu().numpy()
                
                # Filter out padding and special tokens
                true_label = [id2label[l] for l, m in zip(label, mask) if m == 1][1:-1]  # Remove [CLS] and [SEP]
                pred_label = [id2label[p] for p, m in zip(pred, mask) if m == 1][1:-1]  # Remove [CLS] and [SEP]
                
                all_predictions.append(pred_label)
                all_true_labels.append(true_label)
    
    # Calculate metrics
    print("\nToken-level Classification Report:")
    print(classification_report(all_true_labels, all_predictions))
    
    # Calculate span-level metrics
    span_metrics = calculate_span_metrics(all_predictions, all_true_labels)
    print("\nSpan-level Metrics:")
    for metric, value in span_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return {
        'token_f1': f1_score(all_true_labels, all_predictions),
        'token_precision': precision_score(all_true_labels, all_predictions),
        'token_recall': recall_score(all_true_labels, all_predictions),
        'span_metrics': span_metrics
    }

def calculate_span_metrics(predictions, true_labels):
    """Calculate span-level metrics for jargon term detection."""
    correct_spans = 0
    predicted_spans = 0
    true_spans = 0
    
    for pred_seq, true_seq in zip(predictions, true_labels):
        # Get predicted spans
        pred_spans = get_spans(pred_seq)
        true_spans_seq = get_spans(true_seq)
        
        # Update counts
        correct_spans += len(pred_spans & true_spans_seq)
        predicted_spans += len(pred_spans)
        true_spans += len(true_spans_seq)
    
    # Calculate metrics
    precision = correct_spans / predicted_spans if predicted_spans > 0 else 0
    recall = correct_spans / true_spans if true_spans > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'span_precision': precision,
        'span_recall': recall,
        'span_f1': f1
    }

def get_spans(sequence):
    """Extract spans from a sequence of BIO tags."""
    spans = set()
    current_span = []
    
    for i, tag in enumerate(sequence):
        if tag == 'B-JARGON':
            if current_span:
                spans.add(tuple(current_span))
            current_span = [i]
        elif tag == 'I-JARGON' and current_span:
            current_span.append(i)
        elif tag == 'O':
            if current_span:
                spans.add(tuple(current_span))
                current_span = []
    
    if current_span:
        spans.add(tuple(current_span))
    
    return spans

def main():
    # Initialize dataset
    dataset = PLABADataset("data/PLABA_2024-Task_1")
    
    # Get training examples
    training_examples = dataset.get_training_examples()
    
    # Create token classification dataset
    token_dataset = PLABATokenDataset(training_examples, dataset.tokenizer)
    
    # Calculate class weights
    class_weights = calculate_class_weights(training_examples)
    
    # Initialize model with different configuration
    model = AutoModelForTokenClassification.from_pretrained(
        'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
        num_labels=3,
        id2label={0: 'O', 1: 'B-JARGON', 2: 'I-JARGON'},
        label2id={'O': 0, 'B-JARGON': 1, 'I-JARGON': 2},
        hidden_dropout_prob=0.2,
        attention_probs_dropout_prob=0.2,
        classifier_dropout=0.2,
    )
    
    # Create dataloader with smaller batch size
    train_dataloader = DataLoader(
        token_dataset, 
        batch_size=4,
        shuffle=True
    )
    
    # Print some statistics
    print(f"Loaded {len(training_examples)} training examples")
    print(f"Created {len(token_dataset)} training instances")
    
    # Train the model with more epochs and early stopping
    model = train_model(
        model, 
        train_dataloader,
        class_weights,
        num_epochs=100,  # Increased to 100
        learning_rate=3e-5,
        warmup_steps=0.1,
        patience=7  # Stop if no improvement for 7 epochs
    )
    
    # Save the final model
    output_dir = "models/plaba_jargon_detector_final"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    dataset.tokenizer.save_pretrained(output_dir)
    print(f"\nFinal model saved to {output_dir}")
    
    # Load test data and evaluate
    test_examples = dataset.get_test_examples()
    test_dataset = PLABATokenDataset(test_examples, dataset.tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Evaluate the model
    metrics = evaluate_model(model, test_dataloader, model.config.id2label)
    print("\nFinal Metrics:")
    print(f"Token-level F1: {metrics['token_f1']:.4f}")
    print(f"Span-level F1: {metrics['span_metrics']['span_f1']:.4f}")

if __name__ == "__main__":
    main() 