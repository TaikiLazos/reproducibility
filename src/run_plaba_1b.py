import json
import os
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import numpy as np

class PLABA1bDataset:
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
        
        # Define correct action labels from the data
        self.action_labels = ['SUBSTITUTE', 'EXPLAIN', 'GENERALIZE', 'OMIT', 'EXEMPLIFY']
        self.label2id = {label: i for i, label in enumerate(self.action_labels)}
        self.id2label = {i: label for i, label in enumerate(self.action_labels)}
            
    def get_abstract_text(self, abstract_id):
        """Load abstract text from file."""
        abstract_path = self.abstracts_dir / f"{abstract_id}.src.txt"
        with open(abstract_path, 'r') as f:
            return f.read().strip()
    
    def get_training_examples(self):
        """Create training examples by pairing jargon terms with their actions."""
        examples = []
        
        for abstract_id, terms in self.train_data.items():
            abstract_text = self.get_abstract_text(abstract_id)
            
            for term, info_list in terms.items():
                # Convert all actions to a list format
                actions = []
                for item in info_list:
                    if isinstance(item, list):
                        actions.append(item[0])  # Get the action from each [action, explanation] pair
                
                if not actions:  # If no actions found
                    continue  # Skip this example
                    
                examples.append({
                    'abstract_id': abstract_id,
                    'text': abstract_text,
                    'term': term,
                    'actions': actions  # Now using 'actions' instead of 'action'
                })
        
        return examples
    
    def get_test_examples(self):
        """Create test examples with actual labels."""
        examples = []
        
        for abstract_id, terms in self.test_data.items():
            abstract_text = self.get_abstract_text(abstract_id)
            
            for term, action_list in terms.items():
                # Extract just the action types (first element of each sublist)
                actions = [action[0] for action in action_list]
                
                examples.append({
                    'abstract_id': abstract_id,
                    'text': abstract_text,
                    'term': term,
                    'actions': actions
                })
        
        return examples

class JargonActionDataset(Dataset):
    def __init__(self, examples, tokenizer, label2id, max_length=512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        text = f"Abstract: {example['text']} Term: {example['term']}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Create multi-label vector (a term can have multiple valid actions)
        label_vector = torch.zeros(len(self.label2id))
        for action in example['actions']:
            if action in self.label2id:  # Only use valid actions
                label_vector[self.label2id[action]] = 1
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': label_vector
        }

def train_model(model, train_dataloader, num_epochs=1, learning_rate=2e-5, warmup_steps=0.1, patience=5):
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
    
    # Training loop
    print(f"\nTraining on {device}")
    print(f"Total steps: {total_steps}")
    
    best_loss = float('inf')
    no_improve_epochs = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            scheduler.step()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate average loss for epoch
        avg_loss = total_loss / len(train_dataloader)
        print(f"\nEpoch {epoch + 1} - Average loss: {avg_loss:.4f}")
        
        # Check for improvement
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve_epochs = 0
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
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # For multi-label, use sigmoid and threshold
            preds = (torch.sigmoid(outputs.logits) > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays for easier handling
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=list(id2label.values())))
    
    return {
        'f1_macro': f1_score(all_labels, all_preds, average='macro'),
        'precision_macro': precision_score(all_labels, all_preds, average='macro'),
        'recall_macro': recall_score(all_labels, all_preds, average='macro')
    }

def main():
    # Initialize dataset
    dataset = PLABA1bDataset("data/PLABA_2024-Task_1")
    
    # Get training examples
    training_examples = dataset.get_training_examples()
    
    # Create classification dataset
    train_dataset = JargonActionDataset(training_examples, dataset.tokenizer, dataset.label2id)
    
    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
        num_labels=len(dataset.action_labels),
        problem_type="multi_label_classification",
        id2label=dataset.id2label,
        label2id=dataset.label2id
    )
    
    # Create dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    # Print some statistics
    print(f"Loaded {len(training_examples)} training examples")
    
    # Train the model
    model = train_model(model, train_dataloader)
    
    # Save the model
    output_dir = "models/plaba_action_classifier"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    dataset.tokenizer.save_pretrained(output_dir)
    print(f"\nModel saved to {output_dir}")
    
    # Load test data and evaluate
    test_examples = dataset.get_test_examples()
    test_dataset = JargonActionDataset(test_examples, dataset.tokenizer, dataset.label2id)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Evaluate the model
    metrics = evaluate_model(model, test_dataloader, dataset.id2label)
    print("\nFinal Metrics:")
    print(f"Macro F1: {metrics['f1_macro']:.4f}")
    print(f"Macro Precision: {metrics['precision_macro']:.4f}")
    print(f"Macro Recall: {metrics['recall_macro']:.4f}")

if __name__ == "__main__":
    main() 