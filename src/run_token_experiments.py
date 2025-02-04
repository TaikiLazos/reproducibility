import pandas as pd
import torch
from transformers import (
    BertForTokenClassification, 
    RobertaForTokenClassification,
    BertTokenizerFast,
    RobertaTokenizerFast,
)
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, f1_score
import os
from tabulate import tabulate
import json
from torch.optim import AdamW
import numpy as np

class TokenLevelDataset(Dataset):
    def __init__(self, data_csv, data_json, tokenizer, max_length=128, classification_type='binary'):
        self.df = pd.read_csv(data_csv)
        with open(data_json, 'r') as f:
            # Convert from list to dictionary for easier indexing
            json_list = json.load(f)
            self.json_data = {str(i): item for i, item in enumerate(json_list)}
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.classification_type = classification_type
        self.instances = self._process_tokens()
    
    def _process_tokens(self):
        instances = []
        
        for idx, row in self.df.iterrows():
            json_item = self.json_data[str(idx)]
            tokens = json_item['tokens']
            
            # Initialize all tokens as non-complex (0)
            token_labels = [0] * len(tokens)
            
            # Label tokens that are part of entities
            for entity in json_item['entities']:
                start_idx, end_idx, label, _ = entity
                
                # Get the label based on classification type
                if self.classification_type == 'binary':
                    processed_label = 1 if 'complex' in label else 0
                elif self.classification_type == '3-cls':
                    if label.startswith('medical'):
                        processed_label = 0
                    elif label.startswith('abbr'):
                        processed_label = 1
                    else:  # general or multisense
                        processed_label = 2
                else:  # 7-cls
                    label_mapping = {
                        'medical-jargon': 0,
                        'general-complex': 1,
                        'general-medical-multisense': 2,
                        'abbr-medical': 3,
                        'medical-name-entity': 4,
                        'general-name-entity': 5,
                        'abbr-general': 6
                    }
                    base_category = '-'.join(label.split('-')[:2])
                    processed_label = label_mapping.get(base_category, 0)
                
                # Label all tokens in the entity span
                for i in range(start_idx, end_idx):
                    token_labels[i] = processed_label
            
            instances.append({
                'tokens': tokens,
                'token_labels': token_labels
            })
        
        return instances
    
    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, idx):
        instance = self.instances[idx]
        
        # Tokenize the sentence and align labels
        encoding = self.tokenizer(
            instance['tokens'],
            is_split_into_words=True,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Convert token labels to match tokenizer's subwords
        word_ids = encoding.word_ids()
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens get label -100 (ignored in loss calculation)
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # Only label the first token of a word
                label_ids.append(instance['token_labels'][word_idx])
            else:
                # Continuation of a word gets ignored in loss calculation
                label_ids.append(-100)
            previous_word_idx = word_idx
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }

def train_and_evaluate(model, train_loader, val_loader, device, model_save_path, num_labels):
    optimizer = AdamW(model.parameters(), lr=2e-5)
    best_accuracy = 0
    num_epochs = 3
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                labels=batch['labels'].to(device)
            )
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device)
                )
                
                # Get predictions for non-ignored tokens
                labels = batch['labels'].cpu().numpy()
                predictions = outputs.logits.argmax(dim=-1).cpu().numpy()
                
                # Only include non-ignored tokens
                for pred, label in zip(predictions, labels):
                    mask = label != -100
                    all_preds.extend(pred[mask])
                    all_labels.extend(label[mask])
        
        # Calculate metrics
        if len(all_preds) > 0 and len(all_labels) > 0:
            accuracy = accuracy_score(all_labels, all_preds) * 100
            
            print(f"Validation Accuracy: {accuracy:.2f}%")
            print("\nDetailed Classification Report:")
            print(classification_report(all_labels, all_preds))
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), model_save_path)
                print(f"New best model saved with Accuracy: {accuracy:.2f}%")
        else:
            print("Warning: No valid predictions in validation set!")
            accuracy = 0
    
    return best_accuracy

def run_token_experiments():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load and split data first
    df = pd.read_csv('data/readability.csv')
    with open('data/jargon.json', 'r') as f:
        json_data = json.load(f)
    
    # Print initial data stats
    print(f"Total samples in dataset: {len(df)}")
    print(f"Split distribution:\n{df['split'].value_counts()}")
    
    # If all data is in train, create a validation split
    if 'val' not in df['split'].unique():
        print("No validation split found. Creating train/val split...")
        # Randomly assign 20% of train data to validation
        train_mask = df['split'] == 'train'
        train_indices = df[train_mask].index
        val_size = int(0.2 * len(train_indices))
        
        # Randomly select validation indices
        val_indices = np.random.choice(train_indices, size=val_size, replace=False)
        
        # Update splits
        df.loc[val_indices, 'split'] = 'val'
        for idx in val_indices:
            json_data[idx]['split'] = 'val'
    
    # Now split the data
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df = df[df['split'] == 'val'].reset_index(drop=True)
    
    # Split JSON data according to the split field
    train_json = [item for item in json_data if item['split'] == 'train']
    val_json = [item for item in json_data if item['split'] == 'val']
    
    # Print split sizes
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Save split datasets
    os.makedirs('data/splits', exist_ok=True)
    train_df.to_csv('data/splits/train.csv', index=False)
    val_df.to_csv('data/splits/val.csv', index=False)
    with open('data/splits/train.json', 'w') as f:
        json.dump(train_json, f)
    with open('data/splits/val.json', 'w') as f:
        json.dump(val_json, f)
    
    # Configuration settings
    models = {
        'BERT': (BertForTokenClassification, BertTokenizerFast, 'bert-base-uncased', 'bert-large-uncased'),
        'RoBERTa': (RobertaForTokenClassification, RobertaTokenizerFast, 'roberta-base', 'roberta-large'),
        'BioBERT': (BertForTokenClassification, BertTokenizerFast, 'dmis-lab/biobert-base-cased-v1.2', 'dmis-lab/biobert-large-cased-v1.1'),
        'PubMedBERT': (BertForTokenClassification, BertTokenizerFast, 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext', 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')
    }
    
    sizes = ['base', 'large']
    class_types = ['binary', '3-cls', '7-cls']
    
    # Create output directory
    os.makedirs('output_token', exist_ok=True)
    
    # Results dictionary
    results = {size: {model: {cls_type: 0 for cls_type in class_types} 
                     for model in models.keys()} 
              for size in sizes}
    
    # Run experiments
    for model_name, (model_class, tokenizer_class, base_model, large_model) in models.items():
        for size in sizes:
            model_path = large_model if size == 'large' else base_model
            
            for cls_type in class_types:
                num_labels = 2 if cls_type == 'binary' else (3 if cls_type == '3-cls' else 7)
                
                # Initialize model and tokenizer
                tokenizer = tokenizer_class.from_pretrained(model_path)
                model = model_class.from_pretrained(
                    model_path, 
                    num_labels=num_labels,
                    ignore_mismatched_sizes=True
                )
                model.to(device)
                
                # Create datasets
                train_dataset = TokenLevelDataset('data/splits/train.csv', 'data/splits/train.json', 
                                                tokenizer, classification_type=cls_type)
                val_dataset = TokenLevelDataset('data/splits/val.csv', 'data/splits/val.json', 
                                              tokenizer, classification_type=cls_type)
                
                train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=16)
                
                # Train and evaluate
                save_path = f'output_token/{model_name}_{size}_{cls_type}.pt'
                accuracy = train_and_evaluate(model, train_loader, val_loader, device, save_path, num_labels)
                results[size][model_name][cls_type] = accuracy
                
                print(f"Completed {model_name}-{size} {cls_type}: {accuracy:.1f}%")
    
    # Save results
    with open('output_token/results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print table
    headers = ['Models', 'Binary', '3-Cls', '7-Cls']
    table_data = []
    
    # Add large models
    table_data.append(['Large-size Models', '', '', ''])
    for model in models.keys():
        row = [model]
        for cls_type in class_types:
            row.append(f"{results['large'][model][cls_type]:.1f}")
        table_data.append(row)
    
    # Add base models
    table_data.append(['Base-size Models', '', '', ''])
    for model in models.keys():
        row = [model]
        for cls_type in class_types:
            row.append(f"{results['base'][model][cls_type]:.1f}")
        table_data.append(row)
    
    print("\nResults Table:")
    print(tabulate(table_data, headers=headers, tablefmt='grid'))

if __name__ == "__main__":
    run_token_experiments() 