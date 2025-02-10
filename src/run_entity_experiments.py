import pandas as pd
import torch
from transformers import (
    BertForTokenClassification, 
    RobertaForTokenClassification,
    BertTokenizerFast,
    RobertaTokenizerFast,
    AutoTokenizer,
    AutoModelForTokenClassification
)
from torch.utils.data import Dataset, DataLoader
import os
from tabulate import tabulate
import json
from torch.optim import AdamW
import numpy as np

class EntityLevelDataset(Dataset):
    def __init__(self, data_json, tokenizer, classification_type='binary'):
        with open(data_json, 'r') as f:
            self.json_data = json.load(f)
        self.tokenizer = tokenizer
        self.classification_type = classification_type
        
        # Create label mapping
        self.label_map = self._create_label_map()
        # Create reverse mapping for evaluation
        self.id_to_tag = {v: k for k, v in self.label_map.items()}
        
        # Process all sentences
        self.examples = self._process_data()
    
    def _create_label_map(self):
        """Create mapping from BIO tags to indices"""
        if self.classification_type == 'binary':
            return {
                'O': 0,
                'B-COMPLEX': 1,
                'I-COMPLEX': 1
            }
        elif self.classification_type == '3-cls':
            return {
                'O': 0,
                'B-MEDICAL': 1, 'I-MEDICAL': 1,
                'B-ABBR': 2, 'I-ABBR': 2,
                'B-GENERAL': 3, 'I-GENERAL': 3
            }
        else:  # 7-cls
            return {
                'O': 0,
                'B-MEDICAL_CONDITION': 1, 'I-MEDICAL_CONDITION': 1,
                'B-MEDICAL_TREATMENT': 2, 'I-MEDICAL_TREATMENT': 2,
                'B-MEDICAL_TEST': 3, 'I-MEDICAL_TEST': 3,
                'B-ABBR': 4, 'I-ABBR': 4,
                'B-GENERAL_HEALTH': 5, 'I-GENERAL_HEALTH': 5,
                'B-GENERAL_RESEARCH': 6, 'I-GENERAL_RESEARCH': 6,
                'B-MULTISENSE': 7, 'I-MULTISENSE': 7
            }
    
    def _process_data(self):
        """Process all sentences and their labels"""
        examples = []
        
        for item in self.json_data:
            # Print an example item to debug
            if len(examples) == 0:
                print("\nExample JSON item:")
                print(json.dumps(item, indent=2))
            
            # Extract tokens and entities
            tokens = item['tokens']
            entities = item['entities']
            
            # Create BIO labels for the sentence
            labels = ['O'] * len(tokens)  # Initialize all as 'O'
            
            # Mark entity positions with BIO tags
            for start, end, label, _ in entities:
                if self.classification_type == 'binary':
                    bio_tag = 'COMPLEX'
                elif self.classification_type == '3-cls':
                    if 'medical' in label.lower():
                        bio_tag = 'MEDICAL'
                    elif 'abbr' in label.lower():
                        bio_tag = 'ABBR'
                    else:
                        bio_tag = 'GENERAL'
                else:  # 7-cls
                    bio_tag = label.upper()
                
                # Mark beginning of entity
                labels[start] = f'B-{bio_tag}'
                # Mark inside of entity
                for i in range(start + 1, end):
                    labels[i] = f'I-{bio_tag}'
            
            # Print example before tokenization
            if len(examples) == 0:
                print("\nBefore tokenization:")
                print("Tokens:", tokens)
                print("Labels:", labels)
            
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
                    # First subword gets the original label
                    label = labels[word_idx]
                    label_ids.append(self.label_map.get(label, 0))
                else:
                    # Subsequent subwords get the I- version of the label
                    if labels[word_idx].startswith('B-') or labels[word_idx].startswith('I-'):
                        label = 'I-' + labels[word_idx].split('-')[1]
                        label_ids.append(self.label_map.get(label, 0))
                    else:
                        label_ids.append(self.label_map.get('O', 0))
                previous_word_idx = word_idx
            
            # Convert to tensor
            label_ids = torch.tensor(label_ids, dtype=torch.long)
            
            # Print example after tokenization
            if len(examples) == 0:
                print("\nAfter tokenization:")
                print("Tokens:", self.tokenizer.convert_ids_to_tokens(tokenized['input_ids'][0]))
                print("Label IDs:", label_ids.tolist())
            
            examples.append({
                'input_ids': tokenized['input_ids'][0],
                'attention_mask': tokenized['attention_mask'][0],
                'labels': label_ids
            })
        
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def train_and_evaluate(model, train_loader, val_loader, device, model_save_path, num_labels):
    optimizer = AdamW(model.parameters(), lr=2e-5)
    best_f1 = 0
    patience = 3
    patience_counter = 0
    max_epochs = 20
    
    for epoch in range(max_epochs):
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
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{max_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Evaluate
        results = evaluate_entities(model, val_loader, device)
        print(f"\nValidation Results:")
        print(f"Overall F1: {results['overall']['f1']:.2f}")
        print(f"Precision: {results['overall']['precision']:.2f}")
        print(f"Recall: {results['overall']['recall']:.2f}")
        
        # Early stopping check
        current_f1 = results['overall']['f1']
        if current_f1 > best_f1:
            best_f1 = current_f1
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved with F1: {current_f1:.2f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    return best_f1

def evaluate_entities(model, val_loader, device):
    model.eval()
    correct_chunks = 0  # True positives
    found_guessed = 0   # Total predicted chunks
    found_correct = 0   # Total actual chunks
    
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device)
            )
            predictions = torch.argmax(outputs.logits, dim=2)
            
            # Process each sequence in batch
            for pred_seq, label_seq, mask in zip(predictions, batch['labels'], batch['attention_mask']):
                length = mask.sum().item()
                pred_seq = pred_seq[:length].cpu().tolist()
                label_seq = label_seq[:length].cpu().tolist()
                
                # Track current chunks for both predicted and true labels
                curr_pred_chunk = None
                curr_true_chunk = None
                
                # Iterate through sequence
                for pred, true in zip(pred_seq, label_seq):
                    if true == -100:  # Skip special tokens
                        continue
                        
                    # Handle true chunks
                    if true != 0:  # Not O
                        if curr_true_chunk is None:
                            curr_true_chunk = true
                            found_correct += 1
                    else:
                        curr_true_chunk = None
                        
                    # Handle predicted chunks
                    if pred != 0:  # Not O
                        if curr_pred_chunk is None:
                            curr_pred_chunk = pred
                            found_guessed += 1
                            # Check if this is also a true chunk start
                            if pred == true:
                                correct_chunks += 1
                    else:
                        curr_pred_chunk = None
    
    # Calculate metrics
    precision = correct_chunks / found_guessed if found_guessed > 0 else 0
    recall = correct_chunks / found_correct if found_correct > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'overall': {
            'f1': f1 * 100,
            'precision': precision * 100,
            'recall': recall * 100
        }
    }

def run_experiments():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define models to test
    models = {
        # 'BERT': (BertForTokenClassification, BertTokenizerFast, 'bert-base-uncased', 'bert-large-uncased')
        # 'RoBERTa': (RobertaForTokenClassification, RobertaTokenizerFast, 'roberta-base', 'roberta-large')
        'BioBERT': (AutoModelForTokenClassification, AutoTokenizer, 'dmis-lab/biobert-base-cased-v1.2', 'dmis-lab/biobert-large-cased-v1.1')
        # 'PubMedBERT': (AutoModelForTokenClassification, AutoTokenizer, 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext', 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')
    }
    
    sizes = ['base', 'large']
    class_types = ['binary', '3-cls', '7-cls']
    
    # Results dictionary
    results = {size: {model: {cls_type: {'f1': 0, 'precision': 0, 'recall': 0} 
                     for cls_type in class_types} 
              for model in models.keys()} 
              for size in sizes}
    
    # Run experiments
    for model_name, (model_class, tokenizer_class, base_model, large_model) in models.items():
        for size in sizes:
            model_path = large_model if size == 'large' else base_model
            
            for cls_type in class_types:
                print(f"\nRunning experiment for {model_name}-{size} {cls_type}")
                num_labels = {
                    'binary': 2,
                    '3-cls': 4,
                    '7-cls': 8
                }[cls_type]
                
                # Initialize tokenizer and model
                tokenizer = tokenizer_class.from_pretrained(model_path, add_prefix_space=True)
                model = model_class.from_pretrained(model_path, num_labels=num_labels)
                model.to(device)
                
                # Create datasets
                train_dataset = EntityLevelDataset('data/splits/train.json', tokenizer, classification_type=cls_type)
                val_dataset = EntityLevelDataset('data/splits/val.json', tokenizer, classification_type=cls_type)
                
                train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=16)
                
                # Train and evaluate
                save_path = f'output/{model_name}_{size}_{cls_type}.pt'
                best_f1 = train_and_evaluate(model, train_loader, val_loader, device, save_path, num_labels)
                
                # Store metrics
                results[size][model_name][cls_type]['f1'] = best_f1
                
                print(f"Completed {model_name}-{size} {cls_type}")
                print(f"F1: {best_f1:.1f}%")
    
    # Save results
    os.makedirs('output', exist_ok=True)
    with open('output/results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print table with F1 scores
    headers = ['Models', 'Binary', '3-Cls', '7-Cls']
    table_data = []
    
    # Add large models
    table_data.append(['Large-size Models', '', '', ''])
    for model in models.keys():
        row = [model]
        for cls_type in class_types:
            row.append(f"{results['large'][model][cls_type]['f1']:.1f}")
        table_data.append(row)
    
    # Add base models
    table_data.append(['Base-size Models', '', '', ''])
    for model in models.keys():
        row = [model]
        for cls_type in class_types:
            row.append(f"{results['base'][model][cls_type]['f1']:.1f}")
        table_data.append(row)
    
    print("\nResults Table (F1 Scores):")
    print(tabulate(table_data, headers=headers, tablefmt='grid'))

if __name__ == "__main__":
    run_experiments() 