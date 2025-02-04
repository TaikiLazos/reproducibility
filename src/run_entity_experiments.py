import pandas as pd
import torch
from transformers import (
    BertForSequenceClassification, 
    RobertaForSequenceClassification,
    BertTokenizer,
    RobertaTokenizer,
)
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os
from tabulate import tabulate
import json
from torch.optim import AdamW
import numpy as np

class EntityLevelDataset(Dataset):
    def __init__(self, data_csv, data_json, tokenizer, max_length=128, classification_type='binary'):
        self.df = pd.read_csv(data_csv)
        # Load JSON data for entity information
        with open(data_json, 'r') as f:
            # Convert from list to dictionary for easier indexing
            json_list = json.load(f)
            self.json_data = {str(i): item for i, item in enumerate(json_list)}
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.classification_type = classification_type
        
        # Process all entities into a flat list
        self.entity_instances = self._process_entities()
        
    def _process_entities(self):
        entity_instances = []
        total_entities = 0
        
        for idx, row in self.df.iterrows():
            json_item = self.json_data[str(idx)]
            sentence = row.Sentence
            
            # Process each entity in the sentence
            for entity in json_item['entities']:
                total_entities += 1
                start_idx, end_idx, label, tokens = entity
                
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
                
                entity_text = ' '.join(tokens)
                
                entity_instances.append({
                    'sentence': sentence,
                    'entity': entity_text,
                    'label': processed_label
                })
        
        print(f"Dataset split: {self.df['split'].iloc[0]}")
        print(f"Total entities found: {total_entities}")
        
        return entity_instances
    
    def __len__(self):
        return len(self.entity_instances)
    
    def __getitem__(self, idx):
        instance = self.entity_instances[idx]
        
        # Tokenize the sentence
        encoding = self.tokenizer(
            instance['sentence'],
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(instance['label'], dtype=torch.long)
        }

def train_and_evaluate(model, train_loader, val_loader, device, model_save_path, num_labels):
    optimizer = AdamW(model.parameters(), lr=2e-6)
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
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device)
                )
                val_preds.extend(outputs.logits.argmax(dim=-1).cpu().numpy())
                val_labels.extend(batch['labels'].cpu().numpy())
        
        if len(val_preds) > 0 and len(val_labels) > 0:
            # Calculate accuracy
            accuracy = accuracy_score(val_labels, val_preds) * 100
            
            print(f"Validation Accuracy: {accuracy:.2f}%")
            
            # Print detailed classification report
            print("\nDetailed Classification Report:")
            print(classification_report(val_labels, val_preds))
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), model_save_path)
                print(f"New best model saved with Accuracy: {accuracy:.2f}%")
        else:
            print("Warning: No entities found in validation set!")
            accuracy = 0
    
    return best_accuracy

def run_experiments():
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
    print(f"Training JSON entries: {len(train_json)}")
    print(f"Validation JSON entries: {len(val_json)}")
    
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
        'BERT': (BertForSequenceClassification, BertTokenizer, 'bert-base-uncased', 'bert-large-uncased'),
        'RoBERTa': (RobertaForSequenceClassification, RobertaTokenizer, 'roberta-base', 'roberta-large'),
        'BioBERT': (BertForSequenceClassification, BertTokenizer, 'dmis-lab/biobert-base-cased-v1.2', 'dmis-lab/biobert-large-cased-v1.1'),
        'PubMedBERT': (BertForSequenceClassification, BertTokenizer, 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext', 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')
    }
    
    sizes = ['base', 'large']
    class_types = ['binary', '3-cls', '7-cls']
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
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
                train_dataset = EntityLevelDataset('data/splits/train.csv', 'data/splits/train.json', 
                                                 tokenizer, classification_type=cls_type)
                val_dataset = EntityLevelDataset('data/splits/val.csv', 'data/splits/val.json', 
                                               tokenizer, classification_type=cls_type)
                
                train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=16)
                
                # Train and evaluate
                save_path = f'output/{model_name}_{size}_{cls_type}.pt'
                accuracy = train_and_evaluate(model, train_loader, val_loader, device, save_path, num_labels)
                results[size][model_name][cls_type] = accuracy
                
                print(f"Completed {model_name}-{size} {cls_type}: {accuracy:.1f}%")
    
    # Save results
    with open('output/results.json', 'w') as f:
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
    run_experiments() 