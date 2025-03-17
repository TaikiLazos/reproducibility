import argparse
import torch
import os
from transformers import AutoModelForTokenClassification, AutoTokenizer
import json
from torch.utils.data import DataLoader, Dataset
from tabulate import tabulate
from torch.optim import AdamW
from tqdm import tqdm

# Define available models
models = {
    'bert': ('bert-base-uncased', 'bert-large-uncased'),
    'roberta': ('roberta-base', 'roberta-large'),
    'biobert': ('dmis-lab/biobert-base-cased-v1.1', 'dmis-lab/biobert-large-cased-v1.1'),
    'pubmedbert': ('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext', 'microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract')
}


class MedReadmeDataset(Dataset):
    def __init__(self, tokenizer, data_path, classification_type='binary'):
        self.tokenizer = tokenizer
        self.classification_type = classification_type
        self.data_path = data_path
        
        # Create label mapping
        self.label2id = self._create_label_mapping()
        
        # Initialize split data structure
        self.split_data = {
            'train': [],
            'dev': [],
            'test': []
        }
        
        # Process and split the data
        self.create_dataset()
    
    def _create_label_mapping(self):
        """Create mapping from labels to IDs"""
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
                'B-GOOGLE_EASY': 1, 'I-GOOGLE_EASY': 1,
                'B-GOOGLE_HARD': 2, 'I-GOOGLE_HARD': 2,
                'B-MEDICAL_NAME': 3, 'I-MEDICAL_NAME': 3,
                'B-MEDICAL_ABBR': 4, 'I-MEDICAL_ABBR': 4,
                'B-GENERAL_ABBR': 5, 'I-GENERAL_ABBR': 5,
                'B-GENERAL_COMPLEX': 6, 'I-GENERAL_COMPLEX': 6,
                'B-MULTISENSE': 7, 'I-MULTISENSE': 7
            }

    def create_dataset(self):
        """Creates datasets from JSON file using the 'split' field"""
        # Load data
        with open(self.data_path, 'r') as f:
            all_data = json.load(f)
        
        # Process each example
        for item in all_data:
            # Get split type (default to train if not specified)
            split = item.get('split', 'train')
            
            # Process the item
            example = self._process_item(item)
            
            # Add to appropriate split
            self.split_data[split].append(example)
    
    def _process_item(self, item):
        """Process a single data item"""
        tokens = item['tokens']
        entities = item['entities']
        
        # Initialize labels
        labels = ['O'] * len(tokens)
        
        # Fill in entity labels
        for start, end, label, _ in entities:
            # Convert label based on classification type
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
                # Map the original labels to our 7 classes
                label_lower = label.lower()
                if 'medical-jargon-google-easy' in label_lower:
                    bio_tag = 'GOOGLE_EASY'
                elif 'medical-jargon-google-hard' in label_lower:
                    bio_tag = 'GOOGLE_HARD'
                elif 'medical-name-entity' in label_lower:
                    bio_tag = 'MEDICAL_NAME'
                elif 'abbr-medical' in label_lower:
                    bio_tag = 'MEDICAL_ABBR'
                elif 'abbr-general' in label_lower:
                    bio_tag = 'GENERAL_ABBR'
                elif 'general-complex' in label_lower:
                    bio_tag = 'GENERAL_COMPLEX'
                else:
                    bio_tag = 'MULTISENSE'
            # Apply BIO tags
            labels[start] = f'B-{bio_tag}'
            for i in range(start + 1, end):
                labels[i] = f'I-{bio_tag}'
        
        # Tokenize
        # Max length: 200
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=250,
            return_tensors='pt'
        )
        
        # Align labels with subwords
        word_ids = encoding.word_ids()
        label_ids = []
        prev_word_id = None
        
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word_id:
                label_ids.append(self.label2id.get(labels[word_id], 0))
            else:
                if labels[word_id].startswith(('B-', 'I-')):
                    label = 'I-' + labels[word_id].split('-')[1]
                    label_ids.append(self.label2id.get(label, 0))
                else:
                    label_ids.append(self.label2id.get('O', 0))
            prev_word_id = word_id
        
        return {
            'input_ids': encoding['input_ids'][0],
            'attention_mask': encoding['attention_mask'][0],
            'labels': torch.tensor(label_ids)
        }
    
    def __len__(self):
        """Return length of training set by default"""
        return len(self.split_data['train'])
    
    def __getitem__(self, idx):
        """Get item from training set by default"""
        return self.split_data['train'][idx]
    
    def get_split(self, split='train'):
        """Get a specific data split"""
        return self.split_data[split]


def calculate_metrics(all_predictions, all_labels, level='entity', num_labels=2):
    """Helper function to calculate metrics for both token and entity level with per-class metrics"""
    # Initialize per-class counters
    class_metrics = {i: {'tp': 0, 'fp': 0, 'fn': 0} for i in range(num_labels)}
    
    if level == 'token':
        for preds, labels in zip(all_predictions, all_labels):
            for p, l in zip(preds, labels):
                if l == -100:  # Skip padding tokens
                    continue
                
                if l != 0:  # If it's a true entity token
                    if p == l:  # Correct prediction
                        class_metrics[l]['tp'] += 1
                    else:  # Wrong prediction
                        class_metrics[l]['fn'] += 1
                        if p != 0:  # If predicted as different class
                            class_metrics[p]['fp'] += 1
                elif p != 0:  # False positive: predicted entity when there wasn't one
                    class_metrics[p]['fp'] += 1
    else:  # entity-level
        for preds, labels in zip(all_predictions, all_labels):
            pred_entities = extract_entities(preds)
            true_entities = extract_entities(labels)
            
            # Group entities by class
            for ent in pred_entities:
                start, end, class_id = ent
                if ent in true_entities:
                    class_metrics[class_id]['tp'] += 1
                else:
                    class_metrics[class_id]['fp'] += 1
            
            for ent in true_entities:
                start, end, class_id = ent
                if ent not in pred_entities:
                    class_metrics[class_id]['fn'] += 1
    
    # Calculate per-class metrics
    results = {'overall': {'f1': 0, 'precision': 0, 'recall': 0}}
    total_tp = total_fp = total_fn = 0
    
    for class_id, metrics in class_metrics.items():
        tp, fp, fn = metrics['tp'], metrics['fp'], metrics['fn']
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results[f'class_{class_id}'] = {
            'f1': f1 * 100,
            'precision': precision * 100,
            'recall': recall * 100
        }
    
    # Calculate overall metrics
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    results['overall'] = {
        'f1': overall_f1 * 100,
        'precision': overall_precision * 100,
        'recall': overall_recall * 100
    }
    
    return results

def test(model_path, test_loader, device, current_model):
    """Evaluate model on test set for both token and entity level"""
    # Extract the classification type from the model path
    if 'binary' in model_path:
        num_labels = 2
    elif '3-cls' in model_path:
        num_labels = 4
    elif '7-cls' in model_path:
        num_labels = 8
    else:
        raise ValueError(f"Cannot determine number of labels from model path: {model_path}")

    # Create new model with same config as current_model
    model = AutoModelForTokenClassification.from_config(current_model.config)
    # Set the correct number of labels before loading state dict
    model.classifier = torch.nn.Linear(model.config.hidden_size, num_labels)
    model.num_labels = num_labels
    # Load the saved state dict
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    # Add progress bar for testing
    test_progress = tqdm(test_loader, desc='Testing')
    
    # Add debugging information
    label_counts = {i: 0 for i in range(num_labels)}
    pred_counts = {i: 0 for i in range(num_labels)}
    
    with torch.no_grad():
        for batch in test_progress:
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device)
            )
            predictions = torch.argmax(outputs.logits, dim=2)
            
            # Get actual length of each sequence (ignore padding)
            mask = batch['attention_mask'].bool()
            
            # Collect predictions and labels for each sequence
            for pred_seq, label_seq, seq_mask in zip(predictions, batch['labels'], mask):
                length = seq_mask.sum().item()
                
                # Only keep predictions and labels for actual tokens (no padding)
                pred_seq = pred_seq[:length].cpu().tolist()
                label_seq = label_seq[:length].cpu().tolist()
                
                # Count label distributions
                for label in label_seq:
                    if label != -100:  # Ignore padding
                        label_counts[label] += 1
                
                # Count prediction distributions
                for pred in pred_seq:
                    pred_counts[pred] += 1
                
                all_predictions.append(pred_seq)
                all_labels.append(label_seq)
    
    # Print debugging information
    print("\nDebugging Information:")
    print(f"Model path: {model_path}")
    print(f"Number of labels: {num_labels}")
    
    print("\nTrue Label Distribution:")
    total_labels = sum(label_counts.values())
    for label, count in label_counts.items():
        percentage = (count / total_labels) * 100
        print(f"Label {label}: {count} ({percentage:.2f}%)")
    
    print("\nPredicted Label Distribution:")
    total_preds = sum(pred_counts.values())
    for pred, count in pred_counts.items():
        percentage = (count / total_preds) * 100
        print(f"Label {pred}: {count} ({percentage:.2f}%)")
    
    # Calculate metrics for both levels with num_labels
    token_results = calculate_metrics(all_predictions, all_labels, level='token', num_labels=num_labels)
    entity_results = calculate_metrics(all_predictions, all_labels, level='entity', num_labels=num_labels)
    
    return {
        'token': token_results,
        'entity': entity_results
    }

def train_and_evaluate(model, train_loader, val_loader, device, save_path, args):
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    best_f1 = 0
    patience = 3    # for early stopping
    patience_counter = 0
    max_epochs = args.num_epochs

    for epoch in range(max_epochs):
        # Training
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{max_epochs}')
        
        for batch in progress_bar:
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
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1}/{max_epochs}, Average Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        all_predictions = []
        all_labels = []
        
        val_progress = tqdm(val_loader, desc='Validating')
        
        with torch.no_grad():
            for batch in val_progress:
                outputs = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device)
                )
                predictions = torch.argmax(outputs.logits, dim=2)
                
                mask = batch['attention_mask'].bool()
                
                for pred_seq, label_seq, seq_mask in zip(predictions, batch['labels'], mask):
                    length = seq_mask.sum().item()
                    pred_seq = pred_seq[:length].cpu().tolist()
                    label_seq = label_seq[:length].cpu().tolist()
                    all_predictions.append(pred_seq)
                    all_labels.append(label_seq)
        
        # Calculate entity-level metrics for model selection
        results = calculate_metrics(all_predictions, all_labels, level='entity', num_labels=model.num_labels)

        print(f"\nValidation Results:")
        print(f"Overall F1: {results['overall']['f1']:.2f}")
        print(f"Overall Precision: {results['overall']['precision']:.2f}")
        print(f"Overall Recall: {results['overall']['recall']:.2f}")
        
        # Print per-class metrics
        for class_key in sorted([k for k in results.keys() if k != 'overall']):
            print(f"\n{class_key}:")
            print(f"F1: {results[class_key]['f1']:.2f}")
            print(f"Precision: {results[class_key]['precision']:.2f}")
            print(f"Recall: {results[class_key]['recall']:.2f}")
        
        # Early stopping check using overall entity-level F1
        current_f1 = results['overall']['f1']
        if current_f1 > best_f1:
            best_f1 = current_f1
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved with F1: {current_f1:.2f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    return best_f1

def extract_entities(label_seq):
    """Helper function to extract entity spans from label sequence"""
    entities = set()
    current_entity = None
    start_idx = None
    
    for i, label in enumerate(label_seq):
        if label == -100:  # Skip special tokens
            continue
        
        if label == 0:  # O tag
            if current_entity is not None:
                entities.add((start_idx, i, current_entity))  # Add the complete entity
                current_entity = None
                start_idx = None
        else:  # Entity tag
            if current_entity is None:  # Start of new entity
                current_entity = label
                start_idx = i
            elif current_entity != label:  # Change of entity type
                entities.add((start_idx, i, current_entity))
                current_entity = label
                start_idx = i
    
    # Add final entity if sequence ended with one
    if current_entity is not None:
        entities.add((start_idx, len(label_seq), current_entity))
    
    return entities

def get_tokenizer(model_name, cache_dir):
    """
    Initialize the appropriate tokenizer with correct settings based on model type
    """
    # RoBERTa models need add_prefix_space=True
    if 'roberta' in model_name.lower():
        return AutoTokenizer.from_pretrained(model_name, add_prefix_space=True, cache_dir=cache_dir)
    # Other models (BERT, BioBERT, PubMedBERT) don't need this parameter
    else:
        return AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

def run_experiments(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache_dir = "./cache_models"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Add this line to create output directory
    os.makedirs("output/medreadme", exist_ok=True)

    # Get both base and large model names
    base_model, large_model = models[args.model_name]
    
    print(f"Loading base model: {base_model}")
    base_tokenizer = get_tokenizer(base_model, cache_dir=cache_dir)
    base_model = AutoModelForTokenClassification.from_pretrained(base_model, cache_dir=cache_dir)
    base_model.to(device)

    print(f"Loading large model: {large_model}")
    large_tokenizer = get_tokenizer(large_model, cache_dir=cache_dir)
    large_model = AutoModelForTokenClassification.from_pretrained(large_model, cache_dir=cache_dir)
    large_model.to(device)

    sizes = ['base', 'large']
    class_types = ['binary', '3-cls', '7-cls']

    # Modify the results dictionary structure to store both token and entity metrics
    results = {
        size: {
            cls_type: {
                'token': {'f1': 0, 'precision': 0, 'recall': 0},
                'entity': {'f1': 0, 'precision': 0, 'recall': 0}
            } for cls_type in class_types
        } for size in sizes
    }

    for size in sizes:
        # Select appropriate model and tokenizer
        if size == 'base':
            model = base_model
            tokenizer = base_tokenizer
        else:
            model = large_model
            tokenizer = large_tokenizer
            
        for cls_type in class_types:
            print(f"\nRunning experiment for {args.model_name}-{size} {cls_type}")
            
            num_labels = {
                'binary': 2,
                '3-cls': 4, 
                '7-cls': 8
            }[cls_type]
            
            # Reset model classification head for different number of labels
            model.classifier = torch.nn.Linear(model.config.hidden_size, num_labels)
            model.num_labels = num_labels
            model.to(device)
            
            dataset = MedReadmeDataset(tokenizer, args.data_dir, classification_type=cls_type)
            train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(dataset.get_split('dev'), batch_size=args.batch_size)
            test_loader = DataLoader(dataset.get_split('test'), batch_size=args.batch_size)

            save_path = f'output/medreadme/{args.model_name}_{size}_{cls_type}.pt'
            train_and_evaluate(model, train_loader, val_loader, device, save_path, args)    

            # Test the best model
            test_results = test(save_path, test_loader, device, model)
            
            # Store both token and entity results
            results[size][cls_type]['token'] = test_results['token']
            results[size][cls_type]['entity'] = test_results['entity']
            
            print(f"Completed {args.model_name}-{size} {cls_type}")
    
    return results

def print_results(results):
    # Print separate tables for token and entity level metrics
    for level in ['token', 'entity']:
        print(f"\n{level.upper()}-LEVEL METRICS")
        
        for size in ['large', 'base']:
            print(f"\n{size.upper()}-SIZE MODELS")
            
            for cls_type in ['binary', '3-cls', '7-cls']:
                print(f"\n{cls_type} Classification:")
                headers = ['Class', 'F1', 'Precision', 'Recall']
                table_data = []
                
                metrics = results[size][cls_type][level]
                # Add overall metrics
                table_data.append(['Overall',
                                 f"{metrics['overall']['f1']:.2f}",
                                 f"{metrics['overall']['precision']:.2f}",
                                 f"{metrics['overall']['recall']:.2f}"])
                
                # Add per-class metrics
                for class_key in sorted([k for k in metrics.keys() if k != 'overall']):
                    table_data.append([class_key,
                                     f"{metrics[class_key]['f1']:.2f}",
                                     f"{metrics[class_key]['precision']:.2f}",
                                     f"{metrics[class_key]['recall']:.2f}"])
                
                print(tabulate(table_data, headers=headers, tablefmt='grid'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="roberta", help="Name of the pretrained model to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--data_dir", type=str, default="data/medreadme/jargon.json", help="Directory containing the data files")

    args = parser.parse_args()
    results = run_experiments(args)
    print_results(results)