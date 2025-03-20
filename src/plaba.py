import argparse
import torch
import os
import json
from transformers import AutoModelForTokenClassification, AutoTokenizer, AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')

models = {
    'bert': 'bert-large-cased',
    'roberta': 'roberta-large',
    'biobert': 'dmis-lab/biobert-large-cased-v1.1',
    'pubmedbert': 'microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract'
}

class PLABADataset(Dataset):
    def __init__(self, tokenizer, data_path, task='1a'):
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.task = task
        self.train_data = json.load(open(os.path.join(data_path, "train.json")))
        self.test_data = json.load(open(os.path.join(data_path, "task_1_testing.json")))
        
        # Create label mapping for BIO tags (task 1a)
        self.bio_label2id = {
            'O': 0,
            'B': 1,
            'I': 1
        }
        
        # Create label mapping for classification (task 1b)
        self.action_labels = ['SUBSTITUTE', 'EXPLAIN', 'GENERALIZE', 'OMIT', 'EXEMPLIFY']
        self.action_label2id = {label: i for i, label in enumerate(self.action_labels)}
        self.id2action_label = {i: label for i, label in enumerate(self.action_labels)}
        
        self.sent_tokenizer = nltk.sent_tokenize
        
        self.train_dataset, self.val_dataset, self.test_dataset = self.create_dataset()   

    def create_dataset(self):
        def preprocess_text(text):
            # Add spaces around punctuation for proper tokenization
            for punct in ['(', ')', '.', ',', ':', ';']:
                text = text.replace(punct, f' {punct} ')
            return ' '.join(text.split())

        def split_into_sentences(text):
            text = preprocess_text(text)
            sentences = self.sent_tokenizer(text)
            return [sent.split() for sent in sentences]

        def create_bio_tags(tokens, jargon_terms):
            tokens_lower = [t.lower() for t in tokens]
            bio_tags = ['O'] * len(tokens)
            
            for jargon in jargon_terms:
                jargon_tokens = jargon.lower().split()
                for i in range(len(tokens_lower) - len(jargon_tokens) + 1):
                    if tokens_lower[i:i+len(jargon_tokens)] == jargon_tokens:
                        bio_tags[i] = 'B'
                        for j in range(1, len(jargon_tokens)):
                            bio_tags[i+j] = 'I'
            
            return bio_tags

        def get_action_labels(jargon, jargon_dict):
            """Get unique action labels for a jargon term"""
            if jargon not in jargon_dict:
                return []
            
            # Get unique action types from all alternatives
            action_types = set()
            for alt in jargon_dict[jargon]:
                action_types.add(alt[0])  # alt[0] is the action type
            
            # Convert to multi-hot encoding
            multi_hot = [0] * len(self.action_labels)
            for action in action_types:
                if action in self.action_label2id:
                    multi_hot[self.action_label2id[action]] = 1
            
            return multi_hot

        def process_example(tokens, bio_tags, jargon_terms=None, jargon_dict=None):
            encoding = self.tokenizer(
                tokens,
                is_split_into_words=True,
                padding='max_length',
                truncation=True,
                max_length=192,  # Increased from 128 to 192 to cover max length of 171
                return_tensors='pt'
            )

            word_ids = encoding.word_ids()
            label_ids = []
            prev_word_id = None
            action_labels = None

            # Process BIO tags
            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)
                elif word_id != prev_word_id:
                    label_ids.append(self.bio_label2id[bio_tags[word_id]])
                else:
                    if bio_tags[word_id] == 'B':
                        label_ids.append(self.bio_label2id['I'])
                    else:
                        label_ids.append(self.bio_label2id[bio_tags[word_id]])
                prev_word_id = word_id

            # Process action labels for task 1b
            if self.task == '1b' and jargon_terms is not None and jargon_dict is not None:
                action_labels = []
                for word_id in word_ids:
                    if word_id is None:
                        action_labels.append([0] * len(self.action_labels))
                    else:
                        # Find if this token is part of a jargon term
                        current_pos = 0
                        for term in jargon_terms:
                            term_tokens = term.lower().split()
                            if word_id < len(tokens) and tokens[word_id].lower() in term_tokens:
                                action_labels.append(get_action_labels(term, jargon_dict))
                                break
                        else:
                            action_labels.append([0] * len(self.action_labels))

            example = {
                'input_ids': encoding['input_ids'][0],
                'attention_mask': encoding['attention_mask'][0],
                'labels': torch.tensor(label_ids)
            }
            
            if action_labels is not None:
                example['action_labels'] = torch.tensor(action_labels)
            
            return example

        # Process training data
        train_examples = []
        for doc_id, jargon_dict in self.train_data.items():
            with open(f"{self.data_path}/abstracts/{doc_id}.src.txt", 'r') as f:
                text = f.read()
                sentences = split_into_sentences(text)
                
                for sent_tokens in sentences:
                    if not sent_tokens:  # Skip empty sentences
                        continue
                    bio_tags = create_bio_tags(sent_tokens, jargon_dict.keys())
                    example = process_example(
                        sent_tokens, 
                        bio_tags,
                        jargon_dict.keys() if self.task == '1b' else None,
                        jargon_dict if self.task == '1b' else None
                    )
                    train_examples.append(example)

        # Process test data
        test_set = []
        for doc_id, jargon_dict in self.test_data.items():
            with open(f"{self.data_path}/abstracts/{doc_id}.src.txt", 'r') as f:
                text = f.read()
                sentences = split_into_sentences(text)
                
                for sent_tokens in sentences:
                    if not sent_tokens:
                        continue
                    bio_tags = create_bio_tags(sent_tokens, jargon_dict.keys())
                    example = process_example(
                        sent_tokens, 
                        bio_tags,
                        jargon_dict.keys() if self.task == '1b' else None,
                        jargon_dict if self.task == '1b' else None
                    )
                    test_set.append(example)

        # Split training data into train and validation sets
        split_idx = int(len(train_examples) * 0.9)
        train_set = train_examples[:split_idx]
        val_set = train_examples[split_idx:]
        
        return train_set, val_set, test_set

    def __len__(self):
        return len(self.train_dataset)

    def __getitem__(self, idx):
        return self.train_dataset[idx]

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

def calculate_metrics(all_predictions, all_labels, level='entity'):
    """Helper function to calculate metrics for both token and entity level"""
    tp = fp = fn = 0
    
    if level == 'token':
        for preds, labels in zip(all_predictions, all_labels):
            for p, l in zip(preds, labels):
                if l == -100:  # Skip padding tokens
                    continue
                    
                # Only count predictions for actual entity tokens (non-O)
                if l != 0:  # If it's a true entity token
                    if p == l:  # Correct prediction
                        tp += 1
                    else:  # Wrong prediction
                        fn += 1
                elif p != 0:  # False positive: predicted entity when there wasn't one
                    fp += 1
                # Note: We don't count O->O predictions
    else:  # entity-level metrics remain the same since they already only consider entities
        for preds, labels in zip(all_predictions, all_labels):
            pred_entities = extract_entities(preds)
            true_entities = extract_entities(labels)
            
            # Count matches
            for pred_ent in pred_entities:
                if pred_ent in true_entities:
                    tp += 1
                else:
                    fp += 1
            
            for true_ent in true_entities:
                if true_ent not in pred_entities:
                    fn += 1
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'f1': f1 * 100,
        'precision': precision * 100,
        'recall': recall * 100
    }


def train(model, train_loader, val_loader, device, args, save_path):
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    best_f1 = 0
    patience = 5    # for early stopping
    patience_counter = 0
    max_epochs = args.num_epochs
    # max_epochs = 1

    for epoch in range(max_epochs): 
        model.train()
        total_loss = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{max_epochs}')
        for batch in progress_bar:
            optimizer.zero_grad()
            
            # Move everything to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            # print(outputs.logits.shape) # torch.Size([32, 512, 2])
            # print(batch['labels'].shape) # torch.Size([32, 512])
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # Update progress bar with current loss
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1}/{max_epochs}, Average Loss: {avg_loss:.4f}")

        # Evaluate it on the validation set
        model.eval()
        all_predictions = []
        all_labels = []
        
        # Add progress bar for validation
        val_progress = tqdm(val_loader, desc='Validating')
        
        with torch.no_grad():
            for batch in val_progress:
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
                    
                    all_predictions.append(pred_seq)
                    all_labels.append(label_seq)
        
        # Always calculate entity-level metrics for model selection
        results = calculate_metrics(all_predictions, all_labels, level='entity')

        print(f"\nValidation Results:")
        print(f"Overall F1: {results['f1']:.2f}")
        print(f"Precision: {results['precision']:.2f}")
        print(f"Recall: {results['recall']:.2f}")
        
        # Early stopping check using entity-level F1
        current_f1 = results['f1']
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
            

def test(model_path, test_loader, device, model):
    num_labels = 2
    # Create new model with same config as current_model
    model = AutoModelForTokenClassification.from_config(model.config)
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
    
    # Calculate metrics for both levels
    token_results = calculate_metrics(all_predictions, all_labels, level='token')
    entity_results = calculate_metrics(all_predictions, all_labels, level='entity')
    
    return {
        'token': token_results,
        'entity': entity_results
    }

def run_plaba_1a(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache_dir = "./cache_models"
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs("output/plaba", exist_ok=True)

    model = models[args.model_name]
    print(f"Loading the model: {model}")
    
    tokenizer = get_tokenizer(model, cache_dir=cache_dir)
    model = AutoModelForTokenClassification.from_pretrained(
        model, 
        num_labels=2,  # Since we only want to detect the jargon terms
        cache_dir=cache_dir
    )
    model.to(device)

    print("Starting training and evaluation...")
    print(f"Using device: {device}")

    # Create datasets
    dataset = PLABADataset(tokenizer, args.data_dir)
    
    # Create data loaders
    train_loader = DataLoader(dataset.train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset.val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(dataset.test_dataset, batch_size=args.batch_size)
    # Train the model
    save_path = f'output/plaba/{args.model_name}_1a.pt'
    train(model, train_loader, val_loader, device, args, save_path)

    # Test the best model
    test_results = test(save_path, test_loader, device, model)

    print_results(test_results, '1a')
    
    return test_results

def print_results(results, task='1a'):
    """Print results in a formatted table"""
    if task == '1a':
        # For task 1a (detection), show both token and entity level metrics
        for level in ['token', 'entity']:
            print(f"\n{level.upper()}-LEVEL METRICS")
            print(f"F1: {results[level]['f1']:.2f}")
            print(f"Precision: {results[level]['precision']:.2f}") 
            print(f"Recall: {results[level]['recall']:.2f}")
    else:
        # For task 1b (classification), show overall and per-action metrics
        print("\nENTITY-LEVEL METRICS")
        print("\nOverall Metrics:")
        print(f"F1: {results['f1']:.2f}")
        print(f"Precision: {results['precision']:.2f}")
        print(f"Recall: {results['recall']:.2f}")
        
        print("\nPer-Action Metrics:")
        for action in ['SUBSTITUTE', 'EXPLAIN', 'GENERALIZE', 'OMIT', 'EXEMPLIFY']:
            print(f"\n{action}:")
            print(f"F1: {results[action]['f1']:.2f}")
            print(f"Precision: {results[action]['precision']:.2f}")
            print(f"Recall: {results[action]['recall']:.2f}")


def run_plaba_1b(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache_dir = "./cache_models"
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs("output/plaba", exist_ok=True)

    model = models[args.model_name]
    print(f"Loading the model: {model}")
    
    tokenizer = get_tokenizer(model, cache_dir=cache_dir)
    
    # Initialize the model with number of action labels
    model = AutoModelForTokenClassification.from_pretrained(
        model,
        num_labels=5,  # 5 action types
        cache_dir=cache_dir
    )
    
    # Modify the model's forward pass to use sigmoid activation
    def forward_with_sigmoid(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Get the base model (works for both BERT and RoBERTa)
        base_model = getattr(self, 'bert', getattr(self, 'roberta', None))
        if base_model is None:
            raise ValueError("Model architecture not supported. Must be BERT or RoBERTa.")
            
        outputs = base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        probs = torch.sigmoid(logits)
        
        if labels is not None:
            loss_fct = torch.nn.BCELoss()
            loss = loss_fct(probs, labels.float())
            return loss, probs
        return probs
    
    # Replace the model's forward method
    model.forward = forward_with_sigmoid.__get__(model, model.__class__)
    model.to(device)

    print("Starting training and evaluation...")
    print(f"Using device: {device}")

    # Create datasets with task='1b'
    dataset = PLABADataset(tokenizer, args.data_dir, task='1b')
    
    # Create data loaders
    train_loader = DataLoader(dataset.train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset.val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(dataset.test_dataset, batch_size=args.batch_size)

    # Modified training function for multi-label classification
    def train_1b(model, train_loader, val_loader, device, args, save_path):
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
        best_f1 = 0
        patience = 5
        patience_counter = 0
        max_epochs = args.num_epochs

        for epoch in range(max_epochs):
            model.train()
            total_loss = 0

            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{max_epochs}')
            for batch in progress_bar:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                action_labels = batch['action_labels'].to(device)
                
                loss, _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=action_labels
                )
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_loss = total_loss / len(train_loader)
            print(f"\nEpoch {epoch+1}/{max_epochs}, Average Loss: {avg_loss:.4f}")

            # Evaluate on validation set
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
                    predictions = (outputs > 0.5).float()
                    
                    # Get actual length of each sequence (ignore padding)
                    mask = batch['attention_mask'].bool()
                    
                    # Collect predictions and labels for each sequence
                    for pred_seq, label_seq, seq_mask in zip(predictions, batch['action_labels'], mask):
                        length = seq_mask.sum().item()
                        
                        # Only keep predictions and labels for actual tokens (no padding)
                        pred_seq = pred_seq[:length].cpu().numpy()
                        label_seq = label_seq[:length].cpu().numpy()
                        
                        all_predictions.append(pred_seq)
                        all_labels.append(label_seq)
            
            # Calculate metrics for multi-label classification
            results = calculate_multi_label_metrics(all_predictions, all_labels)
            
            print(f"\nValidation Results:")
            print(f"Overall F1: {results['f1']:.2f}")
            print(f"Precision: {results['precision']:.2f}")
            print(f"Recall: {results['recall']:.2f}")
            
            # Early stopping check
            current_f1 = results['f1']
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

    def calculate_multi_label_metrics(predictions, labels):
        """Calculate metrics for multi-label classification"""
        # Initialize counters for overall metrics
        tp = fp = fn = 0
        
        # Initialize counters for per-action metrics
        action_metrics = {
            'SUBSTITUTE': {'tp': 0, 'fp': 0, 'fn': 0},
            'EXPLAIN': {'tp': 0, 'fp': 0, 'fn': 0},
            'GENERALIZE': {'tp': 0, 'fp': 0, 'fn': 0},
            'OMIT': {'tp': 0, 'fp': 0, 'fn': 0},
            'EXEMPLIFY': {'tp': 0, 'fp': 0, 'fn': 0}
        }
        
        for pred, label in zip(predictions, labels):
            # For each label
            for i, action in enumerate(['SUBSTITUTE', 'EXPLAIN', 'GENERALIZE', 'OMIT', 'EXEMPLIFY']):
                pred_labels = pred[:, i]
                true_labels = label[:, i]
                
                # Count for overall metrics
                tp += ((pred_labels == 1) & (true_labels == 1)).sum()
                fp += ((pred_labels == 1) & (true_labels == 0)).sum()
                fn += ((pred_labels == 0) & (true_labels == 1)).sum()
                
                # Count for per-action metrics
                action_metrics[action]['tp'] += ((pred_labels == 1) & (true_labels == 1)).sum()
                action_metrics[action]['fp'] += ((pred_labels == 1) & (true_labels == 0)).sum()
                action_metrics[action]['fn'] += ((pred_labels == 0) & (true_labels == 1)).sum()
        
        # Calculate overall metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate per-action metrics
        results = {
            'f1': f1 * 100,
            'precision': precision * 100,
            'recall': recall * 100
        }
        
        # Add per-action metrics
        for action, metrics in action_metrics.items():
            action_precision = metrics['tp'] / (metrics['tp'] + metrics['fp']) if (metrics['tp'] + metrics['fp']) > 0 else 0
            action_recall = metrics['tp'] / (metrics['tp'] + metrics['fn']) if (metrics['tp'] + metrics['fn']) > 0 else 0
            action_f1 = 2 * action_precision * action_recall / (action_precision + action_recall) if (action_precision + action_recall) > 0 else 0
            
            results[action] = {
                'f1': action_f1 * 100,
                'precision': action_precision * 100,
                'recall': action_recall * 100
            }
        
        return results

    def test_1b(model_path, test_loader, device, model):
        # Load the saved state dict
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()
        
        all_predictions = []
        all_labels = []
        
        test_progress = tqdm(test_loader, desc='Testing')
        
        with torch.no_grad():
            for batch in test_progress:
                outputs = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device)
                )
                predictions = (outputs > 0.5).float()
                
                # Get actual length of each sequence (ignore padding)
                mask = batch['attention_mask'].bool()
                
                # Collect predictions and labels for each sequence
                for pred_seq, label_seq, seq_mask in zip(predictions, batch['action_labels'], mask):
                    length = seq_mask.sum().item()
                    
                    # Only keep predictions and labels for actual tokens (no padding)
                    pred_seq = pred_seq[:length].cpu().numpy()
                    label_seq = label_seq[:length].cpu().numpy()
                    
                    all_predictions.append(pred_seq)
                    all_labels.append(label_seq)
        
        # Calculate metrics
        results = calculate_multi_label_metrics(all_predictions, all_labels)
        
        return results

    # Train the model
    save_path = f'output/plaba/{args.model_name}_1b.pt'
    train_1b(model, train_loader, val_loader, device, args, save_path)

    # Test the best model
    test_results = test_1b(save_path, test_loader, device, model)
    print_results(test_results, '1b')
    
    return test_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="roberta", help="Name of the pretrained model to use")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--experiment_type", type=str, default="1a", choices=["1a", "1b"], help="1a: detection, 1b: classification")
    parser.add_argument("--data_dir", type=str, default="data/PLABA_2024-Task_1", help="Directory containing the data files")
    args = parser.parse_args()

    if args.experiment_type == '1a':
        run_plaba_1a(args)
    elif args.experiment_type == '1b':
        run_plaba_1b(args)
    else:
        print("Something went wrong.")