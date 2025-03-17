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
    def __init__(self, tokenizer, data_path):
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.train_data = json.load(open(os.path.join(data_path, "train.json")))
        self.test_data = json.load(open(os.path.join(data_path, "task_1_testing.json")))
        
        # Create label mapping for BIO tags
        self.label2id = {
            'O': 0,
            'B': 1,
            'I': 1
        }
        
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

        def process_example(tokens, bio_tags):
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

            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)
                elif word_id != prev_word_id:
                    label_ids.append(self.label2id[bio_tags[word_id]])
                else:
                    if bio_tags[word_id] == 'B':
                        label_ids.append(self.label2id['I'])
                    else:
                        label_ids.append(self.label2id[bio_tags[word_id]])
                prev_word_id = word_id

            return {
                'input_ids': encoding['input_ids'][0],
                'attention_mask': encoding['attention_mask'][0],
                'labels': torch.tensor(label_ids)
            }

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
                    example = process_example(sent_tokens, bio_tags)
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
                    example = process_example(sent_tokens, bio_tags)
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

    print_results(test_results)
    
    return test_results

def print_results(results):
    # Print separate tables for token and entity level metrics
    for level in ['token', 'entity']:
        print(f"\n{level.upper()}-LEVEL METRICS")
        print(f"F1: {results[level]['f1']:.2f}")
        print(f"Precision: {results[level]['precision']:.2f}") 
        print(f"Recall: {results[level]['recall']:.2f}")


# TODO: WIP
def run_plaba_1b(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache_dir = "./cache_models"
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs("output/plaba", exist_ok=True)

    model = models[args.model_name]
    print(f"Loading the model: {model}")
    
    tokenizer = get_tokenizer(model, cache_dir=cache_dir)



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