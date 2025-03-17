# Maybe make a PNG after creating the html?
# apply a parser argument

import json
import html
import argparse
from transformers import AutoModelForTokenClassification
import torch
from torch.utils.data import DataLoader, Subset
import os
import random
from medreadme import MedReadmeDataset, get_tokenizer

# Define model mapping
MODELS = {
    'bert': 'bert-large-uncased',
    'roberta': 'roberta-large',
    'biobert': 'dmis-lab/biobert-large-cased-v1.1',
    'pubmedbert': 'microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract'
}

class JargonVisualizer:
    def __init__(self, model_path, model_name, classification_type):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classification_type = classification_type
        
        # Initialize tokenizer
        self.tokenizer = get_tokenizer(MODELS[model_name], cache_dir="./cache_models")
        
        # Get number of labels based on classification type
        num_labels = {'binary': 2, '3-cls': 4, '7-cls': 8}[classification_type]
        
        # Load model
        self.model = AutoModelForTokenClassification.from_pretrained(
            MODELS[model_name],
            num_labels=num_labels,
            cache_dir="./cache_models"
        )
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()

    def predict_batch(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels']  # Keep labels on CPU for comparison

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=2)

        # Get actual sequence lengths
        seq_lengths = attention_mask.sum(dim=1)
        
        all_predictions = []
        all_labels = []
        
        # Process each sequence in the batch
        for pred_seq, label_seq, length in zip(predictions, labels, seq_lengths):
            # Only keep predictions for actual tokens (no padding)
            pred_seq = pred_seq[:length].cpu().tolist()
            label_seq = label_seq[:length].cpu().tolist()
            
            # Filter out padding labels (-100)
            filtered_preds = []
            filtered_labels = []
            for p, l in zip(pred_seq, label_seq):
                if l != -100:  # Not a padding token
                    filtered_preds.append(p)
                    filtered_labels.append(l)
            
            all_predictions.append(filtered_preds)
            all_labels.append(filtered_labels)
        
        return all_predictions, all_labels

def create_html_visualization(tokens, predictions, labels, output_file, class_names):
    """
    Create HTML visualization using the original tokens instead of re-tokenizing the text
    """
    # Build text with proper token boundaries
    html_parts = []
    
    for token, pred, label in zip(tokens, predictions, labels):
        # Determine the class for this token
        classes = []
        if label != 0:  # True entity
            if pred == label:
                classes.append('correct-prediction')
            else:
                classes.append('wrong-prediction')
        elif pred != 0:  # False positive
            classes.append('false-positive')
        
        # Add the token with appropriate styling
        if classes:
            html_parts.append(f'<span class="{" ".join(classes)}">')
        
        # Escape special characters and preserve whitespace
        escaped_token = html.escape(token)
        if token.startswith('##'):  # Handle subwords for BERT-style tokenizers
            html_parts.append(escaped_token[2:])  # Remove '##' prefix
        else:
            # Add space before token unless it's punctuation
            if not token.startswith(('.',',','!','?',';',':',')',']','}')) and len(html_parts) > 0:
                html_parts.append(' ')
            html_parts.append(escaped_token)
            
        if classes:
            html_parts.append('</span>')

    text_html = ''.join(html_parts)
    
    html_content = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: 'Segoe UI', Arial, sans-serif;
                line-height: 1.8;
                max-width: 1000px;
                margin: 40px auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .content {{
                background-color: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .legend {{
                margin-bottom: 30px;
                padding: 15px;
                border-radius: 5px;
                background-color: white;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }}
            .legend span {{
                display: inline-block;
                margin: 5px 10px;
                padding: 5px 10px;
                border-radius: 4px;
            }}
            .correct-prediction {{
                background-color: #00FF0033;
                border-bottom: 2px solid #00FF00;
            }}
            .wrong-prediction {{
                background-color: #FF000033;
                border-bottom: 2px solid #FF0000;
            }}
            .false-positive {{
                background-color: #FF00FF33;
                border-bottom: 2px solid #FF00FF;
            }}
        </style>
    </head>
    <body>
        <div class="legend">
            <span class="correct-prediction">Correct Predictions</span>
            <span class="wrong-prediction">Wrong Predictions</span>
            <span class="false-positive">False Positives</span>
        </div>
        <div class="content">
            <p>{text_html}</p>
        </div>
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html_content)

def get_class_names(classification_type):
    if classification_type == 'binary':
        return ['O', 'COMPLEX']
    elif classification_type == '3-cls':
        return ['O', 'MEDICAL', 'ABBR', 'GENERAL']
    else:  # 7-cls
        return ['O', 'GOOGLE_EASY', 'GOOGLE_HARD', 'MEDICAL_NAME', 
                'MEDICAL_ABBR', 'GENERAL_ABBR', 'GENERAL_COMPLEX', 'MULTISENSE']

def print_sample_details(tokens, predictions, labels, class_names):
    """Print detailed analysis of a single sample"""
    print("\n" + "="*80)
    print("SAMPLE ANALYSIS")
    print("="*80)
    
    print("\nTokens with Predictions and Labels:")
    print("-"*80)
    print(f"{'Token':<30} {'Prediction':<20} {'True Label':<20}")
    print("-"*80)
    
    for token, pred, label in zip(tokens, predictions, labels):
        pred_name = class_names[pred]
        label_name = class_names[label]
        print(f"{token:<30} {pred_name:<20} {label_name:<20}")
    
    print("\nSummary:")
    print("-"*80)
    # Count matches and mismatches
    matches = sum(1 for p, l in zip(predictions, labels) if p == l)
    total = len(predictions)
    print(f"Total tokens: {total}")
    print(f"Correct predictions: {matches} ({matches/total:.2%})")
    print(f"Incorrect predictions: {total-matches} ({(total-matches)/total:.2%})")
    
    # Show confusion details
    print("\nErrors Analysis:")
    for pred, label, token in zip(predictions, labels, tokens):
        if pred != label:
            print(f"Token: '{token}'")
            print(f"  Predicted as: {class_names[pred]}")
            print(f"  True label: {class_names[label]}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, 
                        choices=['bert', 'roberta', 'biobert', 'pubmedbert'])
    parser.add_argument("--classification", type=str, required=True,
                        choices=['binary', '3-cls', '7-cls'])
    parser.add_argument("--data_dir", type=str, default="data/medreadme/jargon.json",
                        help="Path to data file")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of random samples to visualize")
    args = parser.parse_args()

    # Create output directory
    os.makedirs('output/visualization', exist_ok=True)

    # Load model
    model_path = f'output/medreadme/{args.model_name}_large_{args.classification}.pt'
    visualizer = JargonVisualizer(model_path, args.model_name, args.classification)

    # Create dataset
    dataset = MedReadmeDataset(
        visualizer.tokenizer,
        args.data_dir,
        classification_type=args.classification
    )
    test_data = dataset.get_split('test')
    
    # Randomly sample indices
    total_samples = len(test_data)
    sample_indices = random.sample(range(total_samples), min(args.num_samples, total_samples))
    
    # Create dataloader for sampled indices
    sampled_dataset = Subset(test_data, sample_indices)
    test_loader = DataLoader(sampled_dataset, batch_size=1)

    # Process sampled test data
    class_names = get_class_names(args.classification)
    
    print(f"\nAnalyzing {len(sample_indices)} random test examples...")
    
    for i, batch in enumerate(test_loader):
        predictions, labels = visualizer.predict_batch(batch)
        
        # Get original tokens
        tokens = visualizer.tokenizer.convert_ids_to_tokens(
            batch['input_ids'][0],
            skip_special_tokens=True
        )
        
        # Get actual sequence length (non-padding)
        seq_length = batch['attention_mask'][0].sum().item()
        
        # Print detailed token-level analysis
        print_sample_details(
            tokens[:seq_length],
            predictions[0],
            labels[0],
            class_names
        )
        
        # Create visualization using original tokens
        output_file = f'output/visualization/{args.model_name}_large_{args.classification}_sample{i}.html'
        create_html_visualization(
            tokens[:seq_length],
            predictions[0],
            labels[0],
            output_file,
            class_names
        )
        print(f"\nVisualization saved to: {output_file}")

if __name__ == "__main__":
    main() 