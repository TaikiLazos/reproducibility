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
from plaba import PLABADataset
import nltk

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

def create_input_visualization(all_examples, output_file):
    """
    Create HTML visualization of input examples with jargon spans highlighted in green
    All examples from both datasets are shown in one HTML file, with sentences concatenated
    """
    html_content = """
    <html>
    <head>
        <style>
            body {
                font-family: 'Segoe UI', Arial, sans-serif;
                line-height: 1.8;
                max-width: 1200px;
                margin: 40px auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .content {
                background-color: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 30px;
            }
            .dataset-title {
                font-size: 28px;
                color: #333;
                margin-bottom: 30px;
                padding-bottom: 10px;
                border-bottom: 2px solid #eee;
            }
            .jargon-span {
                background-color: #90EE9033;
                border-bottom: 2px solid #90EE90;
                padding: 2px 0;
            }
            .dataset-section {
                margin-bottom: 40px;
            }
            p {
                word-spacing: 0.1em;
                line-height: 1.8;
                text-align: justify;
            }
            .sentence {
                display: inline;
            }
        </style>
    </head>
    <body>
    """
    
    # Process each dataset's examples
    for dataset_name, examples in all_examples.items():
        html_content += f"""
        <div class="dataset-section">
            <div class="dataset-title">{dataset_name} Dataset</div>
            <div class="content"><p>
        """
        
        # Concatenate all sentences with proper spacing
        for i, (text, terms) in enumerate(examples):
            # Sort terms by start index to process them in order
            terms = sorted(terms, key=lambda x: x[0])
            
            last_end = 0
            for start, end in terms:
                # Add text before the jargon term
                html_content += html.escape(text[last_end:start])
                # Add the highlighted jargon term
                html_content += f'<span class="jargon-span">{html.escape(text[start:end])}</span>'
                last_end = end
            
            # Add any remaining text
            if last_end < len(text):
                html_content += html.escape(text[last_end:])
            
            # Add space between sentences
            if i < len(examples) - 1:
                html_content += " "
        
        html_content += "</p></div></div>"
    
    html_content += """
    </body>
    </html>
    """
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

def visualize_dataset_examples(tokenizer, num_samples=3):
    """
    Visualize first N examples from both PLABA and MedReadme datasets
    """
    # Create datasets
    plaba_dataset = PLABADataset(tokenizer, 'data/PLABA_2024-Task_1')
    medreadme_dataset = MedReadmeDataset(tokenizer, 'data/medreadme/jargon.json')
    
    # Dictionary to store examples from both datasets
    all_examples = {
        'PLABA': [],
        'MedReadme': []
    }
    
    def get_word_boundaries(text):
        """Get word boundaries to ensure we only match at word boundaries"""
        boundaries = set()
        boundaries.add(0)  # Start of text
        boundaries.add(len(text))  # End of text
        
        for i in range(len(text)):
            if text[i].isspace() or text[i] in '.,;:!?()[]{}"\'':
                boundaries.add(i)
                if i + 1 < len(text):
                    boundaries.add(i + 1)
        
        return boundaries
    
    # Function to process examples from a dataset
    def process_dataset_examples(dataset, dataset_name):
        if dataset_name == 'PLABA':
            # For PLABA, we need to process raw data
            test_data = json.load(open(os.path.join('data/PLABA_2024-Task_1', "task_1_testing.json")))
            sentences_processed = 0
            
            for doc_id, jargon_dict in test_data.items():
                if sentences_processed >= num_samples:
                    break
                    
                # Read the abstract text
                with open(f"data/PLABA_2024-Task_1/abstracts/{doc_id}.src.txt", 'r') as f:
                    text = f.read()
                
                # Split into sentences
                sentences = nltk.sent_tokenize(text)
                
                for sentence in sentences:
                    if sentences_processed >= num_samples:
                        break
                        
                    # Get word boundaries for this sentence
                    word_boundaries = get_word_boundaries(sentence)
                    
                    # Find jargon terms in this sentence
                    terms = []
                    for jargon_term in jargon_dict.keys():
                        # Find all occurrences of the term in the sentence
                        start = 0
                        while True:
                            start_idx = sentence.lower().find(jargon_term.lower(), start)
                            if start_idx == -1:
                                break
                                
                            end_idx = start_idx + len(jargon_term)
                            
                            # Check if we're at word boundaries
                            if start_idx in word_boundaries and end_idx in word_boundaries:
                                terms.append((start_idx, end_idx))
                            
                            start = start_idx + 1
                    
                    if terms:  # Only add sentences that contain jargon terms
                        all_examples[dataset_name].append((sentence, sorted(terms)))
                        sentences_processed += 1
        else:  # MedReadme
            # Load raw data to get original format
            with open('data/medreadme/jargon.json', 'r') as f:
                all_data = json.load(f)
            
            # Get test examples
            test_examples = [item for item in all_data if item.get('split') == 'test'][:num_samples]
            
            for example in test_examples:
                # Get tokens and reconstruct text
                tokens = example['tokens']
                text = ' '.join(tokens)
                
                # Get spans from entities
                terms = []
                for start, end, _, _ in example['entities']:
                    # Calculate character offsets
                    char_start = len(' '.join(tokens[:start]))
                    if start > 0:
                        char_start += 1  # Add space
                    char_end = len(' '.join(tokens[:end]))
                    terms.append((char_start, char_end))
                
                all_examples[dataset_name].append((text, sorted(terms)))
    
    # Process both datasets
    print(f"\nCollecting first {num_samples} examples from each dataset...")
    process_dataset_examples(plaba_dataset, "PLABA")
    print("PLABA done")
    process_dataset_examples(medreadme_dataset, "MedReadme")
    print("MedReadme done")
    
    # Create single visualization file
    output_file = 'output/visualization/dataset_examples.html'
    create_input_visualization(all_examples, output_file)
    print(f"\nVisualization saved to: {output_file}")

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
    parser.add_argument("--show_inputs", action='store_true',
                        help="Show random input examples from both datasets")
    args = parser.parse_args()

    # Create output directory
    os.makedirs('output/visualization', exist_ok=True)

    if args.show_inputs:
        # Initialize tokenizer
        tokenizer = get_tokenizer(MODELS[args.model_name], cache_dir="./cache_models")
        visualize_dataset_examples(tokenizer, args.num_samples)
        return

    # Load model and continue with model predictions visualization
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