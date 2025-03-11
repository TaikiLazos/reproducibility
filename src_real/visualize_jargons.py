# Maybe make a PNG after creating the html?
# apply a parser argument

import json
import html
import argparse
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
from plaba import get_tokenizer
import os

models = {
    'bert': 'bert-large-cased',
    'roberta': 'roberta-large'
}

class TransferJargonDetector:
    def __init__(self, model_path, model_name='roberta'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize base model and tokenizer
        self.tokenizer = get_tokenizer(models[model_name], cache_dir="./cache_models")
        base_model = AutoModelForTokenClassification.from_pretrained(
            models[model_name],
            num_labels=2,
            cache_dir="./cache_models"
        )
        
        # Load the trained model
        self.model = AutoModelForTokenClassification.from_config(base_model.config)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()

    def detect_jargons(self, text):
        # Split into tokens (same as training)
        tokens = text.split()
        
        # Tokenize using the same parameters as training
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**encoding)
            predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().tolist()

        # Use the same word_ids alignment as in training
        word_ids = encoding.word_ids()
        
        # Convert predictions to jargon spans
        jargons = []
        current_jargon = []
        current_start = None
        prev_word_id = None
        
        for idx, (pred, word_id) in enumerate(zip(predictions, word_ids)):
            if word_id is not None:  # Skip special tokens
                if word_id != prev_word_id:  # First subword of a word
                    if current_jargon and pred == 0:  # End of jargon
                        jargon_text = ' '.join(current_jargon)
                        start_pos = text.find(jargon_text)
                        if start_pos != -1:
                            jargons.append((jargon_text, start_pos, start_pos + len(jargon_text)))
                        current_jargon = []
                    
                    if pred == 1:  # Start or continue jargon
                        if not current_jargon:  # Start new jargon
                            current_start = word_id
                        current_jargon.append(tokens[word_id])
                
                prev_word_id = word_id

        # Don't forget the last jargon if it exists
        if current_jargon:
            jargon_text = ' '.join(current_jargon)
            start_pos = text.find(jargon_text)
            if start_pos != -1:
                jargons.append((jargon_text, start_pos, start_pos + len(jargon_text)))

        return jargons

def load_test_data(abstract_path, annotations_path):
    # Load the first abstract
    with open(abstract_path, 'r') as f:
        text = f.read()
    
    # Load annotations
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    # Get first abstract's jargons (Q19_A1)
    first_id = "Q19_A1"  # Explicitly use Q19_A1
    true_jargons = list(annotations[first_id].keys())
    
    return text, true_jargons

def create_html_visualization(text, true_jargons, predicted_jargons, model_name, output_file):
    # Sort jargons by length (longest first) to handle nested terms correctly
    true_jargons = sorted(true_jargons, key=len, reverse=True)
    predicted_jargons = sorted(predicted_jargons, key=lambda x: len(x[0]), reverse=True)
    
    # Create a list of characters with their tags
    chars = list(text)
    true_tags = [False] * len(chars)
    pred_tags = [False] * len(chars)
    
    # Mark true jargons
    for jargon in true_jargons:
        start = 0
        while True:
            pos = text.find(jargon, start)
            if pos == -1:
                break
            for i in range(pos, pos + len(jargon)):
                true_tags[i] = True
            start = pos + 1
    
    # Mark predicted jargons
    for jargon, start, end in predicted_jargons:
        for i in range(start, end):
            if i < len(pred_tags):
                pred_tags[i] = True
    
    # Build HTML with proper tags
    html_parts = []
    current_true = False
    current_pred = False
    
    for i, char in enumerate(chars):
        # Handle tag changes
        if true_tags[i] != current_true or pred_tags[i] != current_pred:
            # Close previous tags
            if current_true or current_pred:
                html_parts.append('</span>')
            # Open new tags
            if true_tags[i] or pred_tags[i]:
                classes = []
                if true_tags[i] and pred_tags[i]:
                    classes.append('overlap-jargon')
                elif true_tags[i]:
                    classes.append('true-jargon')
                elif pred_tags[i]:
                    classes.append('predicted-jargon')
                html_parts.append(f'<span class="{" ".join(classes)}">')
            current_true = true_tags[i]
            current_pred = pred_tags[i]
        
        # Add the character
        html_parts.append(html.escape(char))
    
    # Close any open tags
    if current_true or current_pred:
        html_parts.append('</span>')
    
    text_html = ''.join(html_parts)
    
    # Create HTML document with updated styling
    html_content = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: 'Segoe UI', Arial, sans-serif;
                line-height: 1.8;
                max-width: 800px;
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
            .true-jargon {{
                background-color: rgba(0, 0, 255, 0.1);  /* Blue */
                border-bottom: 2px solid #0000FF;
                padding: 2px 4px;
                border-radius: 3px;
            }}
            .predicted-jargon {{
                background-color: rgba(255, 0, 0, 0.1);  /* Red */
                border-bottom: 2px solid #FF0000;
                padding: 2px 4px;
                border-radius: 3px;
            }}
            .overlap-jargon {{
                background-color: rgba(0, 255, 0, 0.1);  /* Green */
                border-bottom: 2px solid #00FF00;
                padding: 2px 4px;
                border-radius: 3px;
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
                margin-right: 20px;
                padding: 5px 10px;
                border-radius: 4px;
            }}
            p {{
                margin-bottom: 15px;
                text-align: justify;
            }}
        </style>
    </head>
    <body>
        <h2>Model: {model_name}</h2>
        <div class="legend">
            <span class="true-jargon">True Jargons (Not Predicted)</span>
            <span class="predicted-jargon">False Predictions</span>
            <span class="overlap-jargon">Correct Predictions</span>
        </div>
        <div class="content">
            <p>{text_html}</p>
        </div>
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    # Print statistics
    print(f"\nResults for {model_name}:")
    print("\nTrue jargons:")
    for jargon in true_jargons:
        print(f"- {jargon}")
    
    print("\nPredicted jargons:")
    for jargon, start, end in predicted_jargons:
        print(f"- '{jargon}' (position {start}-{end})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="roberta", choices=['bert', 'roberta'])
    args = parser.parse_args()

    # Create output directory for visualizations
    os.makedirs('output/visualization', exist_ok=True)

    # Load test data
    text, true_jargons = load_test_data(
        "data/PLABA_2024-Task_1/abstracts/Q19_A1.src.txt",
        "data/PLABA_2024-Task_1/task_1_testing.json"
    )
    
    # Define models to visualize
    transfer_models = [
        ('medreadme_to_plaba', 'output/transfer/medreadme_to_plaba.pt'),
        ('plaba_to_medreadme', 'output/transfer/plaba_to_medreadme.pt'),
        ('plaba_plus_medreadme_to_plaba', 'output/transfer/plaba_plus_medreadme_to_plaba.pt'),
        ('medreadme_plus_plaba_to_plaba', 'output/transfer/medreadme_plus_plaba_to_plaba.pt')
    ]
    
    # Create visualization for each model
    for model_name, model_path in transfer_models:
        detector = TransferJargonDetector(model_path, args.model_name)
        predicted_jargons = detector.detect_jargons(text)
        output_file = f'output/visualization/Q19_A1_{model_name}.html'
        create_html_visualization(text, true_jargons, predicted_jargons, model_name, output_file)
        print(f"Visualization saved to {output_file}")

if __name__ == "__main__":
    main() 