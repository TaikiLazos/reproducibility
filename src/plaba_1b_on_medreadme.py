import argparse
import torch
import os
import json
from transformers import AutoModelForTokenClassification, AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from medreadme import MedReadmeDataset, get_tokenizer
import numpy as np

def load_plaba_model(model_path, device):
    """Load the trained PLABA 1b model"""
    # Initialize base model
    model = AutoModelForTokenClassification.from_pretrained(
        'roberta-large',  # Using RoBERTa as base
        num_labels=5,  # 5 action types
        cache_dir="./cache_models"
    )
    
    # Modify the model's forward pass to use sigmoid activation
    def forward_with_sigmoid(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
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
    
    # Load the trained weights
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model

def predict_actions(model, test_loader, device, tokenizer):
    """Make predictions on test data"""
    predictions = []
    original_texts = []
    jargon_terms = []  # Store the actual jargon terms
    term_indices = []  # Store the token indices for each term
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Making predictions"):
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device)
            )
            # Get predictions for all tokens
            preds = (outputs > 0.5).float()
            preds = preds.cpu().numpy()
            
            # Get actual length of each sequence (ignore padding)
            mask = batch['attention_mask'].bool()
            
            # Process each sequence
            for pred_seq, seq_mask, input_ids in zip(preds, mask, batch['input_ids']):
                # Get the length of the actual sequence (excluding padding)
                length = seq_mask.sum().item()
                
                # Get predictions for actual tokens (excluding padding)
                pred_seq = pred_seq[:length]
                
                # Get the original text
                tokens = input_ids[:length]
                text = tokenizer.decode(tokens, skip_special_tokens=True)
                
                # Find jargon terms (tokens with any predicted action)
                jargon_indices = np.where(np.any(pred_seq, axis=1))[0]
                if len(jargon_indices) > 0:
                    # Group consecutive indices to get complete terms
                    current_term = []
                    terms = []
                    for i in range(len(jargon_indices)):
                        if i == 0 or jargon_indices[i] == jargon_indices[i-1] + 1:
                            current_term.append(jargon_indices[i])
                        else:
                            if current_term:
                                terms.append(current_term)
                            current_term = [jargon_indices[i]]
                    if current_term:
                        terms.append(current_term)
                    
                    # Convert token indices to text
                    jargon_texts = []
                    for term_indices in terms:
                        term_tokens = tokens[term_indices[0]:term_indices[-1]+1]
                        term_text = tokenizer.decode(term_tokens, skip_special_tokens=True)
                        jargon_texts.append(term_text)
                else:
                    jargon_texts = []
                    terms = []
                
                # Store predictions and texts
                predictions.append(pred_seq)
                original_texts.append(text)
                jargon_terms.append(jargon_texts)
                term_indices.append(terms)
    
    return predictions, original_texts, jargon_terms, term_indices

def save_predictions(predictions, original_texts, jargon_terms, term_indices, output_file):
    """Save predictions in a formatted way"""
    action_labels = ['SUBSTITUTE', 'EXPLAIN', 'GENERALIZE', 'OMIT', 'EXEMPLIFY']
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("PLABA 1b Model Predictions on MedReadme Test Cases\n")
        f.write("=" * 50 + "\n\n")
        
        # Show only first 20 examples
        for i, (text, preds, terms, indices) in enumerate(zip(original_texts, predictions, jargon_terms, term_indices)):
            if i >= 20:  # Limit to 20 examples
                break
                
            f.write(f"Example {i+1}:\n")
            f.write(f"Text: {text}\n")
            f.write("-" * 50 + "\n")
            
            if terms:
                f.write("Identified Jargon Terms:\n")
                for term in terms:
                    f.write(f"- {term}\n")
                    # Get actions predicted for this term
                    term_actions = []
                    for j, action in enumerate(action_labels):
                        # Check if any token in the sequence has this action predicted
                        if np.any(preds[:, j] == 1):
                            term_actions.append(action)
                    if term_actions:
                        f.write(f"  Predicted Actions: {', '.join(term_actions)}\n")
                    else:
                        f.write("  No specific actions predicted\n")
            else:
                f.write("No jargon terms identified\n")
            
            f.write("\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained PLABA 1b model")
    parser.add_argument("--medreadme_data", type=str, default="data/medreadme/jargon.json", help="Path to MedReadme data")
    parser.add_argument("--output_file", type=str, default="output/plaba/medreadme_predictions.txt", help="Path to save predictions")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the trained model
    print("Loading trained model...")
    model = load_plaba_model(args.model_path, device)
    
    # Load MedReadme test data
    print("Loading MedReadme test data...")
    tokenizer = get_tokenizer('roberta-large', "./cache_models")
    dataset = MedReadmeDataset(tokenizer, args.medreadme_data)
    test_loader = DataLoader(dataset.get_split('test'), batch_size=32)
    
    # Make predictions
    print("Making predictions...")
    predictions, original_texts, jargon_terms, term_indices = predict_actions(model, test_loader, device, tokenizer)
    
    # Save predictions
    print(f"Saving predictions to {args.output_file}...")
    save_predictions(predictions, original_texts, jargon_terms, term_indices, args.output_file)
    print("Done!")

if __name__ == "__main__":
    main()
