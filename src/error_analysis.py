import argparse
import torch
from torch.utils.data import DataLoader
from collections import defaultdict, Counter
from medreadme import MedReadmeDataset, get_tokenizer, models
from transformers import AutoModelForTokenClassification
import numpy as np
from tabulate import tabulate
import json
import os

class ErrorAnalyzer:
    def __init__(self, model_path, model_name, classification_type):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classification_type = classification_type
        
        # Initialize tokenizer
        self.tokenizer = get_tokenizer(models[model_name][1], cache_dir="./cache_models")  # Use large model
        
        # Get number of labels
        num_labels = {'binary': 2, '3-cls': 4, '7-cls': 8}[classification_type]
        
        # Load model
        self.model = AutoModelForTokenClassification.from_pretrained(
            models[model_name][1],
            num_labels=num_labels,
            cache_dir="./cache_models"
        )
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()
        
        self.class_names = self.get_class_names()

    def get_class_names(self):
        if self.classification_type == 'binary':
            return ['O', 'COMPLEX']
        elif self.classification_type == '3-cls':
            return ['O', 'MEDICAL', 'ABBR', 'GENERAL']
        else:  # 7-cls
            return ['O', 'GOOGLE_EASY', 'GOOGLE_HARD', 'MEDICAL_NAME', 
                    'MEDICAL_ABBR', 'GENERAL_ABBR', 'GENERAL_COMPLEX', 'MULTISENSE']

    def analyze_errors(self, test_loader):
        error_stats = {
            'confusion_matrix': defaultdict(lambda: defaultdict(int)),
            'error_examples': defaultdict(list),
            'token_errors': defaultdict(lambda: {'count': 0, 'examples': []}),
            'context_errors': [],
            'length_based_errors': defaultdict(list),
            'class_performance': defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0}),
            'boundary_errors': {'start': 0, 'middle': 0, 'end': 0}
        }

        for batch_idx, batch in enumerate(test_loader):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels']

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=2)

            for seq_idx, (pred_seq, label_seq, mask) in enumerate(zip(predictions, labels, attention_mask)):
                length = mask.sum().item()
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids[seq_idx][:length])
                
                # Analyze each token
                current_entity = {'pred': None, 'true': None, 'tokens': []}
                
                for i, (token, pred, label) in enumerate(zip(tokens, pred_seq[:length], label_seq[:length])):
                    if label.item() == -100:
                        continue
                        
                    pred = pred.item()
                    label = label.item()
                    
                    # Update confusion matrix
                    error_stats['confusion_matrix'][self.class_names[label]][self.class_names[pred]] += 1
                    
                    # Track errors
                    if pred != label:
                        # Store error example
                        context = ' '.join(tokens[max(0, i-5):min(len(tokens), i+6)])
                        error_stats['error_examples'][f"{self.class_names[label]}->{self.class_names[pred]}"].append({
                            'token': token,
                            'context': context,
                            'batch_idx': batch_idx,
                            'seq_idx': seq_idx
                        })
                        
                        # Track problematic tokens
                        error_stats['token_errors'][token]['count'] += 1
                        error_stats['token_errors'][token]['examples'].append({
                            'pred': self.class_names[pred],
                            'true': self.class_names[label],
                            'context': context
                        })
                        
                        # Track entity boundary errors
                        if label != 0:  # Only for actual entities
                            if i == 0 or label_seq[i-1].item() == 0:
                                error_stats['boundary_errors']['start'] += 1
                            elif i == length-1 or label_seq[i+1].item() == 0:
                                error_stats['boundary_errors']['end'] += 1
                            else:
                                error_stats['boundary_errors']['middle'] += 1
                    
                    # Update class performance
                    if label != 0:  # True entity
                        if pred == label:
                            error_stats['class_performance'][self.class_names[label]]['tp'] += 1
                        else:
                            error_stats['class_performance'][self.class_names[label]]['fn'] += 1
                            if pred != 0:
                                error_stats['class_performance'][self.class_names[pred]]['fp'] += 1
                    elif pred != 0:  # False positive
                        error_stats['class_performance'][self.class_names[pred]]['fp'] += 1

        return error_stats

    def clean_token(self, token):
        """Clean token by replacing special characters with readable ones"""
        return token.replace('\u0120', ' ').strip()

    def clean_context(self, context):
        """Clean context string by replacing special characters and normalizing spaces"""
        # Replace the unicode space character and normalize multiple spaces
        cleaned = context.replace('\u0120', ' ')
        cleaned = ' '.join(cleaned.split())
        return cleaned

    def prepare_for_json(self, error_stats):
        """Prepare error statistics for JSON output by cleaning special characters"""
        clean_stats = {}
        
        # Clean error examples
        clean_stats['error_examples'] = {}
        for error_type, examples in error_stats['error_examples'].items():
            clean_stats['error_examples'][error_type] = [
                {
                    'token': self.clean_token(ex['token']),
                    'context': self.clean_context(ex['context']),
                    'batch_idx': ex['batch_idx'],
                    'seq_idx': ex['seq_idx']
                } for ex in examples
            ]
        
        # Clean token errors
        clean_stats['token_errors'] = {}
        for token, data in error_stats['token_errors'].items():
            clean_token = self.clean_token(token)
            clean_stats['token_errors'][clean_token] = {
                'count': data['count'],
                'examples': [
                    {
                        'pred': ex['pred'],
                        'true': ex['true'],
                        'context': self.clean_context(ex['context'])
                    } for ex in data['examples']
                ]
            }
        
        # Copy other statistics as is
        clean_stats['boundary_errors'] = error_stats['boundary_errors']
        clean_stats['confusion_matrix'] = {
            k: dict(v) for k, v in error_stats['confusion_matrix'].items()
        }
        clean_stats['class_performance'] = error_stats['class_performance']
        
        return clean_stats

    def print_analysis(self, error_stats):
        print("\n=== ERROR ANALYSIS REPORT ===\n")
        
        # 1. Overall Class Performance
        print("\nClass Performance:")
        headers = ['Class', 'Precision', 'Recall', 'F1', 'Support']
        rows = []
        for cls_name, stats in error_stats['class_performance'].items():
            if cls_name == 'O':
                continue
            tp = stats['tp']
            fp = stats['fp']
            fn = stats['fn']
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            support = tp + fn
            rows.append([cls_name, f"{precision:.3f}", f"{recall:.3f}", f"{f1:.3f}", support])
        print(tabulate(rows, headers=headers, tablefmt='grid'))

        # 2. Confusion Matrix
        print("\nConfusion Matrix:")
        headers = ['True\\Pred'] + list(error_stats['confusion_matrix'].keys())
        rows = []
        for true_cls in error_stats['confusion_matrix'].keys():
            row = [true_cls]
            for pred_cls in error_stats['confusion_matrix'].keys():
                row.append(error_stats['confusion_matrix'][true_cls][pred_cls])
            rows.append(row)
        print(tabulate(rows, headers=headers, tablefmt='grid'))

        # 3. Most Common Error Types
        print("\nMost Common Error Types:")
        error_types = []
        for error_type, examples in error_stats['error_examples'].items():
            error_types.append((error_type, len(examples)))
        error_types.sort(key=lambda x: x[1], reverse=True)
        for error_type, count in error_types[:5]:
            print(f"{error_type}: {count} occurrences")
            print("Example contexts:")
            for example in error_stats['error_examples'][error_type][:3]:
                clean_context = ' '.join(self.clean_token(t) for t in example['context'].split())
                print(f"  Token: '{self.clean_token(example['token'])}' in context: '{clean_context}'")
            print()

        # 4. Most Problematic Tokens
        print("\nMost Problematic Tokens:")
        problem_tokens = [(self.clean_token(token), data['count']) 
                         for token, data in error_stats['token_errors'].items()]
        problem_tokens.sort(key=lambda x: x[1], reverse=True)
        for token, count in problem_tokens[:10]:
            print(f"'{token}': {count} errors")
            examples = error_stats['token_errors'][token]['examples'][:2]
            for ex in examples:
                print(f"  Predicted as {ex['pred']} instead of {ex['true']}")
                clean_context = ' '.join(self.clean_token(t) for t in ex['context'].split())
                print(f"  Context: '{clean_context}'")
            print()

        # 5. Entity Boundary Errors
        print("\nEntity Boundary Errors:")
        total_boundary_errors = sum(error_stats['boundary_errors'].values())
        if total_boundary_errors > 0:  # Avoid division by zero
            for pos, count in error_stats['boundary_errors'].items():
                print(f"{pos}: {count} errors ({count/total_boundary_errors:.1%})")

        # Save detailed examples to file with cleaned special characters
        output_dir = "output/error_analysis"
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/detailed_errors_{self.classification_type}.json"
        
        # Clean the stats before saving
        clean_stats = self.prepare_for_json(error_stats)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(clean_stats, f, indent=2, ensure_ascii=False)
        
        print(f"\nDetailed error examples saved to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, 
                        choices=['bert', 'roberta', 'biobert', 'pubmedbert'])
    parser.add_argument("--classification", type=str, required=True,
                        choices=['binary', '3-cls', '7-cls'])
    parser.add_argument("--data_dir", type=str, default="data/medreadme/jargon.json",
                        help="Path to data file")
    args = parser.parse_args()

    # Load model and create analyzer
    model_path = f'output/medreadme/{args.model_name}_large_{args.classification}.pt'
    analyzer = ErrorAnalyzer(model_path, args.model_name, args.classification)

    # Create dataset and dataloader
    dataset = MedReadmeDataset(
        analyzer.tokenizer,
        args.data_dir,
        classification_type=args.classification
    )
    test_loader = DataLoader(dataset.get_split('test'), batch_size=1)

    # Perform analysis
    print(f"Analyzing errors for {args.model_name}-large ({args.classification})...")
    error_stats = analyzer.analyze_errors(test_loader)
    analyzer.print_analysis(error_stats)

if __name__ == "__main__":
    main() 