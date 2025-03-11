import torch
from transformers import RobertaTokenizerFast
from torch.utils.data import Dataset
import json
from collections import Counter, defaultdict
import numpy as np
from typing import List, Dict
import matplotlib.pyplot as plt
import os
from finetune_and_evaluate import JargonDataset

def plot_distribution(data: List[int], title: str, xlabel: str, save_path: str):
    """Plot and save distribution histogram."""
    if not data:  # Skip empty data
        return
        
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.savefig(save_path)
    plt.close()

def gather_detailed_statistics(dataset, dataset_name: str, output_dir: str = 'output/statistics'):
    """Gather and print detailed statistics about the dataset."""
    if len(dataset) == 0:
        print(f"\n=== {dataset_name} Dataset is empty ===")
        return None
        
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n=== {dataset_name} Dataset Statistics ===")
    
    # Basic statistics
    num_examples = len(dataset)
    total_jargons = 0
    jargon_lengths = []
    total_tokens = 0
    doc_lengths = []
    
    # Gather statistics
    for example in dataset.examples:
        labels = example['labels']
        attention_mask = example['attention_mask']
        
        # Get document length (excluding padding)
        doc_length = attention_mask.sum().item()
        doc_lengths.append(doc_length)
        total_tokens += doc_length
        
        # Count jargon terms (consecutive 1s)
        current_span = 0
        for label in labels:
            if label == 1:
                current_span += 1
            elif current_span > 0:
                jargon_lengths.append(current_span)
                total_jargons += 1
                current_span = 0
        if current_span > 0:  # Don't forget last span
            jargon_lengths.append(current_span)
            total_jargons += 1
    
    # Calculate statistics
    stats = {
        'num_examples': num_examples,
        'total_jargons': total_jargons,
        'avg_jargons': total_jargons / num_examples if num_examples > 0 else 0,
        'total_tokens': total_tokens,
        'avg_tokens': total_tokens / num_examples if num_examples > 0 else 0,
        'min_doc_length': min(doc_lengths) if doc_lengths else 0,
        'max_doc_length': max(doc_lengths) if doc_lengths else 0,
        'mean_doc_length': np.mean(doc_lengths) if doc_lengths else 0,
        'min_jargon_length': min(jargon_lengths) if jargon_lengths else 0,
        'max_jargon_length': max(jargon_lengths) if jargon_lengths else 0,
        'mean_jargon_length': np.mean(jargon_lengths) if jargon_lengths else 0
    }
    
    # Print statistics
    print("\nBasic Statistics:")
    print(f"Number of examples: {stats['num_examples']}")
    print(f"Total number of jargon terms: {stats['total_jargons']}")
    print(f"Average jargon terms per document: {stats['avg_jargons']:.2f}")
    print(f"Average tokens per document: {stats['avg_tokens']:.2f}")
    
    print("\nDocument Length Statistics:")
    print(f"Min: {stats['min_doc_length']}")
    print(f"Max: {stats['max_doc_length']}")
    print(f"Mean: {stats['mean_doc_length']:.2f}")
    
    if jargon_lengths:
        print("\nJargon Length Statistics:")
        print(f"Min: {stats['min_jargon_length']}")
        print(f"Max: {stats['max_jargon_length']}")
        print(f"Mean: {stats['mean_jargon_length']:.2f}")
    
    # Plot distributions
    plot_distribution(
        jargon_lengths,
        f"{dataset_name} - Jargon Length Distribution",
        "Jargon Length (tokens)",
        os.path.join(output_dir, f"{dataset_name.lower().replace(' ', '_')}_jargon_lengths.png")
    )
    
    plot_distribution(
        doc_lengths,
        f"{dataset_name} - Document Length Distribution",
        "Document Length (tokens)",
        os.path.join(output_dir, f"{dataset_name.lower().replace(' ', '_')}_doc_lengths.png")
    )
    
    print("=" * 60)
    return stats

class MedREADMEDataset(Dataset):
    """Dataset class that handles MedREADME data with splits."""
    def __init__(self, data_path: str, tokenizer, split: str = None):
        self.tokenizer = tokenizer
        self.split = split
        self.examples = self._prepare_data(data_path)
    
    def _prepare_data(self, data_path):
        examples = []
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # Filter by split if specified
        if self.split:
            data = [item for item in data if item.get('split') == self.split]
        
        for item in data:
            tokens = item['tokens']
            entities = item['entities']
            
            # Create BIO labels
            labels = ['O'] * len(tokens)
            for start, end, _, _ in entities:
                labels[start] = 'B-JARGON'
                for i in range(start + 1, end):
                    labels[i] = 'I-JARGON'
            
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
            
            # Convert BIO to binary labels
            word_ids = tokenized.word_ids()
            label_ids = []
            
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                else:
                    label = labels[word_idx]
                    label_ids.append(1 if label.startswith(('B-', 'I-')) else 0)
            
            examples.append({
                'input_ids': tokenized['input_ids'][0],
                'attention_mask': tokenized['attention_mask'][0],
                'labels': torch.tensor(label_ids, dtype=torch.long)
            })
        
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def main():
    # Initialize tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large', add_prefix_space=True)
    
    # Load datasets
    print("Loading datasets...")
    
    # PLABA datasets
    plaba_train = JargonDataset('data/PLABA_2024-Task_1/train.json', tokenizer, is_plaba=True)
    plaba_test = JargonDataset('data/PLABA_2024-Task_1/task_1_testing.json', tokenizer, is_plaba=True)
    
    # Load MedREADME datasets with splits
    medreadme_train = MedREADMEDataset('data/medreadme/jargon.json', tokenizer, split='train')
    medreadme_val = MedREADMEDataset('data/medreadme/jargon.json', tokenizer, split='dev')
    medreadme_test = MedREADMEDataset('data/medreadme/jargon.json', tokenizer, split='test')
    
    # Gather and print statistics
    datasets = {
        "PLABA Training": plaba_train,
        "PLABA Testing": plaba_test,
        "MedREADME Training": medreadme_train,
        "MedREADME Validation": medreadme_val,
        "MedREADME Testing": medreadme_test
    }
    
    stats = {}
    for name, dataset in datasets.items():
        stats[name] = gather_detailed_statistics(dataset, name)

if __name__ == "__main__":
    main() 