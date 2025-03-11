import torch
from transformers import AutoTokenizer
from typing import List
import matplotlib.pyplot as plt
import os
import numpy as np
from plaba import PLABADataset
from medreadme import MedReadmeDataset

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
    examples = dataset.train_dataset if hasattr(dataset, 'train_dataset') else dataset.get_split('train')
    
    for example in examples:
        labels = example['labels']
        attention_mask = example['attention_mask']
        
        # Get document length (excluding padding)
        doc_length = attention_mask.sum().item()
        doc_lengths.append(doc_length)
        total_tokens += doc_length
        
        # Count jargon terms (consecutive non-zero labels)
        current_span = 0
        for label in labels:
            if label != 0 and label != -100:  # Count any non-O and non-padding label
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

def main():
    # Initialize tokenizer (works for both BERT and RoBERTa)
    tokenizer = AutoTokenizer.from_pretrained('roberta-large', add_prefix_space=True)
    
    print("Loading datasets...")
    
    # PLABA datasets
    plaba_dataset = PLABADataset(tokenizer, 'data/PLABA_2024-Task_1')
    
    # MedREADME datasets (binary classification)
    medreadme_dataset = MedReadmeDataset(tokenizer, 'data/medreadme/jargon.json', classification_type='binary')
    
    # Gather and print statistics
    datasets = {
        "PLABA Training": plaba_dataset,
        "MedREADME Binary": medreadme_dataset,
    }
    
    stats = {}
    for name, dataset in datasets.items():
        stats[name] = gather_detailed_statistics(dataset, name)

    # Print comparative statistics
    print("\nComparative Statistics:")
    print("\nAverage Jargons per Document:")
    for name, stat in stats.items():
        print(f"{name}: {stat['avg_jargons']:.2f}")
    
    print("\nAverage Document Length:")
    for name, stat in stats.items():
        print(f"{name}: {stat['mean_doc_length']:.2f}")

if __name__ == "__main__":
    main() 