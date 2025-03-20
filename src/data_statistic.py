import torch
from transformers import AutoTokenizer
from typing import List
import matplotlib.pyplot as plt
import os
import numpy as np
from plaba import PLABADataset
from medreadme import MedReadmeDataset
import argparse

def plot_distribution(data: List[int], title: str, xlabel: str, save_path: str, is_jargon: bool = False):
    """Plot and save distribution histogram."""
    if not data:  # Skip empty data
        return
    
    plt.figure(figsize=(10, 6))
    
    if is_jargon:
        # Cap jargon lengths at 10
        capped_data = [min(x, 10) for x in data]
        # Create bins from 1 to 10
        bins = list(range(1, 12))  # 11 edges to create 10 bins
        # Count how many were capped
        num_capped = sum(1 for x in data if x > 10)
        
        plt.hist(capped_data, bins=bins, edgecolor='black')
        if num_capped > 0:
            plt.text(0.7, 0.95, f"*{num_capped} jargons longer than 10 words", 
                    transform=plt.gca().transAxes, 
                    fontsize=10, 
                    bbox=dict(facecolor='white', alpha=0.8))
    else:
        plt.hist(data, bins=30, edgecolor='black')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    
    if is_jargon:
        plt.xticks(range(1, 11))
        
    plt.savefig(save_path)
    plt.close()

def gather_detailed_statistics(dataset, dataset_name: str, split: str = 'train', output_dir: str = 'output/statistics'):
    """Gather and print detailed statistics about the dataset split."""
    # Get the correct split
    if hasattr(dataset, f'{split}_dataset'):
        examples = getattr(dataset, f'{split}_dataset')
    else:
        examples = dataset.get_split(split)
    
    if not examples:
        print(f"\n=== {dataset_name} {split.upper()} Dataset is empty ===")
        return None
        
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n=== {dataset_name} {split.upper()} Dataset Statistics ===")
    
    # Basic statistics
    num_examples = len(examples)
    total_jargons = 0
    jargon_lengths = []
    total_tokens = 0
    doc_lengths = []
    jargon_examples = []  # Store actual jargon texts
    
    for example in examples:
        labels = example['labels']
        attention_mask = example['attention_mask']
        input_ids = example['input_ids']
        
        # Get document length (excluding padding)
        doc_length = attention_mask.sum().item()
        doc_lengths.append(doc_length)
        total_tokens += doc_length
        
        # Count jargon terms (consecutive non-zero labels)
        current_span = []
        current_tokens = []
        for idx, (label, input_id) in enumerate(zip(labels, input_ids)):
            if label != 0 and label != -100:  # Count any non-O and non-padding label
                current_span.append(idx)
                current_tokens.append(input_id.item())
            elif current_span:
                # Convert tokens to text and count words
                jargon_text = dataset.tokenizer.decode(current_tokens)
                # Clean the text (remove special tokens and extra spaces)
                jargon_text = jargon_text.strip().replace('Ġ', ' ').strip()
                word_count = len(jargon_text.split())
                
                jargon_lengths.append(word_count)
                total_jargons += 1
                jargon_examples.append((jargon_text, word_count))
                current_span = []
                current_tokens = []
                
        if current_span:  # Don't forget last span
            jargon_text = dataset.tokenizer.decode(current_tokens)
            jargon_text = jargon_text.strip().replace('Ġ', ' ').strip()
            word_count = len(jargon_text.split())
            
            jargon_lengths.append(word_count)
            total_jargons += 1
            jargon_examples.append((jargon_text, word_count))
    
    # Sort jargons by word count for analysis
    jargon_examples.sort(key=lambda x: x[1], reverse=True)
    
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
        'mean_jargon_length': np.mean(jargon_lengths) if jargon_lengths else 0,
        'median_jargon_length': np.median(jargon_lengths) if jargon_lengths else 0,
        'longest_jargons': jargon_examples[:5]  # Store top 5 longest jargons
    }
    
    # Print statistics
    print("\nBasic Statistics:")
    print(f"Number of examples: {stats['num_examples']}")
    print(f"Total number of jargon terms: {stats['total_jargons']}")
    print(f"Average jargon terms per sentence: {stats['avg_jargons']:.2f}")
    print(f"Average tokens per sentence: {stats['avg_tokens']:.2f}")
    
    print("\nSentence Length Statistics:")
    print(f"Min: {stats['min_doc_length']}")
    print(f"Max: {stats['max_doc_length']}")
    print(f"Mean: {stats['mean_doc_length']:.2f}")
    
    if jargon_lengths:
        print("\nJargon Length Statistics:")
        print(f"Min: {stats['min_jargon_length']} words")
        print(f"Max: {stats['max_jargon_length']} words")
        print(f"Mean: {stats['mean_jargon_length']:.2f} words")
        print(f"Median: {stats['median_jargon_length']:.2f} words")
        
        print("\nTop 5 Longest Jargons (by word count):")
        for jargon, length in stats['longest_jargons']:
            print(f"{length} words: {jargon}")
    
    # Plot distributions
    plot_distribution(
        jargon_lengths,
        f"{dataset_name} {split.upper()} - Jargon Length Distribution",
        "Jargon Length (words)",
        os.path.join(output_dir, f"{dataset_name.lower().replace(' ', '_')}_{split}_jargon_lengths.png"),
        is_jargon=True
    )
    
    plot_distribution(
        doc_lengths,
        f"{dataset_name} {split.upper()} - Sentence Length Distribution",
        "Sentence Length (tokens)",
        os.path.join(output_dir, f"{dataset_name.lower().replace(' ', '_')}_{split}_doc_lengths.png"),
        is_jargon=False
    )
    
    print("=" * 60)
    return stats

def plot_sentence_length_comparison(datasets_data, output_dir: str):
    """Plot sentence length distributions for multiple datasets in one figure."""
    plt.figure(figsize=(12, 6))
    
    # Set global font sizes
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 20,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14
    })
    
    colors = ['#FF7F50', '#4169E1']  # Coral and Royal Blue
    alpha = 0.7
    bins = np.linspace(0, 200, 25)  # Changed max to 200
    
    # Plot histograms
    for (name, data), color in zip(datasets_data.items(), colors):
        # Cap the data at 200 for visualization
        capped_data = [min(x, 200) for x in data]
        plt.hist(capped_data, bins=bins, alpha=alpha, color=color, label=name, 
                edgecolor='black', linewidth=1.5)
    
    plt.title('Sentence Length Distribution Comparison\n(Training Split)', 
              fontsize=20, pad=15)
    plt.xlabel('Sentence Length (tokens)', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    
    # Make legend bigger and more prominent
    plt.legend(fontsize=18, loc='upper right',
              frameon=True,
              framealpha=1.0,
              edgecolor='black',
              fancybox=True,
              borderpad=1.0,
              labelspacing=0.8)
    
    # Style improvements
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=1.2)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(1.5)
    plt.gca().spines['bottom'].set_linewidth(1.5)
    
    # Set x-axis limit explicitly
    plt.xlim(0, 200)
    
    # Add mean lines with better positioned text
    y_max = plt.gca().get_ylim()[1]
    text_positions = [0.85, 0.7]  # Adjusted heights
    
    for (name, data), color, text_pos in zip(datasets_data.items(), colors, text_positions):
        mean_length = np.mean(data)
        plt.axvline(x=mean_length, color=color, linestyle='--', alpha=0.8, linewidth=2)
        # Make mean value text more prominent
        plt.text(mean_length+10, y_max * text_pos, 
                f'{name}\nMean: {mean_length:.1f}', 
                color=color, 
                alpha=1.0,
                fontsize=16,
                bbox=dict(facecolor='white', 
                         alpha=1.0,  # Fully opaque background
                         edgecolor=color,  # Add border in same color
                         pad=5,
                         boxstyle='round,pad=0.5'))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sentence_length_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs='+', default=['train'], 
                      choices=['train', 'val', 'dev', 'test'],
                      help="Dataset splits to analyze")
    args = parser.parse_args()

    # Initialize tokenizer (works for both BERT and RoBERTa)
    tokenizer = AutoTokenizer.from_pretrained('roberta-large', add_prefix_space=True)
    
    print("Loading datasets...")
    
    # PLABA datasets
    plaba_dataset = PLABADataset(tokenizer, 'data/PLABA_2024-Task_1')
    
    # MedREADME datasets (binary classification)
    medreadme_dataset = MedReadmeDataset(tokenizer, 'data/medreadme/jargon.json', classification_type='binary')
    
    # Gather and print statistics for each split
    datasets = {
        "PLABA": plaba_dataset,
        "MedREADME Binary": medreadme_dataset,
    }
    
    stats = {name: {} for name in datasets}
    for name, dataset in datasets.items():
        for split in args.splits:
            # Handle different split names between datasets
            if name == "PLABA" and split == "dev":
                continue  # PLABA uses 'val' instead of 'dev'
            if name == "MedREADME Binary" and split == "val":
                continue  # MedREADME uses 'dev' instead of 'val'
                
            stats[name][split] = gather_detailed_statistics(dataset, name, split)

    # Print comparative statistics
    print("\nComparative Statistics:")
    for split in args.splits:
        print(f"\n{split.upper()} Split Statistics:")
        print("\nAverage Jargons per Sentence:")
        for name in datasets:
            if split in stats[name]:
                print(f"{name}: {stats[name][split]['avg_jargons']:.2f}")
        
        print("\nAverage Sentence Length:")
        for name in datasets:
            if split in stats[name]:
                print(f"{name}: {stats[name][split]['mean_doc_length']:.2f}")

    # Collect sentence lengths for training split
    datasets_lengths = {}
    for name, dataset in datasets.items():
        if name == "PLABA":
            examples = dataset.train_dataset
        else:  # MedReadMe
            examples = dataset.get_split('train')
            
        doc_lengths = [example['attention_mask'].sum().item() 
                      for example in examples]
        datasets_lengths[name] = doc_lengths
    
    # Create comparison plot
    plot_sentence_length_comparison(datasets_lengths, 'output/statistics')

if __name__ == "__main__":
    main() 