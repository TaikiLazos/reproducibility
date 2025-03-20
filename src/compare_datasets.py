import torch
from transformers import AutoTokenizer
from typing import List, Dict, Set, Tuple
import matplotlib.pyplot as plt
import os
import numpy as np
from plaba import PLABADataset
from medreadme import MedReadmeDataset
import argparse
import json
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
import textstat
from tqdm import tqdm

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

def get_combined_examples(dataset, splits=['train', 'test']):
    """Combine examples from multiple splits."""
    combined_examples = []
    
    for split in splits:
        if hasattr(dataset, f'{split}_dataset'):
            examples = getattr(dataset, f'{split}_dataset')
        else:
            examples = dataset.get_split(split)
        combined_examples.extend(examples)
    
    return combined_examples

def calculate_smog(text: str) -> float:
    """
    Calculate SMOG grade level.
    SMOG = 1.0430 × sqrt(number of polysyllables × (30 / number of sentences)) + 3.1291
    """
    # Count sentences
    sentences = nltk.sent_tokenize(text)
    num_sentences = len(sentences)
    
    # Need at least 30 sentences for accurate SMOG calculation
    if num_sentences < 30:
        return 0.0
    
    # Count polysyllabic words (words with 3 or more syllables)
    words = nltk.word_tokenize(text.lower())
    polysyllable_count = sum(1 for word in words if textstat.syllable_count(word) >= 3)
    
    # Calculate SMOG
    if num_sentences > 0:
        smog = 1.0430 * np.sqrt(polysyllable_count * (30.0 / num_sentences)) + 3.1291
        return smog
    return 0.0

def calculate_readability_metrics(text: str) -> Dict[str, float]:
    """Calculate readability metrics for a text."""
    return {
        'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
        'automated_readability_index': textstat.automated_readability_index(text),
        'smog_index': calculate_smog(text)
    }

def analyze_dataset_statistics(dataset, examples) -> Dict[str, float]:
    """Analyze readability metrics and jargon statistics for a dataset."""
    all_metrics = []
    total_jargons = 0
    total_jargon_lengths = []
    total_sentences = 0
    
    print(f"Analyzing {len(examples)} examples...")
    for example in tqdm(examples):
        # Get actual text and labels
        mask = example['attention_mask'].bool()
        text = dataset.tokenizer.decode(
            example['input_ids'][mask]
        )
        text = text.strip().replace('Ġ', ' ').strip()
        
        # Skip empty texts
        if not text:
            continue
            
        # Get readability metrics
        metrics = calculate_readability_metrics(text)
        all_metrics.append(metrics)
        
        # Count jargons and their lengths
        labels = example['labels'][mask]
        current_span = []
        for label in labels:
            if label != 0 and label != -100:
                current_span.append(label)
            elif current_span:
                total_jargons += 1
                total_jargon_lengths.append(len(current_span))
                current_span = []
        
        if current_span:  # Don't forget last span
            total_jargons += 1
            total_jargon_lengths.append(len(current_span))
        
        # Count sentences
        total_sentences += len(nltk.sent_tokenize(text))
    
    # Average the metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    # Add jargon statistics
    avg_metrics['jargons_per_sentence'] = total_jargons / total_sentences if total_sentences > 0 else 0
    avg_metrics['mean_jargon_length'] = np.mean(total_jargon_lengths) if total_jargon_lengths else 0
    
    return avg_metrics

def create_readability_table(plaba_metrics: Dict[str, float], 
                           medreadme_metrics: Dict[str, float]) -> str:
    """Create LaTeX table for readability metrics."""
    latex_table = (
        "\\begin{table}[H]\n"
        "\\centering\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{lcccccc}\n"
        "\\hline\n"
        "\\textbf{Dataset} & \\textbf{FKGL} & \\textbf{ARI} & \\textbf{SMOG} & "
        "\\textbf{J/S} & \\textbf{J-Len} \\\\\n"
        "\\hline\n"
        f"PLABA & {plaba_metrics['flesch_kincaid_grade']:.2f} & "
        f"{plaba_metrics['automated_readability_index']:.2f} & "
        f"{plaba_metrics['smog_index']:.2f} & "
        f"{plaba_metrics['jargons_per_sentence']:.2f} & "
        f"{plaba_metrics['mean_jargon_length']:.2f} \\\\\n"
        f"MedReadMe & {medreadme_metrics['flesch_kincaid_grade']:.2f} & "
        f"{medreadme_metrics['automated_readability_index']:.2f} & "
        f"{medreadme_metrics['smog_index']:.2f} & "
        f"{medreadme_metrics['jargons_per_sentence']:.2f} & "
        f"{medreadme_metrics['mean_jargon_length']:.2f} \\\\\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\caption{Dataset statistics comparison. FKGL: Flesch-Kincaid Grade Level, "
        "ARI: Automated Readability Index, SMOG: Simple Measure of Gobbledygook, "
        "J/S: Jargons per Sentence, J-Len: Mean Jargon Length (in tokens).}\n"
        "\\label{tab:dataset_statistics}\n"
        "\\end{table}\n"
    )
    return latex_table

def get_jargon_terms(dataset, split: str) -> Set[str]:
    """Extract unique jargon terms from a dataset split."""
    jargon_terms = set()
    
    if hasattr(dataset, f'{split}_dataset'):
        examples = getattr(dataset, f'{split}_dataset')
    else:
        examples = dataset.get_split(split)
    
    for example in examples:
        labels = example['labels']
        input_ids = example['input_ids']
        attention_mask = example['attention_mask']
        
        # Get actual tokens (excluding padding)
        tokens = input_ids[attention_mask.bool()]
        
        # Extract jargon terms (consecutive non-zero labels)
        current_span = []
        for idx, (label, token) in enumerate(zip(labels, tokens)):
            if label != 0 and label != -100:  # Non-O and non-padding label
                current_span.append(token)
            elif current_span:
                # Convert tokens to text
                jargon_text = dataset.tokenizer.decode(current_span)
                jargon_text = jargon_text.strip().replace('Ġ', ' ').strip()
                jargon_terms.add(jargon_text)
                current_span = []
                
        if current_span:  # Don't forget last span
            jargon_text = dataset.tokenizer.decode(current_span)
            jargon_text = jargon_text.strip().replace('Ġ', ' ').strip()
            jargon_terms.add(jargon_text)
    
    return jargon_terms

def analyze_text_characteristics(dataset, split: str) -> Dict:
    """Analyze text characteristics like length, complexity, etc."""
    if hasattr(dataset, f'{split}_dataset'):
        examples = getattr(dataset, f'{split}_dataset')
    else:
        examples = dataset.get_split(split)
    
    text_lengths = []
    sentence_lengths = []
    readability_scores = []
    word_counts = []
    syllable_counts = []
    difficult_word_counts = []
    
    for example in examples:
        input_ids = example['input_ids']
        attention_mask = example['attention_mask']
        
        # Get actual text
        text = dataset.tokenizer.decode(input_ids[attention_mask.bool()])
        text = text.strip().replace('Ġ', ' ').strip()
        
        # Calculate metrics
        text_lengths.append(len(text.split()))
        sentences = nltk.sent_tokenize(text)
        sentence_lengths.extend([len(s.split()) for s in sentences])
        
        # Calculate readability metrics
        readability = calculate_readability_metrics(text)
        readability_scores.append(readability)
        
        # Store additional metrics
        word_counts.append(readability['lexicon_count'])
        syllable_counts.append(readability['syllable_count'])
        difficult_word_counts.append(readability['difficult_words'])
    
    return {
        'text_lengths': text_lengths,
        'sentence_lengths': sentence_lengths,
        'readability_scores': readability_scores,
        'word_counts': word_counts,
        'syllable_counts': syllable_counts,
        'difficult_word_counts': difficult_word_counts
    }

def plot_comparison(data1: List[float], data2: List[float], 
                   title: str, xlabel: str, 
                   save_path: str, 
                   labels: Tuple[str, str] = ('PLABA', 'MedReadme')):
    """Plot comparison between two datasets."""
    plt.figure(figsize=(10, 6))
    
    plt.hist(data1, alpha=0.5, label=labels[0], bins=30)
    plt.hist(data2, alpha=0.5, label=labels[1], bins=30)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.savefig(save_path)
    plt.close()

def combine_readability_scores(results_train: Dict, results_test: Dict) -> Dict:
    """Combine readability scores from train and test splits."""
    combined = {'text_statistics': {'plaba': {'readability': {}}, 'medreadme': {'readability': {}}}}
    
    # Get all metrics from the first result
    metrics = results_train['text_statistics']['plaba']['readability'].keys()
    
    # Combine scores with weighted average based on dataset sizes
    for dataset in ['plaba', 'medreadme']:
        train_size = len(results_train['text_statistics'][dataset]['text_lengths'])
        test_size = len(results_test['text_statistics'][dataset]['text_lengths'])
        total_size = train_size + test_size
        
        for metric in metrics:
            train_score = results_train['text_statistics'][dataset]['readability'][metric]
            test_score = results_test['text_statistics'][dataset]['readability'][metric]
            
            # Weighted average
            combined_score = (train_score * train_size + test_score * test_size) / total_size
            combined['text_statistics'][dataset]['readability'][metric] = combined_score
    
    return combined

def print_readability_table(results: Dict, split: str = "combined"):
    """Print a formatted table of key readability metrics."""
    plaba_scores = results['text_statistics']['plaba']['readability']
    medreadme_scores = results['text_statistics']['medreadme']['readability']
    
    # Define the metrics we want to show
    metrics = {
        'flesch_kincaid_grade': 'FKGL',
        'automated_readability_index': 'ARI',
        'smog_index': 'SMOG'
    }
    
    # Also save as LaTeX table
    latex_table = (
        "\\begin{table}[H]\n"
        "\\centering\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{lccc}\n"
        "\\hline\n"
        "\\textbf{Dataset} & \\textbf{FKGL} & \\textbf{ARI} & \\textbf{SMOG} \\\\\n"
        "\\hline\n"
        f"PLABA & {plaba_scores['flesch_kincaid_grade']:.2f} & {plaba_scores['automated_readability_index']:.2f} & {plaba_scores['smog_index']:.2f} \\\\\n"
        f"MedReadMe & {medreadme_scores['flesch_kincaid_grade']:.2f} & {medreadme_scores['automated_readability_index']:.2f} & {medreadme_scores['smog_index']:.2f} \\\\\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\caption{Overall readability metrics across all splits. FKGL: Flesch-Kincaid Grade Level, ARI: Automated Readability Index, SMOG: Simple Measure of Gobbledygook.}\n"
        "\\label{tab:readability_metrics}\n"
        "\\end{table}\n"
    )
    
    # Save LaTeX table to file
    with open(f'output/dataset_comparison/readability_table_combined.tex', 'w') as f:
        f.write(latex_table)

def compare_datasets():
    """Main function to compare PLABA and MedReadme datasets."""
    # Initialize tokenizer
    print("Loading tokenizer and datasets...")
    tokenizer = AutoTokenizer.from_pretrained('roberta-large', add_prefix_space=True)
    
    # Load datasets
    plaba_dataset = PLABADataset(tokenizer, 'data/PLABA_2024-Task_1')
    medreadme_dataset = MedReadmeDataset(tokenizer, 'data/medreadme/jargon.json')
    
    # Get combined examples
    plaba_examples = get_combined_examples(plaba_dataset)
    medreadme_examples = get_combined_examples(medreadme_dataset)
    
    # Calculate metrics
    print("\nAnalyzing PLABA dataset...")
    plaba_metrics = analyze_dataset_statistics(plaba_dataset, plaba_examples)
    print("\nAnalyzing MedReadMe dataset...")
    medreadme_metrics = analyze_dataset_statistics(medreadme_dataset, medreadme_examples)
    
    # Create output directory
    os.makedirs('output/dataset_comparison', exist_ok=True)
    
    # Create and save table
    latex_table = create_readability_table(plaba_metrics, medreadme_metrics)
    with open('output/dataset_comparison/dataset_statistics.tex', 'w') as f:
        f.write(latex_table)
    
    # Print results to console
    print("\nDataset Statistics:")
    print("-" * 70)
    print(f"{'Metric':<15} {'PLABA':>12} {'MedReadMe':>12}")
    print("-" * 70)
    metrics = {
        'flesch_kincaid_grade': 'FKGL',
        'automated_readability_index': 'ARI',
        'smog_index': 'SMOG',
        'jargons_per_sentence': 'Jargons/Sent',
        'mean_jargon_length': 'Jargon Length'
    }
    for key, name in metrics.items():
        print(f"{name:<15} {plaba_metrics[key]:>12.2f} {medreadme_metrics[key]:>12.2f}")
    print("-" * 70)

if __name__ == "__main__":
    compare_datasets()
