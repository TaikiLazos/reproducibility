# TODO: unnecessary training is occurring. 
# MEDREADME (train) -> PLABA (test) [MODEL 1]
# PLABA (train) -> MEDREADME (test) [MODEL 2]
# PLABA (train) [MODEL 2] + MEDREADME (train) -> MEDREADME (test) [MODEL 3]
# PLABA (train) [MODEL 2] + MEDREADME (train) -> PLABA (test) [MODEL 3]
# MEDREADME (train) [MODEL 1] + PLABA (train) -> MEDREADME (test) [MODEL 4]
# MEDREADME (train) [MODEL 1] + PLABA (train) -> PLABA (test) [MODEL 4]

# Where in the + it means first train on the left and then fine-tune it on the right



import argparse
import torch
import os
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW
from plaba import PLABADataset, train, test, print_results
from medreadme import MedReadmeDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

models = {
    'bert': 'bert-large-cased',
    'roberta': 'roberta-large'
}

def setup_experiment(args):
    """Setup datasets and model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache_dir = "./cache_models"
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs("output/transfer", exist_ok=True)

    # Initialize tokenizer and datasets
    tokenizer = AutoTokenizer.from_pretrained(models[args.model_name], add_prefix_space=True, cache_dir=cache_dir)
    plaba_dataset = PLABADataset(tokenizer, args.data_dir)
    medreadme_dataset = MedReadmeDataset(tokenizer, 'data/medreadme/jargon.json', classification_type='binary')

    return device, tokenizer, plaba_dataset, medreadme_dataset

def create_dataloaders(dataset, batch_size, is_plaba=True):
    """Create dataloaders for a dataset"""
    if is_plaba:
        train_loader = DataLoader(dataset.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset.val_dataset, batch_size=batch_size)
        test_loader = DataLoader(dataset.test_dataset, batch_size=batch_size)
    else:
        train_loader = DataLoader(dataset.get_split('train'), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset.get_split('dev'), batch_size=batch_size)
        test_loader = DataLoader(dataset.get_split('test'), batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

def run_experiment(args, device, model_name, train_dataset, test_dataset, experiment_name):
    """Run a single transfer learning experiment"""
    print(f"\nRunning experiment: {experiment_name}")
    
    # Create model - handle both HuggingFace model names and local .pt files
    if isinstance(model_name, str) and model_name.endswith('.pt'):
        # For local .pt files, first create a new model with the same architecture
        base_model = AutoModelForTokenClassification.from_pretrained(
            models[args.model_name],
            num_labels=2,
            cache_dir="./cache_models"
        )
        model = AutoModelForTokenClassification.from_config(base_model.config)
        # Then load the saved weights
        print(f"Loading weights from {model_name}")
        model.load_state_dict(torch.load(model_name))
    else:
        # For HuggingFace models, load directly
        print(f"Loading model from HuggingFace: {model_name}")
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=2,
            cache_dir="./cache_models"
        )
    
    model.to(device)

    # Create dataloaders
    is_plaba_train = isinstance(train_dataset, PLABADataset)
    is_plaba_test = isinstance(test_dataset, PLABADataset)
    
    train_loader, val_loader, _ = create_dataloaders(train_dataset, args.batch_size, is_plaba_train)
    _, _, test_loader = create_dataloaders(test_dataset, args.batch_size, is_plaba_test)

    # Train model
    save_path = f'output/transfer/{experiment_name}.pt'
    best_f1 = train(model, train_loader, val_loader, device, args, save_path)
    
    # Test model
    test_results = test(save_path, test_loader, device, model)
    
    return test_results

def main(args):
    device, tokenizer, plaba_dataset, medreadme_dataset = setup_experiment(args)
    
    # Store results
    all_results = {}
    
    # Run initial training experiments to get MODEL 1 and MODEL 2
    base_experiments = [
        {
            'name': 'medreadme_to_plaba',  # This will create MODEL 1
            'train': medreadme_dataset,
            'test': plaba_dataset
        },
        {
            'name': 'plaba_to_medreadme',  # This will create MODEL 2
            'train': plaba_dataset,
            'test': medreadme_dataset
        }
    ]

    # Run base experiments and store results
    for exp in base_experiments:
        results = run_experiment(
            args, 
            device, 
            models[args.model_name],
            exp['train'],
            exp['test'],
            exp['name']
        )
        all_results[exp['name']] = results

    # Define fine-tuning experiments using the pre-trained models
    fine_tune_experiments = [
        {
            'name': 'plaba_plus_medreadme_to_medreadme',
            'base_model': 'output/transfer/plaba_to_medreadme.pt',  # MODEL 2
            'finetune': medreadme_dataset,
            'test': medreadme_dataset
        },
        {
            'name': 'plaba_plus_medreadme_to_plaba',
            'base_model': 'output/transfer/plaba_to_medreadme.pt',  # MODEL 2
            'finetune': medreadme_dataset,
            'test': plaba_dataset
        },
        {
            'name': 'medreadme_plus_plaba_to_medreadme',
            'base_model': 'output/transfer/medreadme_to_plaba.pt',  # MODEL 1
            'finetune': plaba_dataset,
            'test': medreadme_dataset
        },
        {
            'name': 'medreadme_plus_plaba_to_plaba',
            'base_model': 'output/transfer/medreadme_to_plaba.pt',  # MODEL 1
            'finetune': plaba_dataset,
            'test': plaba_dataset
        }
    ]

    # Run fine-tuning experiments
    for exp in fine_tune_experiments:
        # Fine-tune directly from the saved base model
        finetune_results = run_experiment(
            args,
            device,
            exp['base_model'],  # Use the appropriate pre-trained model
            exp['finetune'],
            exp['test'],
            exp['name']
        )
        all_results[exp['name']] = finetune_results

    # Save all results
    with open('output/transfer/results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print results
    print("\n=== Transfer Learning Results ===")
    for exp_name, results in all_results.items():
        print(f"\nExperiment: {exp_name}")
        print_results(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="roberta", help="Name of the pretrained model to use")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--data_dir", type=str, default="data/PLABA_2024-Task_1", help="Directory containing the data files")
    args = parser.parse_args()

    main(args)