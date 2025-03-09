import argparse
import torch
import os
from transformers import AutoModelForTokenClassification, AutoTokenizer
from torch.utils.data import Dataset
models = {
    'roberta': ('roberta-base', 'roberta-large'),
}

class PLABADataset(Dataset):
    def __init__(self, tokenizer, data_path, classification_type='binary'):
        self.tokenizer = tokenizer
        self.classification_type = classification_type
        self.data_path = data_path
        



def run_plaba_1a():
    pass

def run_plaba_1b():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="roberta", help="Name of the pretrained model to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--experiment_type", type=str, default="1a", choices=["1a", "1b"], help="1a: detection, 1b: classification")
    # parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the data files")
    # parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model outputs")
    args = parser.parse_args()

    if args.experiment_type == '1a':
        run_plaba_1a()
    elif args.experiment_type == '1b':
        run_plaba_1b()
    else:
        print("Something went wrong.")