import torch
from transformers import RobertaTokenizerFast, RobertaForTokenClassification
from typing import List, Tuple

class JargonDetector:
    def __init__(self, model_path: str):
        """
        Initialize the jargon detector with a trained model
        
        Args:
            model_path: Path to the saved model weights (.pt file)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Use the same base tokenizer as training
        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            'roberta-base',  # Same as training
            add_prefix_space=True
        )
        
        # Initialize model architecture
        self.model = RobertaForTokenClassification.from_pretrained(
            'roberta-base',
            num_labels=2  # Binary classification: O and COMPLEX
        )
        
        # Load your trained weights
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()

    def detect_jargons(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Detect jargons in the given text
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of tuples containing (jargon_term, start_idx, end_idx)
        """
        # Tokenize the text
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=512
        )
        
        # Move inputs to device
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        offset_mapping = inputs["offset_mapping"][0].tolist()
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().tolist()
        
        # Extract jargon terms
        jargons = []
        current_jargon = []
        current_start = None
        
        for idx, (pred, (start, end)) in enumerate(zip(predictions[1:-1], offset_mapping[1:-1])):  # Skip [CLS] and [SEP]
            if pred != 0:  # Non-O prediction
                if not current_jargon:
                    current_start = start
                current_jargon.append(text[start:end])
            elif current_jargon:  # End of jargon term
                jargon_text = ' '.join(current_jargon)
                jargons.append((jargon_text, current_start, end))
                current_jargon = []
                current_start = None
        
        # Handle case where jargon is at end of text
        if current_jargon:
            jargon_text = ' '.join(current_jargon)
            jargons.append((jargon_text, current_start, offset_mapping[-2][1]))
        
        return jargons

def main():
    # Initialize detector with your trained model
    detector = JargonDetector("output/RoBERTa_base_binary.pt")
    
    # Example texts
    texts = [
        "The patient exhibited signs of myocardial infarction.",
        "Treatment included administration of acetylsalicylic acid.",
    ]
    
    # Process each text
    for text in texts:
        print("\nAnalyzing text:", text)
        jargons = detector.detect_jargons(text)
        print("Detected jargons:")
        for term, start, end in jargons:
            print(f"- '{term}' (position {start}-{end})")

if __name__ == "__main__":
    main() 