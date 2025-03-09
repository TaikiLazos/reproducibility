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
            'roberta-large',  # Same as training
            add_prefix_space=True
        )
        
        # Initialize model architecture
        self.model = RobertaForTokenClassification.from_pretrained(
            'roberta-large',
            num_labels=2  # Binary classification: O and COMPLEX
        )
        
        # Load just the model state dict
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()
        
        # Define punctuation characters to exclude
        self.punctuation_chars = set('.,;:!?()[]{}"-')

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
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().tolist()
        
        # Extract jargon terms
        jargons = []
        i = 0
        while i < len(predictions):
            if predictions[i] == 1:  # Jargon token
                start_offset = offset_mapping[i][0]
                
                # Skip special tokens with no character mapping
                if offset_mapping[i][0] == offset_mapping[i][1]:
                    i += 1
                    continue
                
                # Find the end of this jargon span
                j = i
                while j < len(predictions) and predictions[j] == 1:
                    j += 1
                
                end_offset = offset_mapping[j-1][1] if j-1 < len(offset_mapping) else len(text)
                
                # Extract the raw jargon term
                if start_offset < end_offset:
                    jargon_term = text[start_offset:end_offset]
                    
                    # Clean up jargon term
                    jargon_term = jargon_term.strip()
                    
                    # Remove trailing punctuation
                    while jargon_term and jargon_term[-1] in self.punctuation_chars:
                        jargon_term = jargon_term[:-1].strip()
                    
                    # Remove leading punctuation
                    while jargon_term and jargon_term[0] in self.punctuation_chars:
                        jargon_term = jargon_term[1:].strip()
                    
                    # Only add if not empty after cleaning and at least 2 chars
                    if jargon_term and len(jargon_term) > 1:
                        # Adjust start and end positions to match cleaned term
                        new_start = text.find(jargon_term, start_offset)
                        if new_start >= 0:
                            new_end = new_start + len(jargon_term)
                            jargons.append((jargon_term, new_start, new_end))
                
                i = j
            else:
                i += 1
        
        return jargons

def main():
    # Initialize detector with your trained model
    detector = JargonDetector("output/roberta_large_plaba.pt")
    
    # Example texts
    texts = [
        """We studied 36 drop seizures in 5 patients with myoclonic astatic epilepsy of early childhood (MAEE) with simultaneous split-screen video recording and polygraph.
Sixteen were falling attacks and 20 were either less severe attacks exhibiting only deep head nodding or seizures equivalent to drop attacks in terms of ictal pattern but recorded in the supine position.
All seizures except those that occurred in patients in the supine position showed sudden momentary head dropping or collapse of the whole body downward.
Recovery to the preictal position was observed in 0.3-1 s.
As a result of carefully repeated observations, the 36 seizures were classified as myoclonic flexor type in 9, myoclonic atonic type in 2, and atonic type, with and without transient preceding symptoms in the remaining 25.
The MF seizure was characterized by sudden forward flexion of the head and trunk as well as both arms, which caused the patient to fall.
In the myoclonic atonic seizure, patients showed brief myoclonic flexor spasms, immediately followed by atonic falling.
The AT seizure showed abrupt atonic falling, with and without transient preceding facial expression change and/or twitching of extremities.
The ictal EEGs of all 36 seizures exhibited generalized bilaterally synchronous single or multiple spike(s) and wave discharges.
Atonic drop attacks appear to be a common cause of ictal epileptic falling in MAEE."""
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