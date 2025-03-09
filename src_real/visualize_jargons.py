# Maybe make a PNG after creating the html?
# apply a parser argument

import json
from inference_jargon import JargonDetector
import html

def load_test_data(abstract_path, annotations_path):
    # Load the first abstract
    with open(abstract_path, 'r') as f:
        text = f.read()
    
    # Load annotations
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    # Get first abstract's jargons (Q19_A1)
    first_id = "Q19_A1"  # Explicitly use Q19_A1
    true_jargons = list(annotations[first_id].keys())
    
    return text, true_jargons

def create_html_visualization(text, true_jargons, predicted_jargons):
    # Sort jargons by length (longest first) to handle nested terms correctly
    true_jargons = sorted(true_jargons, key=len, reverse=True)
    predicted_jargons = sorted(predicted_jargons, key=lambda x: len(x[0]), reverse=True)
    
    # Create a list of characters with their tags
    chars = list(text)
    true_tags = [False] * len(chars)
    pred_tags = [False] * len(chars)
    
    # Mark true jargons
    for jargon in true_jargons:
        start = 0
        while True:
            pos = text.find(jargon, start)
            if pos == -1:
                break
            for i in range(pos, pos + len(jargon)):
                true_tags[i] = True
            start = pos + 1
    
    # Mark predicted jargons
    for jargon, start, end in predicted_jargons:
        for i in range(start, end):
            if i < len(pred_tags):
                pred_tags[i] = True
    
    # Build HTML with proper tags
    html_parts = []
    current_true = False
    current_pred = False
    
    for i, char in enumerate(chars):
        # Handle tag changes
        if true_tags[i] != current_true or pred_tags[i] != current_pred:
            # Close previous tags
            if current_true or current_pred:
                html_parts.append('</span>')
            # Open new tags
            if true_tags[i] or pred_tags[i]:
                classes = []
                if true_tags[i] and pred_tags[i]:
                    classes.append('overlap-jargon')
                elif true_tags[i]:
                    classes.append('true-jargon')
                elif pred_tags[i]:
                    classes.append('predicted-jargon')
                html_parts.append(f'<span class="{" ".join(classes)}">')
            current_true = true_tags[i]
            current_pred = pred_tags[i]
        
        # Add the character
        html_parts.append(html.escape(char))
    
    # Close any open tags
    if current_true or current_pred:
        html_parts.append('</span>')
    
    text_html = ''.join(html_parts)
    
    # Create HTML document with updated styling
    html_content = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: 'Segoe UI', Arial, sans-serif;
                line-height: 1.8;
                max-width: 800px;
                margin: 40px auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .content {{
                background-color: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .true-jargon {{
                background-color: rgba(0, 0, 255, 0.1);  /* Blue */
                border-bottom: 2px solid #0000FF;
                padding: 2px 4px;
                border-radius: 3px;
            }}
            .predicted-jargon {{
                background-color: rgba(255, 0, 0, 0.1);  /* Red */
                border-bottom: 2px solid #FF0000;
                padding: 2px 4px;
                border-radius: 3px;
            }}
            .overlap-jargon {{
                background-color: rgba(0, 255, 0, 0.1);  /* Green */
                border-bottom: 2px solid #00FF00;
                padding: 2px 4px;
                border-radius: 3px;
            }}
            .legend {{
                margin-bottom: 30px;
                padding: 15px;
                border-radius: 5px;
                background-color: white;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }}
            .legend span {{
                display: inline-block;
                margin-right: 20px;
                padding: 5px 10px;
                border-radius: 4px;
            }}
            p {{
                margin-bottom: 15px;
                text-align: justify;
            }}
        </style>
    </head>
    <body>
        <div class="legend">
            <span class="true-jargon">True Jargons (Not Predicted)</span>
            <span class="predicted-jargon">False Predictions</span>
            <span class="overlap-jargon">Correct Predictions</span>
        </div>
        <div class="content">
            <p>{text_html}</p>
        </div>
    </body>
    </html>
    """
    
    with open('jargon_visualization.html', 'w') as f:
        f.write(html_content)
    
    # Print detected jargons for debugging
    print("\nTrue jargons:")
    for jargon in true_jargons:
        print(f"- {jargon}")
    
    print("\nPredicted jargons:")
    for jargon, start, end in predicted_jargons:
        print(f"- '{jargon}' (position {start}-{end})")

def main():
    # Load data
    text, true_jargons = load_test_data(
        "data/PLABA_2024-Task_1/abstracts/Q19_A1.src.txt",
        "data/PLABA_2024-Task_1/task_1_testing.json"
    )
    
    # Get predictions
    detector = JargonDetector("output/jargon_model_roberta_large.pt")
    predicted_jargons = detector.detect_jargons(text)
    
    # Create visualization
    create_html_visualization(text, true_jargons, predicted_jargons)
    print("Visualization saved to jargon_visualization.html")

if __name__ == "__main__":
    main() 