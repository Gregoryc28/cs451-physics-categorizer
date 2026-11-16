import json
import pandas as pd
from datasets import load_dataset

def prepare_real_physics_data():
    print("Downloading SciQ dataset via Hugging Face...")
    # Updated to include the "allenai/" namespace
    dataset = load_dataset("allenai/sciq", split="train")
    df = dataset.to_pandas()
    
    # Keyword filters to classify raw text into our two target topics
    mechanics_keywords = ['velocity', 'acceleration', 'force', 'gravity', 'mass', 'friction', 'momentum', 'kinetic']
    em_keywords = ['charge', 'voltage', 'current', 'magnetic', 'electric', 'circuit', 'resistance', 'electron']
    
    formatted_problems = []
    
    for _, row in df.iterrows():
        question = row['question'].lower()
        topic = None
        
        if any(kw in question for kw in mechanics_keywords):
            topic = "Mechanics"
        elif any(kw in question for kw in em_keywords):
            topic = "Electromagnetism"
            
        if topic:
            formatted_problems.append({
                "id": f"sciq_{row.name}",
                "text": row['question'],
                "true_topic": topic
            })
            
        # Stop once we have exactly 100 balanced problems for the evaluation
        if len(formatted_problems) == 100:
            break

    print(f"Extracted {len(formatted_problems)} real physics problems.")
    
    with open("physics_dataset.json", "w") as f:
        json.dump(formatted_problems, f, indent=4)
    print("Saved to physics_dataset.json. You can now run evaluate.py!")

if __name__ == "__main__":
    prepare_real_physics_data()