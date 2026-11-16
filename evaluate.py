import json
import os
import csv
import time
import requests
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# ==========================================
# 1. BASELINE: TF-IDF Keyword Matching
# ==========================================
def baseline_tfidf_classifier(problem_text):
    topic_keywords = {
        "Mechanics": "mass velocity acceleration friction gravity force kinematics energy momentum",
        "Electromagnetism": "charge voltage current magnetic electric force circuit resistance capacitance"
    }
    topics = list(topic_keywords.keys())
    documents = list(topic_keywords.values())
    documents.append(problem_text)
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    return topics[similarities.argmax()]

# ==========================================
# 2. PROPOSED METHOD: TypeSafe Jev System 1
# ==========================================
def jev_categorize_problem(problem_text):
    api_key = os.getenv("OPENROUTER_API_KEY")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Jev uses the Decisions API payload format
    payload = {
        "model": "typesafe/jev-1.14",
        "state": problem_text,
        "questions": {
            "topic": {
                "type": "choice",
                "instructions": "Which physics topic best categorizes this problem?",
                "criteria": {
                    "Mechanics": "Problems involving kinematics, mass, velocity, acceleration, friction, gravity, force, energy, or momentum.",
                    "Electromagnetism": "Problems involving charge, voltage, current, magnetic fields, electric fields, circuits, resistance, or capacitance."
                }
            }
        }
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/alpha/decisions",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        content = response.json()
        # Extracting the typed choice from the Jev response
        return content.get("answers", {}).get("topic", {}).get("choice", "Error")
    except Exception as e:
        print(f"API Error: {e}")
        return "Error"

# ==========================================
# 3. BATCH EVALUATION & EXPORT
# ==========================================
def run_full_evaluation(input_filepath="physics_dataset.json", output_filepath="results.csv"):
    print(f"Loading dataset from {input_filepath}...")
    with open(input_filepath, 'r') as f:
        dataset = json.load(f)
        
    baseline_correct = 0
    jev_correct = 0
    results = []
    
    print(f"Evaluating {len(dataset)} problems...\n")
    
    for i, prob in enumerate(dataset):
        print(f"Processing {i+1}/{len(dataset)}...")
        
        baseline_pred = baseline_tfidf_classifier(prob['text'])
        jev_pred = jev_categorize_problem(prob['text'])
        
        if baseline_pred == prob['true_topic']:
            baseline_correct += 1
        if jev_pred == prob['true_topic']:
            jev_correct += 1
            
        results.append({
            "id": prob.get("id", i),
            "text": prob['text'],
            "true_topic": prob['true_topic'],
            "baseline_pred": baseline_pred,
            "jev_pred": jev_pred
        })
        
        time.sleep(1)
        
    with open(output_filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text", "true_topic", "baseline_pred", "jev_pred"])
        writer.writeheader()
        writer.writerows(results)
        
    print("\n--- Final Results ---")
    print(f"Baseline Accuracy: {(baseline_correct / len(dataset)) * 100:.2f}%")
    print(f"Jev System 1 Accuracy: {(jev_correct / len(dataset)) * 100:.2f}%")
    print(f"Detailed results saved to {output_filepath}")

if __name__ == "__main__":
    run_full_evaluation()