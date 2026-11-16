import json
import os
import requests
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# ==========================================
# 1. MOCK DATASET (Subset for Prototyping)
# ==========================================
# We will eventually load the PHYSICS dataset JSON here.
test_problems = [
    {
        "id": "p1",
        "text": "A 5 kg block is dropped from a height of 10 meters. Neglecting air resistance, what is its velocity just before it hits the ground?",
        "true_topic": "Mechanics"
    },
    {
        "id": "p2",
        "text": "Two point charges, +3 µC and -2 µC, are separated by a distance of 0.5 meters in a vacuum. Calculate the magnitude of the electrostatic force between them.",
        "true_topic": "Electromagnetism"
    }
]

# ==========================================
# 2. BASELINE: TF-IDF Keyword Matching
# ==========================================
def baseline_tfidf_classifier(problem_text):
    """
    Uses term frequency to categorize the problem based on predefined topic vocabularies.
    """
    topic_keywords = {
        "Mechanics": "mass velocity acceleration friction gravity force kinematics energy momentum",
        "Electromagnetism": "charge voltage current magnetic electric force circuit resistance capacitance"
    }
    
    topics = list(topic_keywords.keys())
    documents = list(topic_keywords.values())
    documents.append(problem_text) # Add the problem text to compare against the topic docs
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # Compare the problem (last row) to the topics (previous rows)
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    best_match_idx = similarities.argmax()
    
    return topics[best_match_idx]

# ==========================================
# 3. PROPOSED METHOD: OpenRouter LLM Pipeline
# ==========================================
def llm_categorize_and_hint(problem_text):
    """
    Calls an LLM via OpenRouter to categorize the physics problem and generate a 3-step hint.
    """
    system_prompt = """
    You are an expert physics tutor. Given a physics problem, you must output a raw JSON object with exactly two keys: 
    1. "topic": Categorize as exactly "Mechanics", "Electromagnetism", or "Other".
    2. "hints": A list of 3 strings forming a progressive hint structure for a student. Do NOT reveal the final numerical answer.
    """

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("API Error: Set the OPENROUTER_API_KEY environment variable.")
        return {"topic": "Error", "hints": []}
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": os.getenv("OPENROUTER_MODEL", "inclusionai/ling-3.2-flash:free"),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem_text}
        ],
        "response_format": {"type": "json_object"}
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content']
        return json.loads(content)
    except requests.HTTPError as e:
        detail = e.response.text if e.response is not None else str(e)
        print(f"API Error: {e}; response: {detail}")
        return {"topic": "Error", "hints": []}
    except requests.RequestException as e:
        print(f"API Error: {e}")
        return {"topic": "Error", "hints": []}

# ==========================================
# 4. EVALUATION RUNNER
# ==========================================
def run_evaluation():
    print("--- Starting Pipeline Evaluation ---\n")
    baseline_correct = 0
    llm_correct = 0
    
    for prob in test_problems:
        print(f"Problem: {prob['text']}")
        print(f"True Topic: {prob['true_topic']}")
        
        # Run Baseline
        baseline_pred = baseline_tfidf_classifier(prob['text'])
        if baseline_pred == prob['true_topic']:
            baseline_correct += 1
            
        # Run LLM
        llm_output = llm_categorize_and_hint(prob['text'])
        if llm_output.get("topic") == prob['true_topic']:
            llm_correct += 1
            
        print(f"  Baseline Prediction: {baseline_pred}")
        print(f"  LLM Topic: {llm_output.get('topic')}")
        print(f"  LLM Hints: {llm_output.get('hints')}\n")
        
    print("--- Results ---")
    print(f"Baseline Accuracy: {baseline_correct / len(test_problems) * 100}%")
    print(f"LLM Accuracy: {llm_correct / len(test_problems) * 100}%")

if __name__ == "__main__":
    run_evaluation()