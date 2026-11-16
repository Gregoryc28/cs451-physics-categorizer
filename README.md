# AI-Assisted Physics Problem Categorization 

This repository contains the prototype and evaluation pipeline for categorizing physics problems (Mechanics vs. Electromagnetism) to optimize tutoring workflows at the UA ENGenuity Lab.

## Requirements
- Python 3.10+
- Node.js 18+
- An OpenRouter API Key (with access to TypeSafe Jev System 1)

## Setup Instructions

### 1. Data & Evaluation Environment
1. Create and activate a virtual environment: `python -m venv .venv`
2. Install dependencies: `pip install requests scikit-learn pandas datasets fastapi uvicorn python-dotenv`
3. Create a `.env` file in the root directory and add your key: `OPENROUTER_API_KEY=your_key_here`
4. Run the data preparation script to download the Hugging Face SciQ dataset: `python prepare_data.py`
5. Run the evaluation to generate the results CSV: `python evaluate.py`

### 2. Running the Prototype
**Backend (FastAPI):**
In your Python environment, run the API server:
`uvicorn api:app --reload`

**Frontend (Next.js):**
1. Navigate to the `tutor-frontend` directory.
2. Install dependencies (if not already done): `npm install`
3. Start the development server: `npm run dev`
4. Navigate to `http://localhost:3000` to use the Workflow Dropzone.

## Notes
- Ensure that the backend API is running before using the frontend.
- The `.env` file should contain your OpenRouter API key and the model you intend to use.
- The prototype currently supports categorization of physics problems into Mechanics and Electromagnetism categories.