from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import requests
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Physics Tutor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProblemRequest(BaseModel):
    text: str

@app.post("/api/categorize")
async def categorize_problem(request: ProblemRequest):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing API Key")
        
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "typesafe/jev-1.14",
        "state": request.text,
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
        return {"topic": content.get("answers", {}).get("topic", {}).get("choice", "Error")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))