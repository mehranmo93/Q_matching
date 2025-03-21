from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.spatial.distance import cosine
import uvicorn

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")

# In-memory user data (use database in production)
user_profiles = {}  # {user_id: {"text": str, "embedding": np.array}}

class UserText(BaseModel):
    user_id: str
    text: str

@app.post("/submit")
def submit_text(data: UserText):
    # Generate embedding for new user text
    embedding = model.encode(data.text)
    user_profiles[data.user_id] = {"text": data.text, "embedding": embedding}

    # Calculate similarity with all other users
    results = []
    for other_id, info in user_profiles.items():
        if other_id == data.user_id:
            continue
        other_embedding = info["embedding"]
        sim_score = 1 - cosine(embedding, other_embedding)
        results.append({"user_id": other_id, "text": info["text"], "score": round(sim_score, 4)})

    # Sort by similarity score descending
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return {"matches": results}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=10000, reload=True)
