
"""
Simple test server to debug backend issues
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Backend server is running!"}

@app.post("/api/ask")
async def ask_question(request: dict):
    question = request.get("question", "")
    
    if not question:
        return {"answer": "🌾 Welcome to your AI farming assistant! I can help you with crop recommendations, weather advice, and farming best practices. What would you like to know?"}
    
    return {"answer": f"You asked: '{question}'. The backend server is working! (This is a test response)"}

if __name__ == "__main__":
    print("Starting test backend server...")
    uvicorn.run(app, host="0.0.0.0", port=8001)
