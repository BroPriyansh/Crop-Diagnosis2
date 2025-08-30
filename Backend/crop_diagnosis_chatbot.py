from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from google import genai
import uvicorn
from typing import List, Optional
import uuid

# Load environment variables
load_dotenv()

# Get API key (same as ChatBot_fixed.py)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

print(f"GEMINI_API_KEY configured: {bool(GEMINI_API_KEY)}")

# Initialize FastAPI app
app = FastAPI(title="Crop Diagnosis Chatbot", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini client
print("Initializing Gemini client...")
try:
    genai_client = genai.Client(api_key=GEMINI_API_KEY)
    print("Gemini client initialized successfully")
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    raise

# Request models
class DiagnosisRequest(BaseModel):
    disease_name: Optional[str] = None
    disease_image_description: Optional[str] = None
    symptoms: Optional[str] = None
    crop_type: Optional[str] = None
    follow_up_question: Optional[str] = None
    session_id: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class SessionRequest(BaseModel):
    session_id: Optional[str] = None

# Response models
class DiagnosisResponse(BaseModel):
    response: str
    disease_name: Optional[str] = None
    causes: List[str] = []
    symptoms: List[str] = []
    solutions: List[str] = []
    prevention: List[str] = []
    confidence: float = 0.85
    session_id: str

class ChatResponse(BaseModel):
    response: str
    session_id: str

# In-memory conversation storage (use database in production)
conversations = {}

@app.get("/")
async def root():
    return {"message": "Crop Diagnosis Chatbot API is running!", "status": "ok"}

@app.get("/test")
async def test():
    return {"message": "Backend is working!", "timestamp": "2025-08-30T23:20:00", "status": "ok"}

@app.post("/api/start_session")
async def start_session(request: SessionRequest):
    """Start a new conversation session or get existing one"""
    session_id = request.session_id or str(uuid.uuid4())
    
    if session_id not in conversations:
        conversations[session_id] = {
            "messages": [],
            "disease_context": None
        }
    
    return {"session_id": session_id, "message": "Session started. You can now ask about crop diseases!"}

@app.post("/api/diagnose", response_model=DiagnosisResponse)
async def diagnose_crop(request: DiagnosisRequest):
    """
    Main endpoint for crop disease diagnosis - handles initial disease identification and follow-up questions
    """
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        # Initialize session if not exists
        if session_id not in conversations:
            conversations[session_id] = {"messages": [], "disease_context": None}
        
        session = conversations[session_id]
        
        # Determine if this is initial diagnosis or follow-up
        if request.disease_name or request.disease_image_description:
            # Initial disease diagnosis
            prompt = create_disease_diagnosis_prompt(
                disease_name=request.disease_name,
                image_description=request.disease_image_description,
                symptoms=request.symptoms,
                crop_type=request.crop_type
            )
            # Store disease context for follow-up questions
            session["disease_context"] = {
                "disease_name": request.disease_name,
                "crop_type": request.crop_type,
                "symptoms": request.symptoms
            }
        else:
            # Follow-up question
            prompt = create_followup_prompt(
                question=request.follow_up_question,
                disease_context=session["disease_context"],
                conversation_history=session["messages"]
            )
        
        print(f"Sending diagnosis request to Gemini...")
        
        # Call Gemini API
        response = genai_client.models.generate_content(
            model="gemini-1.5-pro",
            contents=prompt
        )
        
        diagnosis_text = response.text
        print(f"Received Gemini response: {diagnosis_text[:100]}...")
        
        # Store conversation
        session["messages"].append({
            "type": "user",
            "content": request.disease_name or request.follow_up_question or "Disease diagnosis request"
        })
        session["messages"].append({
            "type": "assistant",
            "content": diagnosis_text
        })
        
        # Parse structured response for initial diagnosis
        if request.disease_name or request.disease_image_description:
            parsed_response = parse_disease_response(diagnosis_text)
            return DiagnosisResponse(
                response=diagnosis_text,
                disease_name=parsed_response.get("disease_name", request.disease_name),
                causes=parsed_response.get("causes", []),
                symptoms=parsed_response.get("symptoms", []),
                solutions=parsed_response.get("solutions", []),
                prevention=parsed_response.get("prevention", []),
                confidence=0.85,
                session_id=session_id
            )
        else:
            return DiagnosisResponse(
                response=diagnosis_text,
                session_id=session_id
            )
        
    except Exception as e:
        print(f"Error in diagnose_crop: {e}")
        raise HTTPException(status_code=500, detail=f"Diagnosis failed: {str(e)}")

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    """
    Continuous chat endpoint - maintains conversation context
    """
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        # Initialize session if not exists
        if session_id not in conversations:
            conversations[session_id] = {"messages": [], "disease_context": None}
        
        session = conversations[session_id]
        
        # Create context-aware prompt
        prompt = create_chat_prompt(
            message=request.message,
            conversation_history=session["messages"],
            disease_context=session["disease_context"]
        )
        
        print(f"Sending chat request to Gemini...")
        
        # Call Gemini API
        response = genai_client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        
        answer = response.text
        print(f"Received chat response: {answer[:100]}...")
        
        # Store conversation
        session["messages"].append({"type": "user", "content": request.message})
        session["messages"].append({"type": "assistant", "content": answer})
        
        return ChatResponse(response=answer, session_id=session_id)
        
    except Exception as e:
        print(f"Error in chat_with_bot: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

def create_disease_diagnosis_prompt(disease_name: str = None, image_description: str = None, symptoms: str = None, crop_type: str = None):
    """
    Create initial disease diagnosis prompt
    """
    prompt = """You are an expert plant pathologist and agricultural scientist specializing in crop diseases.
    
    Provide a comprehensive analysis of the crop disease with the following structure:
    
    **DISEASE IDENTIFICATION:**
    Disease Name: [Provide the scientific and common name]
    
    **WHAT CAUSES THIS DISEASE:**
    - List the primary causes (pathogen, environmental factors, etc.)
    - Explain how the disease spreads
    - Mention favorable conditions for disease development
    
    **SYMPTOMS TO LOOK FOR:**
    - Early stage symptoms
    - Advanced stage symptoms
    - How to distinguish from similar diseases
    
    **TREATMENT SOLUTIONS:**
    - Immediate treatment steps
    - Chemical treatments (if applicable)
    - Organic/biological treatments
    - Cultural practices for management
    
    **PREVENTION MEASURES:**
    - Preventive practices
    - Resistant varieties (if available)
    - Environmental management
    - Crop rotation recommendations
    
    **ADDITIONAL ADVICE:**
    - When to seek professional help
    - Economic impact considerations
    - Long-term management strategies
    
    Information provided:"""
    
    if disease_name:
        prompt += f"\nDisease Name: {disease_name}"
    if crop_type:
        prompt += f"\nCrop Type: {crop_type}"
    if image_description:
        prompt += f"\nImage Description: {image_description}"
    if symptoms:
        prompt += f"\nObserved Symptoms: {symptoms}"
    
    prompt += "\n\nProvide detailed, practical advice that farmers can easily understand and implement."
    
    return prompt

def create_followup_prompt(question: str, disease_context: dict = None, conversation_history: list = None):
    """
    Create follow-up question prompt with context
    """
    prompt = """You are an expert agricultural advisor continuing a conversation about crop diseases.
    
    Previous context:"""
    
    if disease_context:
        prompt += f"\nDisease being discussed: {disease_context.get('disease_name', 'Unknown')}"
        prompt += f"\nCrop type: {disease_context.get('crop_type', 'Unknown')}"
        prompt += f"\nSymptoms: {disease_context.get('symptoms', 'Not specified')}"
    
    if conversation_history and len(conversation_history) > 0:
        prompt += "\n\nRecent conversation:"
        # Include last 4 messages for context
        for msg in conversation_history[-4:]:
            prompt += f"\n{msg['type'].title()}: {msg['content'][:100]}..."
    
    prompt += f"\n\nNew question: {question}"
    prompt += "\n\nProvide a helpful, detailed response that builds on the previous context. Be specific and practical."
    
    return prompt

def create_chat_prompt(message: str, conversation_history: list = None, disease_context: dict = None):
    """
    Create general chat prompt with conversation context
    """
    prompt = """You are an expert agricultural advisor and crop specialist. 
    Answer questions about crops, farming, plant diseases, and agricultural practices.
    Provide practical, actionable advice."""
    
    if disease_context:
        prompt += f"\n\nCurrent disease context: {disease_context.get('disease_name', 'Unknown')} on {disease_context.get('crop_type', 'crops')}"
    
    if conversation_history and len(conversation_history) > 0:
        prompt += "\n\nConversation history:"
        # Include last 6 messages for context
        for msg in conversation_history[-6:]:
            prompt += f"\n{msg['type'].title()}: {msg['content'][:150]}..."
    
    prompt += f"\n\nUser question: {message}"
    prompt += "\n\nProvide a helpful and informative response."
    
    return prompt

def parse_disease_response(diagnosis_text: str) -> dict:
    """
    Parse structured disease diagnosis response
    """
    result = {
        "disease_name": "",
        "causes": [],
        "symptoms": [],
        "solutions": [],
        "prevention": []
    }
    
    lines = diagnosis_text.split('\n')
    current_section = None
    
    for line in lines:
        line = line.strip()
        
        # Identify sections
        if "DISEASE IDENTIFICATION" in line.upper() or "Disease Name" in line:
            current_section = "disease_name"
        elif "CAUSES" in line.upper() or "WHAT CAUSES" in line.upper():
            current_section = "causes"
        elif "SYMPTOMS" in line.upper():
            current_section = "symptoms"
        elif "TREATMENT" in line.upper() or "SOLUTIONS" in line.upper():
            current_section = "solutions"
        elif "PREVENTION" in line.upper():
            current_section = "prevention"
        elif line.startswith(('**', '#')) or len(line) < 3:
            continue
        
        # Extract content based on current section
        if current_section and line:
            if current_section == "disease_name" and "Disease Name:" in line:
                result["disease_name"] = line.split(":", 1)[1].strip()
            elif current_section in ["causes", "symptoms", "solutions", "prevention"]:
                if line.startswith(('-', '•', '*')) or line[0].isdigit():
                    cleaned = line.lstrip('1234567890.-•* ').strip()
                    if cleaned and len(cleaned) > 5:
                        result[current_section].append(cleaned)
    
    return result

if __name__ == "__main__":
    import socket
    
    # Check if port is available
    def is_port_available(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return True
            except OSError:
                return False
    
    port = 8005
    if not is_port_available(port):
        print(f"Port {port} is already in use. Trying port 8006...")
        port = 8006
        if not is_port_available(port):
            print("Ports 8005 and 8006 are both in use. Please close other applications using these ports.")
            exit(1)
    
    print(f"Starting Crop Diagnosis Chatbot server on http://0.0.0.0:{port}")
    print(f"Web browser can access via: http://localhost:{port}")
    print("Press Ctrl+C to stop the server")
    print("Server is running... Keep this terminal open!")
    
    try:
        # Get local IP address for Android emulator access
        import socket
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        print(f"Local IP address: {local_ip}")
        print(f"For Android emulator, use: http://{local_ip}:{port}")
        
        uvicorn.run(app, host="0.0.0.0", port=port, reload=False, log_level="info")
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Server error: {e}")
        print("Please check if all dependencies are installed: pip install fastapi uvicorn python-dotenv google-generativeai")
    
    
