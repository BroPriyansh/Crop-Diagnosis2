# MultipleFiles/crop_diagnosis_chatbot.py
import os
import asyncio
import json
import re
from typing import Dict, List, Any, Optional
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure Google Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model
generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 60,
    "max_output_tokens": 1024,
}
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    safety_settings=safety_settings,
)

# In-memory storage for conversations (for development purposes)
# In a production environment, replace this with a proper database (e.g., Redis, PostgreSQL)
conversations: Dict[str, Dict[str, Any]] = {}

# --- Pydantic Models ---
class DiagnosisRequest(BaseModel):
    disease_name: Optional[str] = None
    disease_image_description: Optional[str] = None # Used for initial prompt if image was source
    symptoms: Optional[str] = None
    crop_type: Optional[str] = None
    follow_up_question: Optional[str] = None
    session_id: Optional[str] = None
    
    # Fields to receive prediction results from main.py (now sent by frontend)
    predicted_class: Optional[str] = None
    confidence: Optional[float] = None
    top_predictions: Optional[List[Dict[str, Any]]] = None
    model_validation: Optional[Dict[str, Any]] = None

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class SessionRequest(BaseModel):
    session_id: Optional[str] = None

class DiagnosisResponse(BaseModel):
    response: str
    disease_name: Optional[str] = None
    confidence: Optional[float] = None # Added confidence to response
    causes: Optional[List[str]] = None
    symptoms: Optional[List[str]] = None
    solutions: Optional[List[str]] = None
    prevention: Optional[List[str]] = None
    session_id: str

class ChatResponse(BaseModel):
    response: str
    session_id: str

# --- FastAPI App Setup ---
app = FastAPI(
    title="Crop Diagnosis Chatbot",
    description="AI-powered chatbot for crop disease diagnosis and farming advice.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# --- Helper Functions ---

def extract_crop_type_from_disease(disease_class: str) -> str:
    """Extracts the crop type from a disease class string (e.g., 'Apple___Apple_scab' -> 'Apple')."""
    if "___" in disease_class:
        return disease_class.split("___")[0]
    return "Unknown Crop"

def parse_disease_response(text: str) -> Dict[str, Any]:
    """Parses the AI's response into structured sections."""
    parsed_data = {
        "disease_name": None,
        "causes": [],
        "symptoms": [],
        "solutions": [],
        "prevention": []
    }

    # Extract disease name (often the first line or bolded)
    disease_name_match = re.match(r"^\s*([A-Za-z0-9\s\$\$-]+(?: disease)?)\s*[:.]?\s*$", text.split('\n')[0])
    if disease_name_match:
        parsed_data["disease_name"] = disease_name_match.group(1).strip()
    else:
        # Fallback: try to find a common disease name pattern
        first_sentence_match = re.match(r"^(.*?)\.", text)
        if first_sentence_match:
            parsed_data["disease_name"] = first_sentence_match.group(1).strip()
        
    sections = re.split(r'\n\s*(?:Causes|Symptoms|Solutions|Prevention|Treatment|Recommendations|What to do|How to prevent):?\s*\n', text, flags=re.IGNORECASE)
    
    # The first part is usually the general description
    general_description = sections[0].strip()
    
    # Iterate through the rest of the sections
    for i in range(1, len(sections)):
        section_content = sections[i].strip()
        
        # Determine which section it is based on the preceding keyword
        # This is a bit tricky with re.split, so we'll re-match the keywords
        if re.search(r'Causes:', text, re.IGNORECASE) and sections[i-1].endswith(re.search(r'Causes:', text, re.IGNORECASE).group()):
            parsed_data["causes"] = [item.strip() for item in re.split(r'[\n*-]+', section_content) if item.strip()]
        elif re.search(r'Symptoms:', text, re.IGNORECASE) and sections[i-1].endswith(re.search(r'Symptoms:', text, re.IGNORECASE).group()):
            parsed_data["symptoms"] = [item.strip() for item in re.split(r'[\n*-]+', section_content) if item.strip()]
        elif re.search(r'(Solutions|Treatment|Recommendations|What to do):', text, re.IGNORECASE) and sections[i-1].endswith(re.search(r'(Solutions|Treatment|Recommendations|What to do):', text, re.IGNORECASE).group()):
            parsed_data["solutions"] = [item.strip() for item in re.split(r'[\n*-]+', section_content) if item.strip()]
        elif re.search(r'(Prevention|How to prevent):', text, re.IGNORECASE) and sections[i-1].endswith(re.search(r'(Prevention|How to prevent):', text, re.IGNORECASE).group()):
            parsed_data["prevention"] = [item.strip() for item in re.split(r'[\n*-]+', section_content) if item.strip()]

    # If disease_name wasn't extracted from the first line, try to get it from the general description
    if not parsed_data["disease_name"] and general_description:
        first_sentence_match = re.match(r"^(.*?)\.", general_description)
        if first_sentence_match:
            parsed_data["disease_name"] = first_sentence_match.group(1).strip()

    # Ensure lists are not empty if no specific items were found, but general description exists
    if not parsed_data["causes"] and "causes" in general_description.lower():
        pass # Could add more sophisticated parsing here
    
    # The full response text is the primary 'response'
    parsed_data["response"] = text
    
    return parsed_data

def create_disease_diagnosis_prompt(request: DiagnosisRequest, conversation_history: List[Dict[str, str]]) -> str:
    """Creates a detailed prompt for initial disease diagnosis, incorporating ML model predictions."""
    prompt_parts = [
        "You are an expert agricultural AI assistant specializing in crop disease diagnosis and management. "
        "Provide a comprehensive diagnosis and actionable advice. "
        "Structure your response clearly with sections for Disease Name, Causes, Symptoms, Solutions (Treatment/Recommendations), and Prevention. "
        "Use bullet points for lists within sections. Be concise but informative."
    ]

    # Safely access top_predictions
    top_preds = request.top_predictions or []
    top_preds_str = ', '.join([f"{p['class']} ({p['confidence']:.2f})" for p in top_preds])

    if request.predicted_class:
        prompt_parts.append(
            f"\n**Based on an AI image analysis, the most likely disease is: {request.predicted_class} "
            f"with a confidence of {request.confidence:.2f}.** "
            f"Please use this as the primary basis for your diagnosis, but also provide expert insights. "
            f"The top predictions were: {top_preds_str}. "
        )
        # If the frontend sent a description along with the prediction
        if request.disease_image_description:
            prompt_parts.append(f"The user also provided this observation: '{request.disease_image_description}'.")
    elif request.disease_name:
        prompt_parts.append(f"\nThe user has identified a potential disease: {request.disease_name}.")
    elif request.disease_image_description:
        prompt_parts.append(f"\nThe user described the crop's condition: '{request.disease_image_description}'.")
    elif request.symptoms:
        prompt_parts.append(f"\nThe user described the following symptoms: '{request.symptoms}'.")

    if request.crop_type:
        prompt_parts.append(f"The crop type is: {request.crop_type}.")
    
    if request.follow_up_question:
        prompt_parts.append(f"\nAdditionally, the user has a specific question: '{request.follow_up_question}'. Address this within your comprehensive response.")

    prompt_parts.append("\nProvide the diagnosis and advice now:")
    return "\n".join(prompt_parts)

def create_followup_prompt(request: DiagnosisRequest, conversation_history: List[Dict[str, str]], disease_context: Dict[str, Any]) -> str:
    """Creates a prompt for follow-up questions, maintaining context."""
    prompt_parts = [
        "You are an expert agricultural AI assistant. The user is asking a follow-up question about a previously diagnosed crop disease. "
        "Maintain context from the previous diagnosis and conversation. "
    ]
    if disease_context:
        prompt_parts.append(f"\n**Current Disease Context:**")
        for key, value in disease_context.items():
            if value and key != "response": # Don't repeat the full response text
                if isinstance(value, list):
                    prompt_parts.append(f"- {key.replace('_', ' ').title()}: {', '.join(value)}")
                else:
                    prompt_parts.append(f"- {key.replace('_', ' ').title()}: {value}")
    
    prompt_parts.append("\n**Conversation History (most recent first):**")
    # Include last few turns for context
    for msg in conversation_history[-4:]: # Last 4 messages
        prompt_parts.append(f"{msg['sender'].capitalize()}: {msg['text']}")

    prompt_parts.append(f"\n**User's Follow-up Question:** {request.follow_up_question}")
    prompt_parts.append("\nProvide a helpful and concise answer based on the context:")
    return "\n".join(prompt_parts)

def create_chat_prompt(message: str, conversation_history: List[Dict[str, str]], disease_context: Dict[str, Any]) -> str:
    """Creates a prompt for general chat, maintaining context."""
    prompt_parts = [
        "You are an expert agricultural AI assistant. Engage in a helpful conversation about crop diseases or general farming. "
        "Maintain context from previous interactions."
    ]
    if disease_context:
        prompt_parts.append(f"\n**Current Disease Context:**")
        for key, value in disease_context.items():
            if value and key != "response":
                if isinstance(value, list):
                    prompt_parts.append(f"- {key.replace('_', ' ').title()}: {', '.join(value)}")
                else:
                    prompt_parts.append(f"- {key.replace('_', ' ').title()}: {value}")
    
    prompt_parts.append("\n**Conversation History (most recent first):")
    for msg in conversation_history[-6:]: # Last 6 messages
        prompt_parts.append(f"{msg['sender'].capitalize()}: {msg['text']}")

    prompt_parts.append(f"\n**User's Message:** {message}")
    prompt_parts.append("\nProvide a helpful and concise answer:")
    return "\n".join(prompt_parts)

async def get_gemini_response(prompt: str) -> str:
    """Sends a prompt to the Gemini model and returns the response."""
    try:
        chat_session = model.start_chat(history=[])
        response = await chat_session.send_message_async(prompt)
        return response.text
    except Exception as e:
        print(f"Error getting Gemini response: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get response from AI model: {e}"
        )

async def diagnose_crop(request: DiagnosisRequest) -> DiagnosisResponse:
    """Handles the core diagnosis logic, generating a session if needed."""
    session_id = request.session_id or f"session_{os.urandom(16).hex()}"

    if session_id not in conversations:
        conversations[session_id] = {
            "history": [],
            "disease_context": {}
        }

    history = conversations[session_id]["history"]
    disease_context = conversations[session_id]["disease_context"]

    # Add user's initial input to history
    user_input_text = ""
    if request.predicted_class:
        user_input_text = f"Image analysis suggests: {request.predicted_class} (Confidence: {request.confidence:.2f})."
    elif request.disease_name:
        user_input_text = f"I think my crop has {request.disease_name}."
    elif request.disease_image_description:
        user_input_text = f"I observed: {request.disease_image_description}."
    elif request.symptoms:
        user_input_text = f"My crop has these symptoms: {request.symptoms}."
    
    if request.follow_up_question:
        user_input_text += f" My question is: {request.follow_up_question}"

    if user_input_text:
        history.append({"sender": "user", "text": user_input_text})

    if request.follow_up_question and disease_context:
        prompt = create_followup_prompt(request, history, disease_context)
    else:
        prompt = create_disease_diagnosis_prompt(request, history)

    ai_response_text = await get_gemini_response(prompt)
    
    # Parse the AI's response into structured data
    parsed_data = parse_disease_response(ai_response_text)
    
    # Update disease context for future follow-ups
    conversations[session_id]["disease_context"].update({
        "disease_name": parsed_data.get("disease_name"),
        "causes": parsed_data.get("causes"),
        "symptoms": parsed_data.get("symptoms"),
        "solutions": parsed_data.get("solutions"),
        "prevention": parsed_data.get("prevention"),
        "confidence": request.confidence # Store confidence from ML model
    })

    # Add AI's response to history
    history.append({"sender": "bot", "text": ai_response_text})

    return DiagnosisResponse(
        response=ai_response_text,
        disease_name=parsed_data.get("disease_name"),
        confidence=request.confidence, # Pass confidence from ML model to frontend
        causes=parsed_data.get("causes"),
        symptoms=parsed_data.get("symptoms"),
        solutions=parsed_data.get("solutions"),
        prevention=parsed_data.get("prevention"),
        session_id=session_id
    )

# --- API Endpoints ---

@app.get("/")
async def root():
    return {"message": "Crop Diagnosis Chatbot API is running!"}

@app.get("/test")
async def test_endpoint():
    try:
        response = await get_gemini_response("Hello, what is your purpose?")
        return {"message": "Gemini API test successful", "gemini_response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API test failed: {e}")

@app.post("/api/start_session", response_model=SessionRequest)
async def start_session(request: SessionRequest):
    session_id = request.session_id or f"session_{os.urandom(16).hex()}"
    if session_id not in conversations:
        conversations[session_id] = {
            "history": [],
            "disease_context": {}
        }
    return {"session_id": session_id}

@app.post("/api/diagnose", response_model=DiagnosisResponse)
async def diagnose(request: DiagnosisRequest):
    """
    Endpoint for diagnosis requests.
    Receives JSON data, which can include ML model predictions or text descriptions.
    """
    return await diagnose_crop(request)

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    session_id = request.session_id
    if not session_id or session_id not in conversations:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or missing session_id. Please start a diagnosis first."
        )

    history = conversations[session_id]["history"]
    disease_context = conversations[session_id]["disease_context"]

    # Add user message to history
    history.append({"sender": "user", "text": request.message})

    prompt = create_chat_prompt(request.message, history, disease_context)
    ai_response_text = await get_gemini_response(prompt)

    # Add AI response to history
    history.append({"sender": "bot", "text": ai_response_text})

    return ChatResponse(response=ai_response_text, session_id=session_id)

# --- Server Execution ---
if __name__ == "__main__":
    import uvicorn
    import socket

    def get_local_ip():
        """Get the local IP address of the machine."""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # Doesn't actually connect to anything, just used to get the IP
            s.connect(('192.255.255.255', 1))
            IP = s.getsockname()[0]
        except Exception:
            IP = '127.0.0.1'
        finally:
            s.close()
        return IP

    local_ip = get_local_ip()
    port = 8005
    
    print(f"Starting Crop Diagnosis Chatbot API on http://{local_ip}:{port}")
    print(f"Ensure ML Model API (main.py) is running on http://127.0.0.1:8000")
    print(f"For Android emulator/device, use http://{local_ip}:{port} in your React Native app.")
    
    uvicorn.run(app, host="0.0.0.0", port=port)