
from fastapi import FastAPI,Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import motor.motor_asyncio
import os
from dotenv import load_dotenv
from google import genai
from prompt import create_advice_prompt_multiple, create_user_question_prompt
from gtts import gTTS
import io
import uvicorn
from Weather import get_weather
latitude, longitude, temperature, weather_desc,city,humidity,rainfall = get_weather()
from index import get_cropName

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

print(f" MONGO_URI: {MONGO_URI}")
print(f" GEMINI_API_KEY configured: {bool(GEMINI_API_KEY)}")

class OperationModel(BaseModel):
    op_name: str = Field(..., alias="name")
    op_metadata: dict = Field(..., alias="metadata")
    op_done: bool = Field(..., alias="done")
    op_error: str | None = Field(None, alias="error")
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


print(" Connecting to MongoDB...")
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client["chatDB"]
chat_collection = db["chats"]
print(" MongoDB client initialized")


print(" Initializing Gemini client...")
try:
    genai_client = genai.Client(api_key=GEMINI_API_KEY)
    print(" Gemini client initialized")
except Exception as e:
    print(f" Error initializing Gemini client: {e}")
    raise

class AskRequest(BaseModel):
    question: str = None

class TTSRequest(BaseModel):
    text: str

first_prompt_sent = False  
print(latitude, longitude, temperature, weather_desc,humidity)
crop = get_cropName(temperature,humidity,rainfall)
print(crop)
initial_data_multiple = {
    "city":city,
    "crops": crop,
    "gps": f"{latitude},{longitude}",
    "weather": f"{weather_desc}, {temperature}",
    
    "lang": "pa"  
}
@app.get("/api/crop")
async def get_crop():
    return {"recommended_crop": crop}


@app.get("/")
async def root():
    return {"message": "FastAPI server is running!", "status": "ok"}

@app.post("/api/ask")
async def ask_question(req: AskRequest):
    global first_prompt_sent
    print(f" Received request: {req}")
    
    try:
        if not req.question or req.question.strip() == "":
            if not first_prompt_sent:
                first_prompt_sent = True
            question = create_advice_prompt_multiple(**initial_data_multiple)
            print(f" Using initial crop recommendation prompt")
        else:
            question = create_user_question_prompt(
                user_question=req.question,
                city=initial_data_multiple["city"],
                gps=initial_data_multiple["gps"],
                weather=initial_data_multiple["weather"],
                
                lang="pa" 
            )
            print(f" Using formatted user question prompt")

        print(f" Calling Gemini API...")
       
        response = genai_client.models.generate_content(
            model="gemini-1.5-flash",
            contents=question
        )
        answer = response.text
        print(f" Got Gemini response: {answer[:100]}...")

        try:
            await chat_collection.insert_one({
                "question": question,
                "answer": answer
            })
            print(" Saved to MongoDB")
        except Exception as mongo_error:
            print(f"⚠ MongoDB save failed: {mongo_error}")
            
        
        print(f" Returning text response")
        return {"answer": answer}

    except Exception as e:
        print(f" Error in ask_question: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Something went wrong: {str(e)}"}

@app.post("/api/tts")
async def text_to_speech(req: TTSRequest):
    """
    Convert text to speech and return audio file
    """
    print(f" TTS request for text: {req.text[:50]}...")
    
    try:
        
        tts = gTTS(text=req.text, lang=initial_data_multiple["lang"], slow=False, tld='com')
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        
        print(f" TTS generated successfully")
        return Response(
            content=audio_bytes.read(),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=tts_output.mp3"}
        )
        
    except Exception as e:
        print(f" Error in TTS: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"TTS failed: {str(e)}"}
    
if __name__ == "__main__":
    print(" Starting server on http://0.0.0.0:8001")
    print(" Android emulator can access via: http://10.0.2.2:8001")
    print(" Web browser can access via: http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)

