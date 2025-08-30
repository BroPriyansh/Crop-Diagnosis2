from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import firebase_admin
from firebase_admin import credentials, auth, firestore
from datetime import datetime

# Initialize FastAPI
app = FastAPI(
    title="Firebase Auth Service",
    description="Signup, token verification, ping, and signout functionality"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Firebase Admin
if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAuth.json")  # path to service account key
    firebase_admin.initialize_app(cred)

db = firestore.client()

# ---------------------------
# Request Models
# ---------------------------
class SignUpRequest(BaseModel):
    email: str
    display_name: str
    uid: str  # UID will come from client after successful signup

class PingRequest(BaseModel):
    uid: str

# ---------------------------
# Helper: Verify ID Token
# ---------------------------
async def verify_token(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = auth_header.split("Bearer ")[1]
    try:
        decoded_token = auth.verify_id_token(token)
        return decoded_token
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid auth token: " + str(e))

# ---------------------------
# Routes
# ---------------------------

@app.post("/sign-up")
async def signup(req: SignUpRequest, decoded_token=Depends(verify_token)):
    """
    Store user info in Firestore after frontend signup
    (Frontend must call Firebase Auth first to create the user)
    """
    try:
        # Ensure UID from client matches verified token UID
        if req.uid != decoded_token["uid"]:
            raise HTTPException(status_code=403, detail="UID mismatch")

        user_ref = db.collection("users").document(req.uid)
        user_ref.set({
            "uid": req.uid,
            "email": req.email,
            "displayName": req.display_name,
            "createdAt": datetime.utcnow(),
            "lastLoginAt": datetime.utcnow(),
            "isOnline": True,
            "lastPing": datetime.utcnow()
        })
        return {"success": True, "message": "Account info saved successfully.", "uid": req.uid}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/ping")
async def ping(req: PingRequest, decoded_token=Depends(verify_token)):
    """
    Update user's lastPing timestamp
    """
    try:
        if req.uid != decoded_token["uid"]:
            raise HTTPException(status_code=403, detail="UID mismatch")

        user_ref = db.collection("users").document(req.uid)
        user_ref.update({
            "lastPing": datetime.utcnow(),
            "isOnline": True
        })
        return {"success": True, "message": "Ping successful"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/signout")
async def signout(req: PingRequest, decoded_token=Depends(verify_token)):
    """
    Mark user offline in Firestore
    """
    try:
        if req.uid != decoded_token["uid"]:
            raise HTTPException(status_code=403, detail="UID mismatch")

        user_ref = db.collection("users").document(req.uid)
        user_ref.update({
            "isOnline": False,
            "lastPing": datetime.utcnow()
        })
        return {"success": True, "message": "Signed out successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------
# Run server
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("auth_service:app", host="127.0.0.1", port=8004, reload=True)