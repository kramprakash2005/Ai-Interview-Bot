import os
import time
import random 
import requests 
import json 
from datetime import datetime, timedelta
from typing import List, Optional
from email.message import EmailMessage
from email.mime.multipart import MIMEMultipart 
from email.mime.text import MIMEText 
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Depends, Body, BackgroundTasks, Request
from fastapi.responses import RedirectResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from passlib.hash import argon2
from jose import JWTError, jwt
from pymongo import MongoClient

# ---------------- CONFIG ----------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "ai_interview_bot")
JWT_SECRET = os.getenv("JWT_SECRET", "change_this_secret_in_prod")
JWT_ALGO = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60 * 24))

SMTP_HOST = os.getenv("SMTP_HOST", "localhost")
SMTP_PORT = int(os.getenv("SMTP_PORT", 1025))
SMTP_USER = os.getenv("SMTP_USER", "") or None
SMTP_PASS = os.getenv("SMTP_PASS", "") or None
FROM_EMAIL = os.getenv("FROM_EMAIL", SMTP_USER or "no-reply@aiinterview.local")

OTP_EXPIRE_MINUTES = 5 
FRONTEND_PARTICIPANT_URL = os.getenv("FRONTEND_PARTICIPANT_URL", "http://localhost:5500/frontend/participant_interview.html") 

# Gemini Config (Real Use)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent" 
# ----------------------------------------

# NEW: Security check for JWT secret
if JWT_SECRET == "change_this_secret_in_prod":
    raise Exception("SECURITY ERROR: Please change JWT_SECRET in your .env file before running the server.")

app = FastAPI(title="AI Interview Bot Backend (dev)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "null"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Connect to MongoDB (synchronous pymongo for simplicity)
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]

# DEBUG print: confirm config at startup
print("STARTUP: SMTP CONFIG:", SMTP_HOST, SMTP_PORT, "USER:", SMTP_USER, "FROM:", FROM_EMAIL)
print("STARTUP: FRONTEND_PARTICIPANT_URL:", FRONTEND_PARTICIPANT_URL)
print("STARTUP: MONGO_URI:", MONGO_URI)
print("STARTUP: DB_NAME:", DB_NAME)

# ---------------- Pydantic models ----------------
class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str
    role: str = Field(..., regex="^(interviewer|participant|admin)$")
    organization: Optional[str] = None 

class TokenResp(BaseModel):
    access_token: str
    token_type: str = "bearer"

class SessionCreate(BaseModel):
    title: str
    description: Optional[str] = ""
    time_limit_minutes: Optional[int] = 60
    end_date: Optional[datetime] = None
    config: Optional[dict] = {}

class QuestionCreate(BaseModel):
    text: str
    time_limit_seconds: Optional[int] = 120
    weight: Optional[int] = 1

class BulkInvite(BaseModel):
    emails: List[EmailStr]

class ParticipantStartResponse(BaseModel):
    participant_id: str
    token: str
    session_id: str
    questions: List[dict]
    time_limit_minutes: int 

class AnswerPayload(BaseModel):
    question_id: str
    transcript: str
    duration_seconds: Optional[int] = None
    confidence: Optional[float] = None

class OTPRequest(BaseModel):
    email: EmailStr
    password: str

class OTPVerify(BaseModel):
    email: EmailStr
    otp: str

class ParticipantProfileUpdate(BaseModel):
    name: str
    phone_number: Optional[str] = None
    other_details: Optional[str] = None

# NEW: Evaluation Models (for mock and Gemini structure)
class SkillScore(BaseModel):
    skill: str
    score: int
    rationale: str

class EvaluationSummary(BaseModel):
    overall_score: int = Field(..., ge=1, le=100)
    summary: str
    skill_scores: List[SkillScore]
    red_flags: List[str]
    follow_up_questions: List[str]

class GeminiSchema(BaseModel):
    type: str = "OBJECT"
    properties: dict = {
        "overall_score": {"type": "INTEGER", "description": "Final score out of 100."},
        "summary": {"type": "STRING", "description": "Concise summary and hiring recommendation."},
        "skill_scores": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "skill": {"type": "STRING"},
                    "score": {"type": "INTEGER", "description": "Score from 1 to 10."},
                    "rationale": {"type": "STRING"}
                }
            }
        },
        "red_flags": {"type": "ARRAY", "items": {"type": "STRING"}},
        "follow_up_questions": {"type": "ARRAY", "items": {"type": "STRING"}}
    }


# ---------------- Utilities ----------------
def hash_password(password: str) -> str:
    return argon2.hash(password)

def verify_password(password: str, hash_: str) -> bool:
    try:
        return argon2.verify(password, hash_)
    except Exception as e:
        print("Password verify error:", e)
        return False

def create_access_token(subject: str, expires_delta: Optional[timedelta] = None):
    now = datetime.utcnow()
    expire = now + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    payload = {"sub": subject, "exp": int(expire.timestamp()), "iat": int(now.timestamp())}
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)
    return token

def decode_token(token: str):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        return payload
    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")

def get_current_user(request: Request):
    """Robustly extracts and validates the JWT token from the Authorization header."""
    auth = request.headers.get("Authorization")
    if not auth:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    
    try:
        scheme, token = auth.split()
        if scheme.lower() != 'bearer':
             raise HTTPException(status_code=401, detail="Authentication scheme must be Bearer")

        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        user_id = payload.get("sub")
        
        if not user_id:
             raise HTTPException(status_code=401, detail="Invalid token structure")

        user = db.users.find_one({"_id": user_id})
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except JWTError as e:
        print(f"JWT Decoding Error: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid Authorization header format. Must be 'Bearer <token>'")
    except Exception as e:
        print(f"Auth Error: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed unexpectedly")

def generate_otp() -> str:
    """Generates a secure 6-digit OTP."""
    return "".join([str(random.randint(0, 9)) for _ in range(6)])

def _get_participant_counts(session_id: str) -> dict:
    """Helper to count participants by status for a given session."""
    pipeline = [
        {"$match": {"session_id": session_id}},
        {"$group": {"_id": "$status", "count": {"$sum": 1}}}
    ]
    results = list(db.participants.aggregate(pipeline))
    
    counts = defaultdict(int)
    for res in results:
        counts[res["_id"]] = res["count"]
        
    total = sum(counts.values())
    
    return {
        "invited": counts["invited"],
        "started": counts["started"],
        "completed": counts["completed"],
        "total": total
    }

# ---------------- Robust email sender (OTP) ----------------
def _send_otp_email_sync(to_email: str, otp: str):
    import smtplib
    try:
        host = os.getenv("SMTP_HOST", SMTP_HOST)
        port = int(os.getenv("SMTP_PORT", SMTP_PORT))
        user = os.getenv("SMTP_USER", SMTP_USER) or None
        pwd  = os.getenv("SMTP_PASS", SMTP_PASS) or None
        fr   = os.getenv("FROM_EMAIL", FROM_EMAIL)

        msg = MIMEMultipart('alternative')
        msg["From"] = fr
        msg["To"] = to_email
        msg["Subject"] = "Your One-Time Password (OTP) for Login" 

        # --- Plaintext Part ---
        text = f"Your login verification code (OTP) is: {otp}. This code is valid for {OTP_EXPIRE_MINUTES} minutes."
        
        # --- HTML Part (Enhanced and attractive) ---
        html = f"""\
<html>
  <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; margin: 0; padding: 0; background-color: #f4f4f9;">
    <div style="max-width: 600px; margin: 20px auto; background-color: #ffffff; border-radius: 12px; overflow: hidden; box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);">
      
      <div style="background-color: #6366f1; color: white; padding: 30px; text-align: center;">
        <h1 style="margin: 0; font-size: 28px; font-weight: 700;">AI Interview Platform</h1>
        <p style="margin: 5px 0 0; font-size: 14px;">Secure Login Verification</p>
      </div>
      
      <div style="padding: 30px; text-align: center;">
        <p style="font-size: 16px; margin-bottom: 25px;">
            To complete your sign-in, please use the verification code below:
        </p>
        
        <!-- OTP Code Box -->
        <div style="display: inline-block; background-color: #eef2ff; border: 2px dashed #a5b4fc; padding: 20px 40px; margin-bottom: 30px; border-radius: 8px;">
            <p style="font-size: 36px; font-weight: bold; color: #1e293b; letter-spacing: 10px; margin: 0;">
                {otp}
            </p>
        </div>
        
        <p style="font-size: 15px; color: #ef4444; font-weight: bold; margin-bottom: 30px;">
            This code is valid for the next <strong>{OTP_EXPIRE_MINUTES} minutes</strong>.
        </p>
        
        <p style="font-size: 14px; color: #6b7280;">
            Please return to the login screen and enter the code to gain access.
        </p>
      </div>
      
      <div style="background-color: #f7f7f7; padding: 15px; text-align: center; border-top: 1px solid #e0e0e0;">
        <p style="margin: 0; font-size: 12px; color: #9ca3af;">If you did not request this, please ignore this email.</p>
      </div>
    </div>
  </body>
</html>
"""
        msg.attach(MIMEText(text, 'plain'))
        msg.attach(MIMEText(html, 'html'))
        
        # SMTP connection logic
        with smtplib.SMTP(host, port, timeout=15) as smtp:
            smtp.ehlo()
            try:
                smtp.starttls()
                smtp.ehlo()
            except Exception as e:
                pass
            
            if user and pwd:
                try:
                    smtp.login(user, pwd)
                except Exception as e:
                    raise
            
            smtp.send_message(msg)

    except Exception as e:
        print("Failed to send OTP email:", type(e).__name__, e)

def _send_invite_email_sync(to_email: str, link: str, session_title: str, session_description: str, organization_name: str):
    """
    Sends a rich HTML invitation email with session and organization details.
    """
    import smtplib
    try:
        host = os.getenv("SMTP_HOST", SMTP_HOST)
        port = int(os.getenv("SMTP_PORT", SMTP_PORT))
        user = os.getenv("SMTP_USER", SMTP_USER) or None
        pwd  = os.getenv("SMTP_PASS", SMTP_PASS) or None
        fr   = os.getenv("FROM_EMAIL", FROM_EMAIL)

        org_display = f"from {organization_name}" if organization_name else ""
        
        msg = MIMEMultipart('alternative')
        msg["From"] = fr
        msg["To"] = to_email
        msg["Subject"] = f"Invitation: AI Interview for {session_title} {org_display}" 

        # --- Plaintext Part ---
        text = f"""\
You have been invited to participate in an AI Interview for the position: {session_title} {org_display}.

Description: {session_description}

Please start your interview by clicking the link below or copying it into your browser:

{link}

(This is an automated invite.)
"""
        
        # --- HTML Part (Enhanced and professional) ---
        html = f"""\
<html>
  <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; margin: 0; padding: 0;">
    <div style="max-width: 600px; margin: 20px auto; background-color: #f7f7f7; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);">
      
      <div style="background-color: #6366f1; color: white; padding: 25px; text-align: center; border-top-left-radius: 12px; border-top-right-radius: 12px;">
        <h1 style="margin: 0; font-size: 24px;">AI Interview Invitation</h1>
      </div>
      
      <div style="padding: 30px;">
        <p style="font-size: 16px; margin-bottom: 20px;">Hello,</p>
        
        <p style="font-size: 16px; margin-bottom: 25px;">
            You have been selected to participate in a structured, AI-driven interview session for the role of:
        </p>
        
        <!-- Job Details Card -->
        <div style="background-color: #eef2ff; border-left: 5px solid #6366f1; padding: 15px 20px; margin-bottom: 30px; border-radius: 8px;">
            <p style="font-size: 18px; font-weight: bold; color: #1e293b; margin: 0;">
                <span style="color: #6366f1;">Position:</span> {session_title}
            </p>
            {"<p style='font-size: 14px; color: #4b5563; margin-top: 5px; margin-bottom: 0;'>Organization: " + organization_name + "</p>" if organization_name else ""}
            <p style="font-size: 14px; color: #4b5563; margin-top: 10px; margin-bottom: 0;">
                <span style="font-weight: bold;">Description:</span> {session_description or 'No description provided.'}
            </p>
        </div>
        
        <p style="font-size: 16px; margin-bottom: 30px;">
            Please ensure you have a stable internet connection and a working microphone/headset. Click below to start:
        </p>
        
        <!-- Call to Action Button -->
        <p style="text-align:center; margin-bottom: 30px;">
          <a href="{link}" style="background-color: #ec4899; color: white; padding: 15px 30px; text-align: center; text-decoration: none; display: inline-block; border-radius: 10px; font-weight: bold; font-size: 16px; box-shadow: 0 4px 8px rgba(236, 72, 153, 0.4);">
            Start Interview Now
          </a>
        </p>
        
        <p style="text-align: center; font-size: 12px; color: #6b7280; margin-top: 20px;">
            If the button doesn't work, copy and paste this link into your browser: <br>
            <code style="word-break: break-all;">{link}</code>
        </p>
      </div>
      
      <div style="background-color: #e5e7eb; padding: 15px; text-align: center; border-bottom-left-radius: 12px; border-bottom-right-radius: 12px;">
        <p style="margin: 0; font-size: 12px; color: #6b7280;">This is an automated message. Do not reply.</p>
      </div>
    </div>
  </body>
</html>
"""
        msg.attach(MIMEText(text, 'plain'))
        msg.attach(MIMEText(html, 'html'))
        
        # SMTP connection logic
        with smtplib.SMTP(host, port, timeout=15) as smtp:
            smtp.ehlo()
            try:
                smtp.starttls()
                smtp.ehlo()
            except Exception as e:
                pass
            
            if user and pwd:
                try:
                    smtp.login(user, pwd)
                except Exception as e:
                    raise
            
            smtp.send_message(msg)

    except Exception as e:
        db.email_failures.insert_one({"to": to_email, "link": link, "error": str(e), "when": datetime.utcnow()})


# ---------------- Auth endpoints (OTP Integrated) ----------------
@app.post("/auth/register", response_model=dict)
def register(u: UserCreate):
    existing = db.users.find_one({"email": u.email.lower()})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user_doc = {
        "_id": str(time.time_ns()),
        "name": u.name,
        "email": u.email.lower(),
        "password_hash": hash_password(u.password),
        "role": u.role,
        "organization": u.organization,
        "created_at": datetime.utcnow()
    }
    db.users.insert_one(user_doc)
    return {"msg": "registered"}

@app.post("/auth/send_otp", response_model=dict)
def send_otp_for_login(req: OTPRequest, background_tasks: BackgroundTasks):
    user = db.users.find_one({"email": req.email.lower()})
    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    otp_code = generate_otp()
    otp_hash = hash_password(otp_code)
    otp_expiry = datetime.utcnow() + timedelta(minutes=OTP_EXPIRE_MINUTES)
    
    db.users.update_one(
        {"_id": user["_id"]}, 
        {"$set": {"otp_hash": otp_hash, "otp_expiry": otp_expiry}}
    )
    
    background_tasks.add_task(_send_otp_email_sync, req.email.lower(), otp_code)
    
    return {"msg": f"OTP sent to {req.email.lower()}. Valid for {OTP_EXPIRE_MINUTES} minutes."}

@app.post("/auth/verify_otp", response_model=TokenResp)
def verify_otp_and_login(req: OTPVerify):
    user = db.users.find_one({"email": req.email.lower()})
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    otp_hash = user.get("otp_hash")
    otp_expiry = user.get("otp_expiry")
    
    if not otp_hash or not otp_expiry or otp_expiry < datetime.utcnow():
        raise HTTPException(status_code=401, detail="OTP invalid or expired. Please request a new code.")
        
    if not verify_password(req.otp, otp_hash):
        raise HTTPException(status_code=401, detail="Invalid OTP provided.")
        
    db.users.update_one(
        {"_id": user["_id"]}, 
        {"$unset": {"otp_hash": "", "otp_expiry": ""}} 
    )
    
    token = create_access_token(user["_id"])
    return {"access_token": token}


# ---------------- Sessions & Questions ----------------
@app.post("/api/sessions", response_model=dict)
def create_session(s: SessionCreate, current=Depends(get_current_user)):
    if current["role"] not in ("interviewer", "admin"):
        raise HTTPException(status_code=403, detail="Not allowed")
    doc = {
        "_id": str(time.time_ns()),
        "owner_id": current["_id"],
        "title": s.title,
        "description": s.description,
        "time_limit_minutes": s.time_limit_minutes,
        "end_date": s.end_date,
        "config": s.config or {},
        "created_at": datetime.utcnow(),
        "status": "open",
    }
    db.sessions.insert_one(doc)
    return {"session_id": doc["_id"]}

@app.get("/api/sessions")
def list_sessions(current=Depends(get_current_user)):
    # Only list sessions owned by the current user
    cursor = db.sessions.find({"owner_id": current["_id"]}) 
    items = list(cursor)
    
    # NEW: Fetch metrics for all sessions
    for item in items:
        item['metrics'] = _get_participant_counts(item['_id'])
    
    return items

@app.get("/api/sessions/{session_id}/metrics")
def get_session_metrics(session_id: str, current=Depends(get_current_user)):
    sess = db.sessions.find_one({"_id": session_id})
    if not sess or sess["owner_id"] != current["_id"]:
        raise HTTPException(403, "Not allowed")
    
    return _get_participant_counts(session_id)

@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str, current=Depends(get_current_user)):
    sess = db.sessions.find_one({"_id": session_id})
    if not sess or sess["owner_id"] != current["_id"]:
        raise HTTPException(403, "Session not found or not authorized to delete")

    # 1. Delete session and related data (Cascading Deletes)
    db.sessions.delete_one({"_id": session_id})
    
    # Get all related participant IDs
    participant_ids = [p["_id"] for p in db.participants.find({"session_id": session_id}, {"_id": 1})]

    # Delete dependent data
    db.questions.delete_many({"session_id": session_id})
    db.participants.delete_many({"session_id": session_id})
    db.answers.delete_many({"participant_id": {"$in": participant_ids}})
    db.proctor_logs.delete_many({"participant_id": {"$in": participant_ids}})

    return {"msg": "Session and all related data deleted"}

@app.get("/api/sessions/{session_id}/info")
def get_session_info_public(session_id: str):
    sess = db.sessions.find_one({"_id": session_id})
    if not sess:
        raise HTTPException(404, "Session not found")
    
    return {
        "title": sess["title"],
        "description": sess.get("description"),
        "time_limit_minutes": sess.get("time_limit_minutes")
    }


@app.post("/api/sessions/{session_id}/questions")
def add_question(session_id: str, q: QuestionCreate, current=Depends(get_current_user)):
    sess = db.sessions.find_one({"_id": session_id})
    if not sess or sess["owner_id"] != current["_id"]:
        raise HTTPException(status_code=403, detail="Session not found or not owner")
    doc = {
        "_id": str(time.time_ns()),
        "session_id": session_id,
        "order": int(time.time_ns()) % 100000,
        "text": q.text,
        "time_limit_seconds": q.time_limit_seconds,
        "weight": q.weight
    }
    db.questions.insert_one(doc)
    return {"question_id": doc["_id"]}

@app.get("/api/sessions/{session_id}/questions")
def get_questions(session_id: str, current=Depends(get_current_user)):
    cursor = db.questions.find({"session_id": session_id})
    out = list(cursor)
    return out

# ---------------- Invites (bulk) ----------------
@app.post("/api/sessions/{session_id}/invite")
def bulk_invite(session_id: str, invites: BulkInvite, background_tasks: BackgroundTasks, current=Depends(get_current_user)):
    sess = db.sessions.find_one({"_id": session_id})
    if not sess or sess["owner_id"] != current["_id"]:
        raise HTTPException(status_code=403, detail="Session not found or not owner")
    
    organization_name = current.get("organization", "")
    
    created = []
    skipped = [] 
    
    for em in invites.emails:
        em_lower = em.lower()
        
        existing_p = db.participants.find_one({"session_id": session_id, "email": em_lower})
        if existing_p:
            skipped.append(em)
            continue
            
        pid = str(time.time_ns())
        token = create_access_token(pid, expires_delta=timedelta(days=7))
        pdoc = {
            "_id": pid,
            "session_id": session_id,
            "email": em_lower,
            "name": None,
            "token": token,
            "status": "invited",
            "invited_at": datetime.utcnow()
        }
        db.participants.insert_one(pdoc)

        if FRONTEND_PARTICIPANT_URL:
            link = f"{FRONTEND_PARTICIPANT_URL}?token={token}&participant_id={pid}&session_id={session_id}"
        else:
            link = f"http://localhost:8000/participant_start?token={token}&participant_id={pid}&session_id={session_id}"

        background_tasks.add_task(
            _send_invite_email_sync, 
            em, 
            link, 
            sess["title"], 
            sess.get("description", "No description provided."),
            organization_name
        )
        created.append({"email": em, "participant_id": pid})
        
    return {"invited": created, "skipped": skipped}

# ---------------- Participant Profile Update ----------------
@app.post("/api/participants/{participant_id}/profile")
def update_participant_profile(participant_id: str, profile: ParticipantProfileUpdate):
    result = db.participants.update_one(
        {"_id": participant_id}, 
        {"$set": {
            "name": profile.name, 
            "phone_number": profile.phone_number, 
            "other_details": profile.other_details,
            "profile_updated_at": datetime.utcnow()
        }}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Participant not found")
        
    return {"msg": "profile updated"}

# ---------------- Gemini Evaluation (Real API Call) ----------------
def call_gemini_evaluation(prompt: str, schema: dict):
    """
    Executes the call to the Gemini API with structured output requirements.
    """
    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY is missing. AI evaluation cannot proceed.")
        raise HTTPException(status_code=503, detail="AI Evaluation Service Unavailable: GEMINI_API_KEY not configured in backend.")
    
    headers = {"Content-Type": "application/json"}
    api_url = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": schema
        }
    }
    
    print("--- Sending Prompt to Gemini ---")
    
    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status() 
        
        result = response.json()
        
        if (result.get('candidates') and result['candidates'][0].get('content') and 
            result['candidates'][0]['content'].get('parts')):
            
            json_text = result['candidates'][0]['content']['parts'][0]['text']
            return json.loads(json_text)
        else:
            print(f"Gemini Response Structure Invalid: {result}")
            raise HTTPException(status_code=500, detail="AI returned invalid structured output.")

    except requests.exceptions.RequestException as e:
        print(f"Gemini API Request Error: {e}")
        raise HTTPException(status_code=500, detail=f"AI Evaluation Service Failed: {e}")
    except Exception as e:
        print(f"Evaluation Processing Error: {e}")
        raise HTTPException(status_code=500, detail=f"AI Evaluation Processing Failed.")


def _perform_evaluation_sync(participant_id: str):
    """Performs the full evaluation process synchronously (intended for BackgroundTasks)."""
    
    p = db.participants.find_one({"_id": participant_id})
    if not p: return

    sess = db.sessions.find_one({"_id": p['session_id']})
    if not sess: return
    
    # Check if already evaluated to prevent redundant processing
    if p.get("is_evaluated"): return

    # Fetch user data to get organization name for context
    user = db.users.find_one({"_id": sess['owner_id']})
    
    answers = list(db.answers.find({"participant_id": participant_id}))
    questions = list(db.questions.find({"session_id": p['session_id']}).sort("order"))
    
    if not answers:
        # Cannot evaluate without answers
        db.participants.update_one({"_id": participant_id}, {"$set": {"evaluation_status": "NO_ANSWERS"}})
        return
        
    q_map = {q['_id']: q['text'] for q in questions}
    
    # 1. Build Comprehensive Context Prompt
    prompt_context = f"""
    You are a Senior Technical Interviewer hired by {user.get('organization', 'Organization X') if user else 'Organization X'}.
    Your task is to evaluate a candidate based on their responses to an interview session.
    
    --- CONTEXT ---
    Job Role: {sess.get('title', 'N/A')}
    Job Description: {sess.get('description', 'N/A')}
    Participant: {p.get('name') or p['email']}
    
    --- CANDIDATE RESPONSES ---
    """
        
    for i, answer in enumerate(answers):
        q_text = q_map.get(answer['question_id'], f"Unknown Question {i+1}")
        prompt_context += f"""
        Q{i+1}. Question: {q_text}
        A{i+1}. Transcript: {answer['transcript']}
        ---
        """

    system_instruction = """
    Analyze the candidate's responses against the Job Role and Job Description.
    Provide an objective, structured evaluation in the required JSON format.
    Focus on clarity, technical accuracy, and relevance.
    """
    
    try:
        # 2. Get Structured Evaluation (Real API Call)
        evaluation_result = call_gemini_evaluation(
            prompt=system_instruction + prompt_context,
            schema=GeminiSchema().dict(by_alias=True) 
        )

        # 3. Save Evaluation Summary
        db.participants.update_one(
            {"_id": participant_id},
            {"$set": {
                "evaluation_summary": evaluation_result,
                "is_evaluated": True,
                "evaluated_at": datetime.utcnow()
            }}
        )
        print(f"Evaluation complete for participant {participant_id}")

    except HTTPException as e:
        # Log failure reason to DB for debugging
        db.participants.update_one(
            {"_id": participant_id},
            {"$set": {"evaluation_status": f"FAILED: {e.detail}", "is_evaluated": False}}
        )
        print(f"Evaluation failed for participant {participant_id}: {e.detail}")
    except Exception as e:
        db.participants.update_one(
            {"_id": participant_id},
            {"$set": {"evaluation_status": f"CRITICAL_FAIL: {str(e)}", "is_evaluated": False}}
        )

@app.post("/api/participants/{participant_id}/evaluate", response_model=EvaluationSummary)
def evaluate_participant(participant_id: str, current=Depends(get_current_user)):
    """
    Endpoint removed from dashboard flow, but kept for direct debugging/re-run capability if needed.
    """
    p = db.participants.find_one({"_id": participant_id})
    if not p:
        raise HTTPException(404, "Participant not found")
        
    if p.get("evaluation_summary"):
        return EvaluationSummary(**p["evaluation_summary"])

    # Trigger synchronous evaluation for direct API calls
    _perform_evaluation_sync(participant_id)
    p_updated = db.participants.find_one({"_id": participant_id})
    
    if p_updated.get("evaluation_summary"):
        return EvaluationSummary(**p_updated["evaluation_summary"])
    elif p_updated.get("evaluation_status"):
        raise HTTPException(400, detail=f"Evaluation failed: {p_updated['evaluation_status']}")
    else:
        raise HTTPException(500, detail="Evaluation is running asynchronously or failed to start.")


# ---------------- Participant start & answer submission ----------------
@app.post("/api/sessions/{session_id}/start", response_model=ParticipantStartResponse)
def participant_start(session_id: str, token: str = Body(..., embed=True)):
    payload = decode_token(token)
    participant_id = payload.get("sub")
    participant = db.participants.find_one({"_id": participant_id, "session_id": session_id})
    if not participant:
        raise HTTPException(404, "Participant not found")
        
    if participant.get("status") == "completed":
        raise HTTPException(400, detail="Interview already completed.")
        
    session_data = db.sessions.find_one({"_id": session_id})
    if not session_data:
        raise HTTPException(404, "Session not found")
        
    q_cursor = db.questions.find({"session_id": session_id})
    questions = [{"id": q["_id"], "text": q["text"], "time_limit_seconds": q.get("time_limit_seconds", 120)} for q in q_cursor]
    
    return {
        "participant_id": participant_id, 
        "token": token, 
        "session_id": session_id, 
        "questions": questions,
        "time_limit_minutes": session_data.get("time_limit_minutes", 60) 
    }

@app.post("/api/participants/{participant_id}/answer")
def submit_answer(participant_id: str, payload: AnswerPayload):
    participant = db.participants.find_one({"_id": participant_id})
    if not participant:
        raise HTTPException(status_code=404, detail="Participant not found")
    
    if participant.get("is_evaluated"):
        raise HTTPException(400, "Cannot submit answers; evaluation has already been finalized.")
        
    # --- FIX: Use Update/Upsert instead of Insert to replace existing answers ---
    
    # 1. Define the unique filter (Participant + Question)
    filter_query = {
        "participant_id": participant_id,
        "question_id": payload.question_id
    }

    # 2. Define the document to set/replace
    update_fields = {
        # Using $set to update fields and ensuring creation of an _id only if document is new
        "$set": {
            "transcript": payload.transcript,
            "duration_seconds": payload.duration_seconds,
            "confidence": payload.confidence,
            "created_at": datetime.utcnow()
        },
        # Ensure the unique ID is set if this is the first time the answer is saved (upsert)
        "$setOnInsert": {
            "_id": str(time.time_ns()) # Use a unique ID generator for the document itself
        }
    }

    # 3. Perform the upsert operation
    db.answers.update_one(
        filter_query,
        update_fields,
        upsert=True
    )
    
    # We do not return the document ID as it might be an update, but we confirm success.
    return {"msg": "Answer saved/updated"}
    # --- END FIX ---


@app.post("/api/participants/{participant_id}/complete")
def complete_interview(participant_id: str, background_tasks: BackgroundTasks):
    """
    Endpoint to mark the interview as completed and trigger asynchronous evaluation.
    """
    update_fields = {"completed_at": datetime.utcnow(), "status": "completed"}
    
    result = db.participants.update_one(
        {"_id": participant_id}, 
        {"$set": update_fields}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Participant not found")
        
    # Trigger AI Evaluation as a background task
    background_tasks.add_task(_perform_evaluation_sync, participant_id)

    return {"msg": "completed, evaluation started"}

# ---------------- Proctor logs ----------------
@app.post("/api/proctor/logs")
def proctor_logs(body: dict = Body(...)):
    participant_id = body.get("participant_id")
    logs = body.get("logs", [])
    items = []
    for l in logs:
        doc = {
            "_id": str(time.time_ns()) + "_" + l.get("type", "UNKNOWN"),
            "participant_id": participant_id,
            "question_id": l.get("question_id"), 
            "type": l.get("type"),
            "detail": l.get("detail"),
            "timestamp": l.get("timestamp") or datetime.utcnow(),
            "severity": l.get("severity", "info")
        }
        items.append(doc)
    if items:
        db.proctor_logs.insert_many(items)
    return {"ingested": len(items)}

# ---------------- Results & review ----------------
@app.get("/api/sessions/{session_id}/results")
def session_results(session_id: str, current=Depends(get_current_user)):
    sess = db.sessions.find_one({"_id": session_id})
    if not sess:
        raise HTTPException(404, "Session not found")
    if current["role"] not in ("admin",) and sess["owner_id"] != current["_id"]:
        raise HTTPException(403, "Not allowed")
    
    participants = []
    for p in db.participants.find({"session_id": session_id}):
        # FIX: When fetching answers, we automatically get only the unique/latest answer 
        # because the upsert ensures only one document per (participant_id, question_id) pair exists.
        answers = list(db.answers.find({"participant_id": p["_id"]}))
        logs = list(db.proctor_logs.find({"participant_id": p["_id"]}))
        
        # Calculate violation metrics for the report
        violation_counts = defaultdict(int)
        for log in logs:
            violation_counts[log.get('type', 'UNKNOWN')] += 1
        
        participants.append({
            "participant": p, 
            "answers": answers, 
            "proctor_logs": logs,
            "violation_metrics": violation_counts
        })
    return {"session": sess, "participants": participants}

@app.get("/api/users")
def list_users(current=Depends(get_current_user)):
    if current["role"] != "admin":
        raise HTTPException(403, "Not allowed")
    out = []
    for u in db.users.find({}):
        out.append({"_id": u["_id"], "email": u["email"], "name": u.get("name"), "role": u.get("role")})
    return out

# ---------------- Root ----------------
@app.get("/")
def root():
    return {"ok": True, "time": datetime.utcnow().isoformat()}

# ---------------- Launch Server ----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)