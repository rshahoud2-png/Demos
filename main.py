"""
Abe's Barbershop AI Receptionist Demo
Voice + SMS powered by Twilio and OpenAI
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Form, Request
from fastapi.responses import Response
import httpx
from dotenv import load_dotenv
from twilio.twiml.voice_response import VoiceResponse, Gather
from twilio.twiml.messaging_response import MessagingResponse

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000")

# Ensure logs directory exists
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Abe's Barbershop AI Receptionist",
    description="Voice + SMS AI receptionist demo using Twilio and OpenAI",
    version="1.0.0"
)

# ============================================================================
# BUSINESS INFORMATION - SINGLE SOURCE OF TRUTH
# ============================================================================
BUSINESS_INFO = """
Business Name: Abe's Barbershop
Address: 927 3rd St, Whitehall, PA 18052
Phone: (610) 533-2246

Hours:
- Monday: Closed
- Tuesday: 10:00 AM - 6:00 PM
- Wednesday: 10:00 AM - 6:00 PM
- Thursday: 10:00 AM - 7:00 PM
- Friday: 9:00 AM - 6:00 PM
- Saturday: 9:00 AM - 3:00 PM
- Sunday: Closed

Services Offered:
- Men's Haircut
- Beard Trimming
- Men's Shaving (hot towel shave)
- Head Shave

Booking: We use Square for booking appointments.
"""

SYSTEM_PROMPT = f"""You are a friendly AI receptionist for Abe's Barbershop. Be warm, professional, and concise.

{BUSINESS_INFO}

STRICT RULES:
1. ONLY provide information from the business info above. Never invent prices, policies, or details not listed.
2. If asked about pricing, say "Prices vary depending on the service. I'd be happy to take your information and have someone call you back with pricing details."
3. For appointment requests, collect: name, requested service, preferred day/time, and phone number. Confirm all details before logging.
4. Keep responses SHORT and conversational (1-3 sentences max for voice).
5. If you don't know something, say so honestly and offer to have someone call them back.
6. Always be helpful and friendly, representing Abe's Barbershop professionally.
"""

# In-memory conversation state (keyed by phone number)
conversation_state: dict = {}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def log_conversation(
    channel: str,
    from_number: str,
    user_input: str,
    ai_reply: str,
    call_sid: Optional[str] = None,
    message_sid: Optional[str] = None
):
    """Log conversation to JSONL file."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "channel": channel,
        "from": from_number,
        "transcript" if channel == "voice" else "body": user_input,
        "ai_reply": ai_reply,
    }

    if call_sid:
        log_entry["call_sid"] = call_sid
    if message_sid:
        log_entry["message_sid"] = message_sid

    log_file = LOGS_DIR / "conversations.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

    logger.info(f"Logged {channel} conversation from {from_number}")


def log_appointment(
    name: str,
    service: str,
    preferred_time: str,
    phone: str,
    channel: str
):
    """Log appointment request to text file."""
    log_file = LOGS_DIR / "appointments.txt"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    entry = f"""
================================================================================
Appointment Request - {timestamp}
Channel: {channel}
--------------------------------------------------------------------------------
Name: {name}
Service: {service}
Preferred Day/Time: {preferred_time}
Phone: {phone}
================================================================================
"""

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(entry)

    logger.info(f"Logged appointment request from {name}")


async def get_openai_response(
    messages: list,
    max_tokens: int = 150
) -> str:
    """Get response from OpenAI API."""
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not set")
        return "I'm sorry, I'm having technical difficulties. Please call back later."

    try:
        # Explicit timeouts: 10s connect, 25s read (total ~30s max)
        timeout = httpx.Timeout(connect=10.0, read=25.0, write=10.0, pool=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": 0.7
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return "I'm sorry, I'm having technical difficulties. Please call back later."


def get_or_create_conversation(phone: str, channel: str) -> dict:
    """Get or create conversation state for a phone number."""
    if phone not in conversation_state:
        conversation_state[phone] = {
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}],
            "turn_count": 0,
            "channel": channel,
            "appointment_data": {}
        }
    return conversation_state[phone]


def check_for_goodbye(text: str) -> bool:
    """Check if the user is saying goodbye."""
    goodbye_phrases = [
        "goodbye", "bye", "good bye", "thank you bye",
        "thanks bye", "that's all", "thats all", "nothing else",
        "no thank you", "no thanks", "i'm good", "im good"
    ]
    text_lower = text.lower().strip()
    return any(phrase in text_lower for phrase in goodbye_phrases)


def extract_appointment_info(ai_response: str, user_input: str) -> Optional[dict]:
    """
    Simple heuristic to detect if an appointment was confirmed.
    In production, you'd use function calling or structured output.
    """
    # This is a simplified detection - in production use OpenAI function calling
    confirmation_phrases = [
        "i've got you down", "appointment request", "i have your appointment",
        "booking request", "scheduled", "i'll log that"
    ]

    if any(phrase in ai_response.lower() for phrase in confirmation_phrases):
        return {"detected": True, "response": ai_response}
    return None


# ============================================================================
# VOICE ENDPOINTS
# ============================================================================

@app.post("/voice")
async def voice_entry():
    """
    Initial voice endpoint - greets caller and starts gathering speech.
    """
    response = VoiceResponse()

    gather = Gather(
        input="speech",
        action="/voice/handle",
        method="POST",
        timeout=5,
        speech_timeout="auto"
    )
    gather.say(
        "Thanks for calling Abe's Barbershop! How can I help you today?",
        voice="Polly.Matthew"
    )
    response.append(gather)

    # If no input, prompt again
    response.say("I didn't catch that. Please call back if you need assistance. Goodbye!")
    response.hangup()

    return Response(content=str(response), media_type="application/xml")


@app.post("/voice/handle")
async def voice_handle(
    SpeechResult: str = Form(default=""),
    From: str = Form(default=""),
    CallSid: str = Form(default="")
):
    """
    Handle speech input and generate AI response.
    """
    response = VoiceResponse()

    # Get conversation state
    conv = get_or_create_conversation(From, "voice")
    conv["turn_count"] += 1

    user_input = SpeechResult.strip()
    logger.info(f"Voice input from {From}: {user_input}")

    # Check for goodbye or max turns
    if check_for_goodbye(user_input) or conv["turn_count"] >= 6:
        # Generate farewell
        conv["messages"].append({"role": "user", "content": user_input})
        conv["messages"].append({
            "role": "user",
            "content": "The caller is ending the call. Say a brief, friendly goodbye."
        })

        ai_reply = await get_openai_response(conv["messages"], max_tokens=50)

        # Log conversation
        log_conversation("voice", From, user_input, ai_reply, call_sid=CallSid)

        response.say(ai_reply, voice="Polly.Matthew")
        response.say("Thanks for calling Abe's Barbershop. Have a great day!", voice="Polly.Matthew")
        response.hangup()

        # Clean up conversation state
        if From in conversation_state:
            del conversation_state[From]

        return Response(content=str(response), media_type="application/xml")

    # Add user message and get AI response
    conv["messages"].append({"role": "user", "content": user_input})
    ai_reply = await get_openai_response(conv["messages"])
    conv["messages"].append({"role": "assistant", "content": ai_reply})

    # Log conversation
    log_conversation("voice", From, user_input, ai_reply, call_sid=CallSid)

    # Check if appointment was confirmed and log it
    appointment_info = extract_appointment_info(ai_reply, user_input)
    if appointment_info:
        # In a real app, you'd parse structured data from OpenAI
        log_appointment(
            name="[Collected via voice]",
            service="[See conversation log]",
            preferred_time="[See conversation log]",
            phone=From,
            channel="voice"
        )

    # Continue conversation
    gather = Gather(
        input="speech",
        action="/voice/handle",
        method="POST",
        timeout=5,
        speech_timeout="auto"
    )
    gather.say(ai_reply, voice="Polly.Matthew")
    response.append(gather)

    # If no input, end call
    response.say("I didn't hear anything. If you need more help, please call back. Goodbye!")
    response.hangup()

    return Response(content=str(response), media_type="application/xml")


# ============================================================================
# SMS ENDPOINTS
# ============================================================================

@app.post("/sms")
async def sms_handle(
    Body: str = Form(default=""),
    From: str = Form(default=""),
    MessageSid: str = Form(default="")
):
    """
    Handle incoming SMS messages.
    """
    response = MessagingResponse()

    user_input = Body.strip()
    logger.info(f"SMS from {From}: {user_input}")

    # Get conversation state
    conv = get_or_create_conversation(From, "sms")
    conv["turn_count"] += 1

    # Add context for SMS (can be slightly longer than voice)
    conv["messages"].append({"role": "user", "content": user_input})

    # Get AI response (allow slightly longer for SMS)
    ai_reply = await get_openai_response(conv["messages"], max_tokens=250)
    conv["messages"].append({"role": "assistant", "content": ai_reply})

    # Log conversation
    log_conversation("sms", From, user_input, ai_reply, message_sid=MessageSid)

    # Check if appointment was confirmed
    appointment_info = extract_appointment_info(ai_reply, user_input)
    if appointment_info:
        log_appointment(
            name="[Collected via SMS]",
            service="[See conversation log]",
            preferred_time="[See conversation log]",
            phone=From,
            channel="sms"
        )

    response.message(ai_reply)

    return Response(content=str(response), media_type="application/xml")


# ============================================================================
# HEALTH CHECK & INFO ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check and info endpoint."""
    return {
        "status": "running",
        "service": "Abe's Barbershop AI Receptionist",
        "version": "1.0.0",
        "endpoints": {
            "voice": "/voice",
            "voice_handler": "/voice/handle",
            "sms": "/sms"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint for Fly.io."""
    return {"status": "ok"}


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8080))
    # Note: reload=True should only be used in local development, not in production
    # Fly.io runs the container directly, so this block is mainly for local testing
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("ENV", "production") == "development"
    )
