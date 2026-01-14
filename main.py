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

# Fallback system state
FALLBACK_ACTIVE = "fallback_active"
FALLBACK_MODE = "fallback_mode"  # "appointment" or "reschedule"
FALLBACK_STEP = "fallback_step"  # current step in the collection process
FALLBACK_DATA = "fallback_data"  # collected data

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
) -> tuple[str, bool]:
    """
    Get response from OpenAI API.
    Returns: (response_text, success_flag)
    """
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not set")
        return ("I'm sorry, I'm having technical difficulties switching to automated system.", False)

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
            return (data["choices"][0]["message"]["content"].strip(), True)
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return ("OpenAI service unavailable. Switching to automated system.", False)


def get_or_create_conversation(phone: str, channel: str) -> dict:
    """Get or create conversation state for a phone number."""
    if phone not in conversation_state:
        conversation_state[phone] = {
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}],
            "turn_count": 0,
            "channel": channel,
            "appointment_data": {},
            FALLBACK_ACTIVE: False,
            FALLBACK_MODE: None,
            FALLBACK_STEP: 0,
            FALLBACK_DATA: {}
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


def detect_keyword(text: str) -> Optional[str]:
    """
    Detect appointment-related keywords in user input.
    Returns: "appointment" or "reschedule" or None
    """
    text_lower = text.lower().strip()

    appointment_keywords = ["appointment", "book", "schedule", "make an appointment"]
    reschedule_keywords = ["reschedule", "change appointment", "move appointment", "change my appointment"]

    # Check reschedule first (more specific)
    if any(keyword in text_lower for keyword in reschedule_keywords):
        return "reschedule"

    if any(keyword in text_lower for keyword in appointment_keywords):
        return "appointment"

    return None


def initialize_fallback(conv: dict, mode: str):
    """Initialize fallback mode for appointment or reschedule."""
    conv[FALLBACK_ACTIVE] = True
    conv[FALLBACK_MODE] = mode
    conv[FALLBACK_STEP] = 0
    conv[FALLBACK_DATA] = {}
    logger.info(f"Fallback mode activated: {mode}")


def get_fallback_prompt(conv: dict) -> str:
    """
    Get the prompt for the current fallback step.
    Returns the question to ask the user.
    """
    mode = conv[FALLBACK_MODE]
    step = conv[FALLBACK_STEP]

    if mode == "appointment":
        prompts = [
            "I can help you schedule an appointment. May I have your full name please?",
            "Great! What service would you like? We offer men's haircut, beard trimming, hot towel shave, or head shave.",
            "Perfect! What date and time would you prefer for your appointment?",
            "And what's the best phone number to reach you at?"
        ]
    elif mode == "reschedule":
        prompts = [
            "I can help you reschedule your appointment. What is your name?",
            "Thank you! What date and time would you like to reschedule to?",
            "And what's your phone number for confirmation?"
        ]
    else:
        return "I'm sorry, I'm having trouble understanding. How can I help you?"

    if step < len(prompts):
        return prompts[step]
    else:
        return "Thank you for providing all the information!"


def process_fallback_input(conv: dict, user_input: str) -> tuple[str, bool]:
    """
    Process user input in fallback mode.
    Returns: (response_text, is_complete)
    """
    mode = conv[FALLBACK_MODE]
    step = conv[FALLBACK_STEP]

    # Handle "waiting" mode - user needs to specify appointment or reschedule
    if mode == "waiting":
        keyword = detect_keyword(user_input)
        if keyword in ["appointment", "reschedule"]:
            initialize_fallback(conv, keyword)
            return (get_fallback_prompt(conv), False)
        else:
            return ("I'm sorry, I didn't catch that. Please say 'appointment' to schedule a new appointment, or 'reschedule' to change an existing appointment.", False)

    # Store the user's response
    if mode == "appointment":
        if step == 0:  # Name
            conv[FALLBACK_DATA]["name"] = user_input
            conv[FALLBACK_STEP] += 1
            return (get_fallback_prompt(conv), False)
        elif step == 1:  # Service
            conv[FALLBACK_DATA]["service"] = user_input
            conv[FALLBACK_STEP] += 1
            return (get_fallback_prompt(conv), False)
        elif step == 2:  # Date/Time
            conv[FALLBACK_DATA]["datetime"] = user_input
            conv[FALLBACK_STEP] += 1
            return (get_fallback_prompt(conv), False)
        elif step == 3:  # Phone
            conv[FALLBACK_DATA]["phone"] = user_input
            # Complete - log the appointment
            log_appointment(
                name=conv[FALLBACK_DATA].get("name", "Unknown"),
                service=conv[FALLBACK_DATA].get("service", "Not specified"),
                preferred_time=conv[FALLBACK_DATA].get("datetime", "Not specified"),
                phone=conv[FALLBACK_DATA].get("phone", "Not provided"),
                channel=conv["channel"]
            )
            confirmation = f"Thank you, {conv[FALLBACK_DATA].get('name', 'there')}! I've recorded your appointment request for {conv[FALLBACK_DATA].get('service', 'your service')} on {conv[FALLBACK_DATA].get('datetime', 'your preferred date')}. Someone will call you at {conv[FALLBACK_DATA].get('phone', 'your number')} to confirm. Is there anything else I can help you with?"
            return (confirmation, True)

    elif mode == "reschedule":
        if step == 0:  # Name
            conv[FALLBACK_DATA]["name"] = user_input
            conv[FALLBACK_STEP] += 1
            return (get_fallback_prompt(conv), False)
        elif step == 1:  # New Date/Time
            conv[FALLBACK_DATA]["new_datetime"] = user_input
            conv[FALLBACK_STEP] += 1
            return (get_fallback_prompt(conv), False)
        elif step == 2:  # Phone
            conv[FALLBACK_DATA]["phone"] = user_input
            # Complete - log the reschedule request
            log_file = LOGS_DIR / "appointments.txt"
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            entry = f"""
================================================================================
Reschedule Request - {timestamp}
Channel: {conv['channel']}
--------------------------------------------------------------------------------
Name: {conv[FALLBACK_DATA].get('name', 'Unknown')}
New Preferred Date/Time: {conv[FALLBACK_DATA].get('new_datetime', 'Not specified')}
Phone: {conv[FALLBACK_DATA].get('phone', 'Not provided')}
================================================================================
"""
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(entry)

            confirmation = f"Thank you, {conv[FALLBACK_DATA].get('name', 'there')}! I've recorded your request to reschedule to {conv[FALLBACK_DATA].get('new_datetime', 'your preferred time')}. Someone will call you at {conv[FALLBACK_DATA].get('phone', 'your number')} to confirm the change. Is there anything else I can help you with?"
            return (confirmation, True)

    return ("I'm sorry, I didn't understand that. Could you please repeat?", False)


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
    Handle speech input and generate AI response with fallback system.
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
        if not conv[FALLBACK_ACTIVE]:
            conv["messages"].append({"role": "user", "content": user_input})
            conv["messages"].append({
                "role": "user",
                "content": "The caller is ending the call. Say a brief, friendly goodbye."
            })

            ai_reply, success = await get_openai_response(conv["messages"], max_tokens=50)
        else:
            ai_reply = "Thank you for calling!"

        # Log conversation
        log_conversation("voice", From, user_input, ai_reply, call_sid=CallSid)

        response.say(ai_reply, voice="Polly.Matthew")
        response.say("Thanks for calling Abe's Barbershop. Have a great day!", voice="Polly.Matthew")
        response.hangup()

        # Clean up conversation state
        if From in conversation_state:
            del conversation_state[From]

        return Response(content=str(response), media_type="application/xml")

    # ===== FALLBACK SYSTEM LOGIC =====

    # If fallback is already active, process the input
    if conv[FALLBACK_ACTIVE]:
        ai_reply, is_complete = process_fallback_input(conv, user_input)

        # Log conversation
        log_conversation("voice", From, user_input, ai_reply, call_sid=CallSid)

        if is_complete:
            # Reset fallback mode
            conv[FALLBACK_ACTIVE] = False
            conv[FALLBACK_MODE] = None
            conv[FALLBACK_STEP] = 0

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

        response.say("I didn't hear anything. If you need more help, please call back. Goodbye!")
        response.hangup()

        return Response(content=str(response), media_type="application/xml")

    # ===== TRY OPENAI FIRST =====

    # Add user message and get AI response
    conv["messages"].append({"role": "user", "content": user_input})
    ai_reply, openai_success = await get_openai_response(conv["messages"])

    # If OpenAI failed, activate fallback system
    if not openai_success:
        logger.warning(f"OpenAI failed, activating fallback system for {From}")

        # Detect keywords in user input
        keyword = detect_keyword(user_input)

        if keyword:
            # Initialize fallback mode
            initialize_fallback(conv, keyword)
            ai_reply = f"{ai_reply} {get_fallback_prompt(conv)}"
        else:
            # Ask what they need
            ai_reply = "I'm currently in automated mode. Would you like to schedule an appointment or reschedule an existing one? Please say appointment or reschedule."
            initialize_fallback(conv, "waiting")  # Waiting for user to specify

        # Log conversation
        log_conversation("voice", From, user_input, ai_reply, call_sid=CallSid)

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

        response.say("I didn't hear anything. If you need more help, please call back. Goodbye!")
        response.hangup()

        return Response(content=str(response), media_type="application/xml")

    # ===== OPENAI SUCCESS - USE NORMAL FLOW =====

    # Check if user mentioned appointment/reschedule keywords
    keyword = detect_keyword(user_input)
    if keyword:
        # User wants appointment or reschedule - activate fallback to collect data systematically
        initialize_fallback(conv, keyword)
        ai_reply = get_fallback_prompt(conv)
        logger.info(f"User requested {keyword}, switching to structured collection")

    conv["messages"].append({"role": "assistant", "content": ai_reply})

    # Log conversation
    log_conversation("voice", From, user_input, ai_reply, call_sid=CallSid)

    # Check if appointment was confirmed and log it (for AI-collected appointments)
    if not conv[FALLBACK_ACTIVE]:
        appointment_info = extract_appointment_info(ai_reply, user_input)
        if appointment_info:
            log_appointment(
                name="[Collected via AI]",
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
    Handle incoming SMS messages with fallback system support.
    """
    response = MessagingResponse()

    user_input = Body.strip()
    logger.info(f"SMS from {From}: {user_input}")

    # Get conversation state
    conv = get_or_create_conversation(From, "sms")
    conv["turn_count"] += 1

    # ===== FALLBACK SYSTEM LOGIC =====

    # If fallback is already active, process the input
    if conv[FALLBACK_ACTIVE]:
        ai_reply, is_complete = process_fallback_input(conv, user_input)

        # Log conversation
        log_conversation("sms", From, user_input, ai_reply, message_sid=MessageSid)

        if is_complete:
            # Reset fallback mode
            conv[FALLBACK_ACTIVE] = False
            conv[FALLBACK_MODE] = None
            conv[FALLBACK_STEP] = 0

        response.message(ai_reply)
        return Response(content=str(response), media_type="application/xml")

    # ===== TRY OPENAI FIRST =====

    # Add context for SMS (can be slightly longer than voice)
    conv["messages"].append({"role": "user", "content": user_input})

    # Get AI response (allow slightly longer for SMS)
    ai_reply, openai_success = await get_openai_response(conv["messages"], max_tokens=250)

    # If OpenAI failed, activate fallback system
    if not openai_success:
        logger.warning(f"OpenAI failed for SMS, activating fallback system for {From}")

        # Detect keywords in user input
        keyword = detect_keyword(user_input)

        if keyword:
            # Initialize fallback mode
            initialize_fallback(conv, keyword)
            ai_reply = f"{ai_reply}\n\n{get_fallback_prompt(conv)}"
        else:
            # Ask what they need
            ai_reply = "I'm currently in automated mode. Would you like to schedule an appointment or reschedule an existing one? Please reply with 'appointment' or 'reschedule'."
            initialize_fallback(conv, "waiting")

        # Log conversation
        log_conversation("sms", From, user_input, ai_reply, message_sid=MessageSid)

        response.message(ai_reply)
        return Response(content=str(response), media_type="application/xml")

    # ===== OPENAI SUCCESS - USE NORMAL FLOW =====

    # Check if user mentioned appointment/reschedule keywords
    keyword = detect_keyword(user_input)
    if keyword:
        # User wants appointment or reschedule - activate fallback to collect data systematically
        initialize_fallback(conv, keyword)
        ai_reply = get_fallback_prompt(conv)
        logger.info(f"User requested {keyword} via SMS, switching to structured collection")

    conv["messages"].append({"role": "assistant", "content": ai_reply})

    # Log conversation
    log_conversation("sms", From, user_input, ai_reply, message_sid=MessageSid)

    # Check if appointment was confirmed (for AI-collected appointments)
    if not conv[FALLBACK_ACTIVE]:
        appointment_info = extract_appointment_info(ai_reply, user_input)
        if appointment_info:
            log_appointment(
                name="[Collected via AI SMS]",
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
