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
from enum import Enum

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
3. Keep responses SHORT and conversational (1-3 sentences max for voice).
4. If you don't know something, say so honestly and offer to have someone call them back.
5. Always be helpful and friendly, representing Abe's Barbershop professionally.

CALL FLOW - Follow this structured conversation flow:
1. IDENTIFY PURPOSE: First, determine if the caller wants to:
   - Make a NEW RESERVATION (book an appointment)
   - MODIFY an existing reservation (change or cancel)
   - Just has a general INQUIRY (questions about services, hours, etc.)

2. COLLECT CLIENT INFO: For reservations or modifications, you MUST collect:
   - Client's NAME (first and last name preferred)
   - Client's PHONE NUMBER (confirm or get it if different from calling number)

3. COLLECT APPOINTMENT DETAILS:
   - For NEW RESERVATION: Get service type, preferred date/time
   - For MODIFICATION: Get existing appointment date/time, and what changes they want

4. CONFIRM: Always summarize and confirm the information before finalizing.

RESPONSE FORMAT:
- Ask ONE question at a time
- Wait for the answer before moving to the next question
- Be conversational but efficient
- When you have collected all required information, confirm it back to the caller
"""

# ============================================================================
# CALL FLOW STATE MACHINE
# ============================================================================
class CallFlowStage(str, Enum):
    """Stages in the voice call flow."""
    GREETING = "greeting"              # Initial greeting, determine call purpose
    IDENTIFY_PURPOSE = "identify_purpose"  # Reservation or modification
    COLLECT_NAME = "collect_name"      # Get client's name
    COLLECT_PHONE = "collect_phone"    # Get/confirm client's phone number
    COLLECT_DETAILS = "collect_details"  # Get service/appointment details
    CONFIRM = "confirm"                # Confirm all information
    COMPLETE = "complete"              # Call completed


class CallPurpose(str, Enum):
    """Purpose of the call."""
    NEW_RESERVATION = "new_reservation"
    MODIFICATION = "modification"
    INQUIRY = "inquiry"
    UNKNOWN = "unknown"


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
            "call_flow_stage": CallFlowStage.GREETING,
            "call_purpose": CallPurpose.UNKNOWN,
            "client_data": {
                "name": None,
                "phone": phone,  # Default to calling number
                "phone_confirmed": False,
                "service": None,
                "preferred_time": None,
                "existing_appointment": None,
                "modification_details": None
            }
        }
    return conversation_state[phone]


def detect_call_purpose(text: str) -> CallPurpose:
    """Detect the purpose of the call from user input."""
    text_lower = text.lower()

    # Check for modification/change/cancel intent
    modification_keywords = [
        "change", "modify", "cancel", "reschedule", "move",
        "different time", "different day", "existing appointment",
        "already have", "booked", "scheduled"
    ]
    if any(keyword in text_lower for keyword in modification_keywords):
        return CallPurpose.MODIFICATION

    # Check for reservation/booking intent
    reservation_keywords = [
        "appointment", "book", "reserve", "schedule", "haircut",
        "beard", "shave", "trim", "cut my hair", "get a cut",
        "make an appointment", "set up", "come in"
    ]
    if any(keyword in text_lower for keyword in reservation_keywords):
        return CallPurpose.NEW_RESERVATION

    # Check for general inquiry
    inquiry_keywords = [
        "hours", "open", "close", "price", "cost", "where",
        "address", "location", "what services", "do you", "how much"
    ]
    if any(keyword in text_lower for keyword in inquiry_keywords):
        return CallPurpose.INQUIRY

    return CallPurpose.UNKNOWN


def extract_name_from_response(text: str) -> Optional[str]:
    """Try to extract a name from user response."""
    text_lower = text.lower().strip()

    # Common patterns: "my name is X", "I'm X", "this is X", "it's X", just "X"
    name_patterns = [
        "my name is ", "name is ", "i'm ", "im ", "i am ",
        "this is ", "it's ", "its ", "call me "
    ]

    for pattern in name_patterns:
        if pattern in text_lower:
            # Extract everything after the pattern
            idx = text_lower.find(pattern) + len(pattern)
            name = text[idx:].strip()
            # Clean up - take first few words (likely name)
            words = name.split()
            if len(words) >= 1:
                # Return up to 3 words (first middle last) or until punctuation
                name_words = []
                for word in words[:3]:
                    clean_word = word.strip(".,!?")
                    if clean_word:
                        name_words.append(clean_word)
                    if word.endswith((".", ",", "!", "?")):
                        break
                if name_words:
                    return " ".join(name_words).title()

    # If no pattern found but input is short (1-3 words), it might be just the name
    words = text.strip().split()
    if 1 <= len(words) <= 3:
        clean_words = [w.strip(".,!?").title() for w in words if w.strip(".,!?")]
        if clean_words:
            return " ".join(clean_words)

    return None


def extract_phone_from_response(text: str) -> Optional[str]:
    """Try to extract a phone number from user response."""
    import re

    # Remove common filler words
    text_clean = text.lower().replace("my number is", "").replace("phone number is", "")
    text_clean = text_clean.replace("it's", "").replace("its", "").replace("is", "")

    # Pattern to match phone numbers (various formats)
    # Matches: 1234567890, 123-456-7890, (123) 456-7890, 123 456 7890, etc.
    phone_pattern = r'[\+]?[(]?[0-9]{1,3}[)]?[-\s\.]?[(]?[0-9]{1,3}[)]?[-\s\.]?[0-9]{3,4}[-\s\.]?[0-9]{4}'

    matches = re.findall(phone_pattern, text_clean)
    if matches:
        # Return the first match, cleaned up
        phone = re.sub(r'[^\d+]', '', matches[0])
        return phone

    # Check for verbal number format (spelled out)
    # This is a simplified version - in production, use a more robust solution
    return None


def update_call_flow_stage(conv: dict, user_input: str) -> str:
    """
    Update the call flow stage based on current stage and user input.
    Returns a context message to add to the AI prompt.
    """
    current_stage = conv.get("call_flow_stage", CallFlowStage.GREETING)
    client_data = conv.get("client_data", {})
    context_msg = ""

    # Stage: GREETING -> IDENTIFY_PURPOSE
    if current_stage == CallFlowStage.GREETING:
        purpose = detect_call_purpose(user_input)
        conv["call_purpose"] = purpose
        conv["call_flow_stage"] = CallFlowStage.COLLECT_NAME

        if purpose == CallPurpose.NEW_RESERVATION:
            context_msg = "[SYSTEM: Caller wants to make a NEW reservation. Ask for their name first.]"
        elif purpose == CallPurpose.MODIFICATION:
            context_msg = "[SYSTEM: Caller wants to MODIFY an existing appointment. Ask for their name first to look up their reservation.]"
        elif purpose == CallPurpose.INQUIRY:
            context_msg = "[SYSTEM: Caller has a general inquiry. Answer their question, then ask if they'd like to make an appointment.]"
            conv["call_flow_stage"] = CallFlowStage.IDENTIFY_PURPOSE
        else:
            context_msg = "[SYSTEM: Unclear call purpose. Ask the caller if they'd like to make a reservation, modify an existing one, or have a question.]"
            conv["call_flow_stage"] = CallFlowStage.IDENTIFY_PURPOSE

    # Stage: IDENTIFY_PURPOSE (for unclear or inquiry cases)
    elif current_stage == CallFlowStage.IDENTIFY_PURPOSE:
        purpose = detect_call_purpose(user_input)
        if purpose in [CallPurpose.NEW_RESERVATION, CallPurpose.MODIFICATION]:
            conv["call_purpose"] = purpose
            conv["call_flow_stage"] = CallFlowStage.COLLECT_NAME
            context_msg = f"[SYSTEM: Caller wants {'a new reservation' if purpose == CallPurpose.NEW_RESERVATION else 'to modify an appointment'}. Ask for their name.]"
        else:
            context_msg = "[SYSTEM: Answer the caller's question. If appropriate, ask if they'd like to make an appointment.]"

    # Stage: COLLECT_NAME
    elif current_stage == CallFlowStage.COLLECT_NAME:
        extracted_name = extract_name_from_response(user_input)
        if extracted_name:
            client_data["name"] = extracted_name
            conv["call_flow_stage"] = CallFlowStage.COLLECT_PHONE
            context_msg = f"[SYSTEM: Client name collected: {extracted_name}. Now confirm their phone number. The calling number is {client_data.get('phone', 'unknown')}. Ask if this is the best number to reach them.]"
        else:
            context_msg = "[SYSTEM: Could not clearly identify the name. Ask politely for their name again.]"

    # Stage: COLLECT_PHONE
    elif current_stage == CallFlowStage.COLLECT_PHONE:
        # Check if they confirmed the existing number or provided a new one
        text_lower = user_input.lower()
        if any(word in text_lower for word in ["yes", "yeah", "yep", "correct", "that's right", "that works", "good", "fine"]):
            client_data["phone_confirmed"] = True
            conv["call_flow_stage"] = CallFlowStage.COLLECT_DETAILS
            context_msg = "[SYSTEM: Phone number confirmed. Now collect appointment details - ask about the service they want and their preferred date/time.]"
        else:
            extracted_phone = extract_phone_from_response(user_input)
            if extracted_phone:
                client_data["phone"] = extracted_phone
                client_data["phone_confirmed"] = True
                conv["call_flow_stage"] = CallFlowStage.COLLECT_DETAILS
                context_msg = f"[SYSTEM: New phone number collected: {extracted_phone}. Now collect appointment details - ask about the service they want and their preferred date/time.]"
            elif any(word in text_lower for word in ["no", "different", "another", "other"]):
                context_msg = "[SYSTEM: Caller wants to provide a different phone number. Ask them for the best phone number to reach them.]"
            else:
                context_msg = "[SYSTEM: Response unclear. Ask to confirm the phone number or provide a better number to reach them.]"

    # Stage: COLLECT_DETAILS
    elif current_stage == CallFlowStage.COLLECT_DETAILS:
        # In this stage, let the AI naturally collect service and time preferences
        # Move to confirm when we have enough info
        purpose = conv.get("call_purpose", CallPurpose.UNKNOWN)
        if purpose == CallPurpose.MODIFICATION:
            context_msg = "[SYSTEM: Collect details about the existing appointment they want to modify and what changes they need.]"
        else:
            context_msg = "[SYSTEM: Collect the service type and preferred date/time. When you have all details, summarize and confirm.]"
        conv["call_flow_stage"] = CallFlowStage.CONFIRM

    # Stage: CONFIRM
    elif current_stage == CallFlowStage.CONFIRM:
        context_msg = f"[SYSTEM: Summarize all collected information - Name: {client_data.get('name')}, Phone: {client_data.get('phone')}, Purpose: {conv.get('call_purpose', 'unknown')}. Confirm the details and let them know what happens next.]"
        conv["call_flow_stage"] = CallFlowStage.COMPLETE

    return context_msg


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
    Initiates the call flow by asking the purpose of the call.
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
        "Thanks for calling Abe's Barbershop! Are you calling to make a new appointment, or to change an existing reservation?",
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
    Handle speech input and generate AI response using the call flow state machine.
    Tracks conversation stages: purpose detection, name collection, phone confirmation, etc.
    """
    response = VoiceResponse()

    # Get conversation state
    conv = get_or_create_conversation(From, "voice")
    conv["turn_count"] += 1

    user_input = SpeechResult.strip()
    logger.info(f"Voice input from {From}: {user_input}")
    logger.info(f"Current call flow stage: {conv.get('call_flow_stage', 'unknown')}")

    # Check for goodbye or max turns
    if check_for_goodbye(user_input) or conv["turn_count"] >= 10:
        # Generate farewell with collected information summary
        client_data = conv.get("client_data", {})
        farewell_context = ""
        if client_data.get("name"):
            farewell_context = f"Client: {client_data.get('name')}, Phone: {client_data.get('phone')}. "

        conv["messages"].append({"role": "user", "content": user_input})
        conv["messages"].append({
            "role": "user",
            "content": f"[SYSTEM: {farewell_context}The caller is ending the call. Thank them by name if known and say a brief, friendly goodbye.]"
        })

        ai_reply = await get_openai_response(conv["messages"], max_tokens=75)

        # Log conversation
        log_conversation("voice", From, user_input, ai_reply, call_sid=CallSid)

        # Log appointment if we collected enough data
        client_data = conv.get("client_data", {})
        if client_data.get("name") and client_data.get("phone"):
            log_appointment(
                name=client_data.get("name", "[Unknown]"),
                service=client_data.get("service", "[See conversation log]"),
                preferred_time=client_data.get("preferred_time", "[See conversation log]"),
                phone=client_data.get("phone", From),
                channel="voice"
            )

        response.say(ai_reply, voice="Polly.Matthew")
        response.say("Thanks for calling Abe's Barbershop. Have a great day!", voice="Polly.Matthew")
        response.hangup()

        # Clean up conversation state
        if From in conversation_state:
            del conversation_state[From]

        return Response(content=str(response), media_type="application/xml")

    # Update call flow stage and get context for AI
    call_flow_context = update_call_flow_stage(conv, user_input)

    # Add user message with call flow context
    conv["messages"].append({"role": "user", "content": user_input})

    # Add call flow context as a system hint
    if call_flow_context:
        conv["messages"].append({"role": "user", "content": call_flow_context})

    # Get AI response
    ai_reply = await get_openai_response(conv["messages"], max_tokens=150)
    conv["messages"].append({"role": "assistant", "content": ai_reply})

    # Log conversation
    log_conversation("voice", From, user_input, ai_reply, call_sid=CallSid)

    # Check if appointment was confirmed and log it
    appointment_info = extract_appointment_info(ai_reply, user_input)
    if appointment_info:
        client_data = conv.get("client_data", {})
        log_appointment(
            name=client_data.get("name", "[Collected via voice]"),
            service=client_data.get("service", "[See conversation log]"),
            preferred_time=client_data.get("preferred_time", "[See conversation log]"),
            phone=client_data.get("phone", From),
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
    Uses the same call flow logic for consistency.
    """
    response = MessagingResponse()

    user_input = Body.strip()
    logger.info(f"SMS from {From}: {user_input}")

    # Get conversation state
    conv = get_or_create_conversation(From, "sms")
    conv["turn_count"] += 1

    # Update call flow stage and get context for AI
    call_flow_context = update_call_flow_stage(conv, user_input)

    # Add user message with call flow context
    conv["messages"].append({"role": "user", "content": user_input})

    # Add call flow context as a system hint
    if call_flow_context:
        conv["messages"].append({"role": "user", "content": call_flow_context})

    # Get AI response (allow slightly longer for SMS)
    ai_reply = await get_openai_response(conv["messages"], max_tokens=250)
    conv["messages"].append({"role": "assistant", "content": ai_reply})

    # Log conversation
    log_conversation("sms", From, user_input, ai_reply, message_sid=MessageSid)

    # Check if appointment was confirmed
    appointment_info = extract_appointment_info(ai_reply, user_input)
    if appointment_info:
        client_data = conv.get("client_data", {})
        log_appointment(
            name=client_data.get("name", "[Collected via SMS]"),
            service=client_data.get("service", "[See conversation log]"),
            preferred_time=client_data.get("preferred_time", "[See conversation log]"),
            phone=client_data.get("phone", From),
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
        "version": "1.1.0",
        "features": {
            "call_flow": "Structured conversation flow with purpose detection",
            "data_collection": "Collects client name and phone number",
            "call_purposes": ["new_reservation", "modification", "inquiry"]
        },
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
