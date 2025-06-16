# ai_receptionist/llm/llm.py

import os
from dotenv import load_dotenv
import google.generativeai as genai
from ai_receptionist.audio.tts import speak

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ GEMINI_API_KEY not found in .env.")
    exit(1)

genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# Helper fallback phrases Gemini might generate when unsure
fallback_phrases = [
    "i'm sorry", "could you", "please clarify", "i didn't understand",
    "too short", "unclear", "rephrase"
]

def polish_text(user_input: str) -> str:
    user_input = user_input.strip()

    if not user_input:
        response_text = "I didn't catch that. Could you say it again?"
        speak(response_text)
        return response_text

    prompt = f"""
You are a friendly assistant. Respond helpfully and clearly to the user input.
If the input is a greeting, acknowledge it politely.
If the input is unclear, gently ask for clarification.
Otherwise, polish the sentence and make it clearer.

User said: "{user_input}"
Response:
"""

    try:
        response = model.generate_content(prompt)
        reply = response.text.strip()

        # Detect if Gemini is giving fallback text instead of real polish
        if any(p in reply.lower() for p in fallback_phrases):
            reply = "I'm here to help — could you say that again more clearly?"

        speak(reply)
        return reply

    except Exception as e:
        error_msg = "There was an error processing your request."
        print(f"[Gemini Error]: {e}")
        speak(error_msg)
        return error_msg

# Manual test
if __name__ == "__main__":
    while True:
        user_input = input("🎙️ Say something: ")
        response = polish_text(user_input)
        print("🤖:", response)
