# speech_to_text/ai_receptionist/llm/llm.py

import os
from dotenv import load_dotenv
import google.generativeai as genai
from ai_receptionist.audio.tts import speak

# Load API key from .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ API key not found. Please check your .env file.")
    exit(1)

# Configure the generative AI with the API key
genai.configure(api_key=api_key)

# Load Gemini model
model_name_to_use = "gemini-1.5-flash"
model = genai.GenerativeModel(model_name=model_name_to_use)

def polish_text(text):
    """Send input text to Gemini and return polished version."""
    prompt = f"Polish this sentence: {text}"
    try:
        response = model.generate_content(prompt)
        polished = response.text.strip()
        speak(polished)
        return polished
    except Exception as e:
        error_msg = f"[Gemini Error]: {e}"
        speak("Sorry, there was an error processing your request.")  # ✅ Speak error
        return error_msg

# Test code (run this file directly to test)
if __name__ == "__main__":
    test_input = input("Enter a sentence to polish: ")
    polished = polish_text(test_input)
    print("\n🔍 Gemini Response:")
    print(polished)
