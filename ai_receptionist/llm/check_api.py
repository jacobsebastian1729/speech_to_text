import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key from .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ API key not found. Please check your .env file.")
    exit(1)

# Configure the generative AI with the API key
genai.configure(api_key=api_key)

#check for available models
#print("Attempting to list available models...")
#try:
#    for m in genai.list_models():
#        # Check if the model supports text generation
#        if "generateContent" in m.supported_generation_methods:
#            print(f"✅ Available Model: {m.name} (Supports generateContent)")
#        else:
#            print(f"  Available Model: {m.name} (Does NOT support generateContent)")
#except Exception as e:
#    print("❌ Error listing models:", e)
#    exit(1)

# Create a model instance
model_name_to_use = "gemini-1.5-flash" # <-- Recommended
# Or: model_name_to_use = "gemini-1.5-pro" # <-- For higher quality, potentially higher cost/latency

try:
    model = genai.GenerativeModel(model_name=model_name_to_use)
    response = model.generate_content("Tell me a fun fact about AI.")
    print(f"✅ Gemini response using {model_name_to_use}:\n", response.text)
except Exception as e:
    print(f"❌ Error communicating with Gemini API using model '{model_name_to_use}':", e)


