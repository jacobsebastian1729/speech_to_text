import os

# Define the structure
structure = {
    "ai_receptionist": [
        "audio",
        "ai",
        "core",
        "data",
        "model",
        "utils"
    ],
    "files": {
        "main.py": "",
        "config.py": "# Configuration constants like SAMPLE_RATE, MODEL_PATH, etc.\n",
        "README.md": "# AI Receptionist\n\nAn intelligent speech-based receptionist using STT, LLM, and TTS.",
        "audio/microphone.py": "# Speech-to-text using microphone input\n",
        "audio/tts.py": "# Text-to-speech conversion logic\n",
        "audio/mic_test_utils.py": "# Optional: microphone testing and device utility functions\n",
        "ai/llm.py": "# LLM interaction logic (OpenAI, etc.)\n",
        "ai/prompt.py": "# System prompt templates\n",
        "ai/rag.py": "# RAG (Retrieval-Augmented Generation) logic\n",
        "core/handler.py": "# Handle bookings, cancellations, and other tasks\n",
        "core/memory.py": "# Track conversation history\n",
        "core/storage.py": "# Store/retrieve user/booking data\n",
        "utils/logger.py": "# Optional: log and debug helpers\n",
        "data/.gitkeep": "",  # placeholder to keep the folder
        "model/.gitkeep": "", # you can move your vosk model manually here
    }
}

base = os.getcwd()
root = os.path.join(base, "ai_receptionist")

# Create main folders
for folder in structure["ai_receptionist"]:
    os.makedirs(os.path.join(root, folder), exist_ok=True)

# Create stub files
for path, content in structure["files"].items():
    full_path = os.path.join(root, path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w") as f:
        f.write(content)

print("✅ Project structure created in 'ai_receptionist/' folder.")
