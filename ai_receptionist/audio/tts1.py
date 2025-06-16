import asyncio
import numpy as np
import sounddevice as sd
from edge_tts import Communicate
from pydub import AudioSegment
import io

async def speak_async(text: str, voice: str = "en-US-JennyNeural"):
    communicate = Communicate(text=text, voice=voice)
    stream = await communicate.stream()
    audio_data = b""

    async for chunk in stream:
        if chunk["type"] == "audio":
            audio_data += chunk["data"]

    # Decode MP3 in memory and convert to NumPy array
    audio = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    samples /= 2 ** 15  # Normalize 16-bit audio

    # Play audio using sounddevice
    sd.play(samples, samplerate=audio.frame_rate)
    sd.wait()

def speak(text: str):
    if text.strip():
        asyncio.run(speak_async(text))

if __name__ == "__main__":
    speak("Hello! How can I help you today?")
