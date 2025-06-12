import queue
import sounddevice as sd
import sys
import json
import os
from vosk import Model, KaldiRecognizer

# Configuration
MODEL_PATH = "model"
SAMPLE_RATE = 16000
CHUNK_SIZE = 4000

def initialize_model():
    """Initialize Vosk model with error checking."""
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at '{MODEL_PATH}'")
        print("Please download a model from https://alphacephei.com/vosk/models")
        sys.exit(1)
    
    try:
        model = Model(MODEL_PATH)
        recognizer = KaldiRecognizer(model, SAMPLE_RATE)
        print(f"✅ Model loaded successfully from '{MODEL_PATH}'")
        return model, recognizer
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

def list_audio_devices():
    """List available audio input devices."""
    print("\n🎙️ Available audio devices:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"  {i}: {device['name']} ({device['max_input_channels']} channels)")

def callback(indata, frames, time, status):
    """Audio callback function."""
    if status:
        print(f"⚠️ Audio Status: {status}", file=sys.stderr)
    audio_queue.put(bytes(indata))

def test_microphone(device_index):
    import numpy as np
    print("🎤 Testing mic volume...")
    def callback(indata, frames, time, status):
        volume = np.linalg.norm(indata) * 10
        print("📈 Mic volume level:", int(volume))
    with sd.InputStream(callback=callback, device=device_index, channels=1, samplerate=16000):
        sd.sleep(5000)

def recognize_stream():
    """Main speech recognition loop."""
    model, recognizer = initialize_model()
    
    global audio_queue
    audio_queue = queue.Queue()
    
    # Check available devices
    try:
        sd.check_input_settings(samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    except Exception as e:
        print(f"❌ Audio settings error: {e}")
        list_audio_devices()
        return
    
    print(f"🎤 Starting live speech recognition...")
    print(f"📊 Sample Rate: {SAMPLE_RATE} Hz, Chunk Size: {CHUNK_SIZE}")
    print("💡 Speak into your microphone (Ctrl+C to stop)")
    print("-" * 50)
    
    try:
        with sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=CHUNK_SIZE,
        dtype='int16',
        channels=1,
        device=15,  # ✅ Use the working device index
        callback=callback
        ):
            while True:
                data = audio_queue.get()
                
                if recognizer.AcceptWaveform(data):
                    # Final result
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "").strip()
                    if text:
                        print(f"✅ Final: {text}")
                else:
                    # Partial result
                    partial = json.loads(recognizer.PartialResult())
                    partial_text = partial.get("partial", "").strip()
                    if partial_text:
                        print(f"🔄 Partial: {partial_text}", end="\r")
                        
    except KeyboardInterrupt:
        print("\n🛑 Speech recognition stopped.")
    except Exception as e:
        print(f"\n❌ Error during recognition: {e}")
    finally:
        # Get any remaining results
        final_result = json.loads(recognizer.FinalResult())
        final_text = final_result.get("text", "").strip()
        if final_text:
            print(f"✅ Final: {final_text}")

if __name__ == "__main__":
    test_microphone(1)
    recognize_stream()