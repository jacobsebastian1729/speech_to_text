# Enhanced Speech-to-text using microphone input with noise reduction and post-processing
import queue
import sounddevice as sd
import sys
import json
import os
import re
import threading
import time
from collections import deque
from vosk import Model, KaldiRecognizer
import numpy as np

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "model"))
SAMPLE_RATE = 16000
CHUNK_SIZE = 4000
NOISE_GATE_THRESHOLD = 0.01  # Minimum volume to process audio
SILENCE_TIMEOUT = 2.0  # Seconds of silence before finalizing text

class SpeechRecognizer:
    def __init__(self):
        self.model = None
        self.recognizer = None
        self.audio_queue = queue.Queue()
        self.text_buffer = deque(maxlen=50)  # Store recent transcriptions
        self.last_audio_time = time.time()
        self.is_recording = False
        
    def initialize_model(self):
        """Initialize Vosk model with error checking."""
        if not os.path.exists(MODEL_PATH):
            print(f"Error: Model not found at '{MODEL_PATH}'")
            print("Please download a model from https://alphacephei.com/vosk/models")
            sys.exit(1)
        
        try:
            self.model = Model(MODEL_PATH)
            self.recognizer = KaldiRecognizer(self.model, SAMPLE_RATE)
            # Enable word-level timestamps and confidence scores
            self.recognizer.SetWords(True)
            print(f"Model loaded successfully from '{MODEL_PATH}'")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def noise_gate(self, audio_data):
        """Apply noise gate to filter out low-level noise."""
        audio_float = audio_data.astype(np.float32) / 32768.0
        volume = np.sqrt(np.mean(audio_float ** 2))
        return volume > NOISE_GATE_THRESHOLD

    def preprocess_audio(self, audio_data):
        """Apply audio preprocessing including noise reduction."""
        # Convert to float
        audio_float = audio_data.astype(np.float32) / 32768.0
        
        # Simple high-pass filter to remove low-frequency noise
        if len(audio_float) > 1:
            audio_float = np.diff(audio_float, prepend=audio_float[0])
        
        # Normalize audio
        if np.max(np.abs(audio_float)) > 0:
            audio_float = audio_float / np.max(np.abs(audio_float)) * 0.8
        
        # Convert back to int16
        return (audio_float * 32767).astype(np.int16)

    def clean_text(self, text):
        """Clean and post-process recognized text."""
        if not text:
            return ""
        
        # Remove repeated characters (e.g., "speakak" -> "speak")
        text = re.sub(r'(.)\1{2,}', r'\1', text)
        
        # Fix common repetition patterns
        words = text.split()
        cleaned_words = []
        
        for i, word in enumerate(words):
            # Skip if this word is very similar to the previous word
            if i > 0 and self.are_similar_words(word, words[i-1]):
                continue
            cleaned_words.append(word)
        
        # Join and clean up spacing
        cleaned_text = ' '.join(cleaned_words)
        
        # Remove extra spaces and normalize punctuation
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text

    def are_similar_words(self, word1, word2, threshold=0.7):
        """Check if two words are similar (for deduplication)."""
        if not word1 or not word2:
            return False
        
        # Exact match
        if word1 == word2:
            return True
        
        # Check if one word is contained in another
        if word1 in word2 or word2 in word1:
            return True
        
        # Simple similarity check based on common characters
        common_chars = set(word1) & set(word2)
        similarity = len(common_chars) / max(len(set(word1)), len(set(word2)))
        
        return similarity > threshold

    def audio_callback(self, indata, frames, time_info, status):
        """Enhanced audio callback with noise gating."""
        if status:
            print(f"Audio Status: {status}", file=sys.stderr)
        
        # Convert to numpy array
        audio_data = np.frombuffer(indata, dtype=np.int16)
        
        # Apply noise gate
        if self.noise_gate(audio_data):
            # Preprocess audio
            processed_audio = self.preprocess_audio(audio_data)
            self.audio_queue.put(bytes(processed_audio))
            self.last_audio_time = time.time()
            if not self.is_recording:
                self.is_recording = True
                print("🔴 Recording...")
        else:
            # Check for silence timeout
            if self.is_recording and (time.time() - self.last_audio_time) > SILENCE_TIMEOUT:
                self.is_recording = False
                print("⏸️  Silence detected, processing...")

    def get_confidence_score(self, result_dict):
        """Extract confidence score from Vosk result."""
        if 'result' in result_dict:
            words = result_dict['result']
            if words:
                confidences = [word.get('conf', 0) for word in words]
                return sum(confidences) / len(confidences) if confidences else 0
        return 0

    def list_audio_devices(self):
        """List available audio input devices."""
        print("\nAvailable audio devices:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"  {i}: {device['name']} ({device['max_input_channels']} channels)")

    def test_microphone(self, device_index=None):
        """Test microphone levels."""
        print("Testing mic volume for 5 seconds...")
        
        def volume_callback(indata, frames, time_info, status):
            volume = np.linalg.norm(indata) * 10
            bars = '█' * int(volume)
            print(f"Volume: {int(volume):3d} |{bars:<20}|", end='\r')
        
        try:
            with sd.InputStream(
                callback=volume_callback,
                device=device_index,
                channels=1,
                samplerate=SAMPLE_RATE,
                dtype='int16'
            ):
                sd.sleep(5000)
        except Exception as e:
            print(f"Error testing microphone: {e}")
        print("\nMicrophone test complete.")

    def recognize_stream(self, device_index=15):
        """Main speech recognition loop with improvements."""
        if not self.initialize_model():
            return
        
        print(f"🎤 Starting enhanced speech recognition...")
        print(f"📊 Sample Rate: {SAMPLE_RATE} Hz, Chunk Size: {CHUNK_SIZE}")
        print(f"🔇 Noise gate threshold: {NOISE_GATE_THRESHOLD}")
        print("💡 Speak into your microphone (Ctrl+C to stop)")
        print("-" * 60)
        
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                blocksize=CHUNK_SIZE,
                dtype='int16',
                channels=1,
                device=device_index,
                callback=self.audio_callback
            ):
                sentence_buffer = ""
                last_partial = ""
                
                while True:
                    try:
                        data = self.audio_queue.get(timeout=0.1)
                        
                        if self.recognizer.AcceptWaveform(data):
                            # Final result
                            result = json.loads(self.recognizer.Result())
                            text = result.get("text", "").strip()
                            
                            if text:
                                # Clean the text
                                cleaned_text = self.clean_text(text)
                                
                                if cleaned_text:
                                    # Get confidence score
                                    confidence = self.get_confidence_score(result)
                                    
                                    # Add to sentence buffer
                                    sentence_buffer += " " + cleaned_text
                                    
                                    # Print with confidence indicator
                                    conf_indicator = "✅" if confidence > 0.7 else "⚠️" if confidence > 0.4 else "❌"
                                    print(f"Final {conf_indicator}: {cleaned_text} (conf: {confidence:.2f})")
                                    
                                    # Store in buffer for context
                                    self.text_buffer.append(cleaned_text)
                        else:
                            # Partial result
                            partial = json.loads(self.recognizer.PartialResult())
                            partial_text = partial.get("partial", "").strip()
                            
                            if partial_text and partial_text != last_partial:
                                cleaned_partial = self.clean_text(partial_text)
                                if cleaned_partial:
                                    print(f"Partial: {cleaned_partial}", end="                    \r")
                                    last_partial = partial_text
                                    
                    except queue.Empty:
                        # Check for sentence completion during silence
                        if sentence_buffer and not self.is_recording:
                            print(f"\n📝 Complete sentence: {sentence_buffer.strip()}")
                            print("-" * 60)
                            sentence_buffer = ""
                        continue
                        
        except KeyboardInterrupt:
            print("\n🛑 Speech recognition stopped.")
        except Exception as e:
            print(f"\n❌ Error during recognition: {e}")
        finally:
            # Get any remaining results
            try:
                final_result = json.loads(self.recognizer.FinalResult())
                final_text = final_result.get("text", "").strip()
                if final_text:
                    cleaned_final = self.clean_text(final_text)
                    if cleaned_final:
                        print(f"Final: {cleaned_final}")
            except:
                pass

def run_microphone_input():
    """Main function to run the enhanced speech recognizer."""
    recognizer = SpeechRecognizer()
    
    # Uncomment to list available devices
    # recognizer.list_audio_devices()
    
    # Uncomment to test microphone
    # recognizer.test_microphone(device_index=15)
    
    # Start recognition
    recognizer.recognize_stream(device_index=15)
