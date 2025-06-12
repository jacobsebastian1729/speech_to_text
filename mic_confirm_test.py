import sounddevice as sd
import numpy as np

def callback(indata, frames, time, status):
    volume_norm = np.linalg.norm(indata) * 10
    print("📈 Mic volume level:", int(volume_norm))

print("🎤 Testing mic input on device 0...")
with sd.InputStream(callback=callback, device=0, channels=1, samplerate=16000):
    sd.sleep(5000)
