import sounddevice as sd
import numpy as np

def test_all_input_devices():
    print("🎤 Testing all input devices for activity (speak into each)...\n")

    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        if dev['max_input_channels'] < 1:
            continue

        print(f"🔎 Testing device {idx}: {dev['name']}")
        try:
            with sd.InputStream(device=idx, channels=1, samplerate=16000) as stream:
                sd.sleep(1500)  # wait a moment
                data, _ = stream.read(1024)
                volume = np.linalg.norm(data) * 10
                print(f"📈 Volume level: {int(volume)}\n")
        except Exception as e:
            print(f"⚠️ Skipped: {e}\n")

test_all_input_devices()


