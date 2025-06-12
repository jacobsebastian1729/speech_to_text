import sounddevice as sd

def test_device(device_index):
    try:
        print(f"\n🔍 Testing device index: {device_index} — {sd.query_devices(device_index)['name']}")
        def callback(indata, frames, time, status):
            print("🎙️ Capturing audio...")

        with sd.InputStream(callback=callback, device=device_index):
            sd.sleep(3000)
        return True
    except Exception as e:
        print(f"❌ Device {device_index} failed: {e}")
        return False

print("🎤 Starting microphone input test...\n")

for i, device in enumerate(sd.query_devices()):
    if device['max_input_channels'] > 0:
        if test_device(i):
            print(f"✅ Working device found: {i} — {device['name']}")
            break
