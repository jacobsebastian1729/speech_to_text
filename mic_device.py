import sounddevice as sd

print("\n🎙️ Input devices supporting 16000 Hz mono:")
for i, dev in enumerate(sd.query_devices()):
    if dev['max_input_channels'] > 0:
        try:
            sd.check_input_settings(device=i, samplerate=16000, channels=1, dtype='int16')
            print(f"{i}: {dev['name']} ✅")
        except Exception as e:
            print(f"{i}: {dev['name']} ❌ ({e})")
