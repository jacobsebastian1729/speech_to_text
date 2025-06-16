# Text-to-speech conversion logic
# ai_receptionist/audio/tts.py

import pyttsx3
import time

# Initialize the engine globally so it loads only once
engine = pyttsx3.init()

#voices = engine.getProperty('voices')
#for voice in voices:
#    print(f"Name: {voice.name}, ID: {voice.id}")


#want a male voce comment the below two lines that inialize the 'voice'
# For Windows (typically supports Microsoft Zira)
engine.setProperty('voice', 'HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_ZIRA_11.0')
#want another voice uncomment the below line and comment the aove line
#engine.setProperty('voice', 'HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-GB_HAZEL_11.0')



# Optional: adjust speech rate and volume
engine.setProperty('rate', 130)    # Default ~200
engine.setProperty('volume', 1)  # Max volume

#engine.say("Initializing voice.")  # Hidden pre-buffer
engine.runAndWait()

def speak(text):
    """Speak the given text aloud."""
    if not text:
        return
    #time.sleep(1)
    engine.say("hmmm ")           # Dummy warm-up
    engine.runAndWait()
    time.sleep(0.1) 
    engine.say(text)
    engine.runAndWait()

#testing code(run this file directly to test)
if __name__ == "__main__":
    #test_input = input("enter a sentence to speak: ")
    test_input = "hello! how are you, how can i help you?"
    #test_input = "                     " + test_input
    speak(test_input)
    print("done speech")

