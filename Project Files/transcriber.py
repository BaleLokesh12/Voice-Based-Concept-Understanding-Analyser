import os
import speech_recognition as sr
from openai import OpenAI

def transcribe_audio(audio_path, api_key=None, backend="google"):
    """
    Transcribes the audio file at audio_path.
    Supported backends:
      - "google": Google Web Speech API (free, online, no API key required)
      - "openai": OpenAI Whisper API (requires OpenAI API key)
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
    if backend == "openai":
        if not api_key:
            raise ValueError("OpenAI API key is required for Whisper API transcription.")
        return _transcribe_whisper_api(audio_path, api_key)
    elif backend == "google":
        return _transcribe_google(audio_path)
    else:
        raise ValueError(f"Unknown transcription backend: {backend}. Use 'google' or 'openai'.")

def _transcribe_google(audio_path):
    """
    Transcribes audio using SpeechRecognition's Google Web Speech API.
    Does not require local ffmpeg for WAV files.
    """
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            # Adjust for ambient noise and record
            audio_data = recognizer.record(source)
            
        # Call Google API
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        # Audio was not understood
        return ""
    except sr.RequestError as e:
        raise ConnectionError(f"Could not request results from Google Speech Recognition service; {e}")
    except Exception as e:
        raise RuntimeError(f"Error during Google transcription: {e}")

def _transcribe_whisper_api(audio_path, api_key):
    """
    Transcribes audio using OpenAI's hosted Whisper API (whisper-1).
    """
    try:
        client = OpenAI(api_key=api_key)
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return transcript.text
    except Exception as e:
        raise RuntimeError(f"OpenAI Whisper API error: {e}")
