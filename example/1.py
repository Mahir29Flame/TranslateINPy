import sys
sys.path.append("src")
from translateinpy import Translator
import os

api_key = os.getenv("GEMINI_API_KEY")

Translator(r"https://https://storage.googleapis.com/generativeai-downloads/gemini-cookbook/audio/gemini-live-translate-sample.wav", "en", api_key)