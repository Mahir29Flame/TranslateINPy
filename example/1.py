import sys
sys.path.append("src")
from translateinpy import Translator
import os

api_key = os.getenv("GEMINI_API_KEY")

Translator("../sample/goru.wav", "en", api_key)