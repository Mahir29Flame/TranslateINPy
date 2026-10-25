import sys
sys.path.append("src")
from translateinpy import Translator
import os

api_key = os.getenv("GEMINI_API_KEY")

Translator("../sample/hi-2.mp3", "en", api_key)