# TranslateINPy
A python translation module for translating live responses using Gemini API inside your application.

## Usage

```python
from translateinpy import Translator
import os

api_key = os.getenv("GEMINI_API_KEY")

Translator("../sample/hi-1.wav", "en", api_key)
```