# TranslateINPy
A python translation module for translating live responses using Gemini API inside your application.\
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/translateinpy?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=BLUE&left_text=Downloads)](https://pepy.tech/projects/translateinpy)

## Usage

### As a Module:

```python
from translateinpy import Translator
import os

api_key = os.getenv("GEMINI_API_KEY")

Translator("../sample/hi-1.wav", "en", api_key)
```

### As a CLI Tool:

```bash
translateinpy --url  "./sample/Hi-1.wav" --target bn --api-key "YOUR_API_KEY"
```

## Installation

### As a Module:

```bash
pip install TranslateINPY
```

or via UV:

```bash
uv add TranslateINPy
```

### As a CLI Tool:

```bash
pipx install TranslateINPY
```

or via UV:

```bash
uv tool install TranslateINPY
```
