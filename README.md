# Universal Video Summarizer

---

**Universal Video Summarizer** is a python based QT5 desktop application that automates summarizing videos. The application utilizes `yt-dlp` to download videos, then ASR to transcribe it and LLMs to summarize/process the transcript according to user directives. 

## Installation

---

### Prerequisites
- Python 3.8+
- `pip` installed on your system.
- A CUDA-enabled GPU (Currently required without any modifications).

### Install Dependencies 
```bash
pip install PyQt5 yt-dlp transformers torch
```

## Contributions

---

Contributions are welcome! Feel free to fork this repository, submit issues, or create pull requests.