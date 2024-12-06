# Universal Video Summarizer

---

**Universal Video Summarizer** is a python based QT5 desktop application that automates summarizing videos. The application utilizes `yt-dlp` to download videos, then ASR to transcribe it and LLMs to summarize/process the transcript according to user directives. 

## Installation

---

### Prerequisites
- Python 3.8+
- `ffmpeg` installed on your system.

### Install Dependencies 
```bash
pip install -r requirements.txt
```

## Usage

---

### Running the Application
```bash
python main.py
```

### Using the Application
1. Enter the URL or local path of the video you want to summarize.
2. Select the desired summarization method.
3. Click the `Summarize` button to start the summarization process.
4. The application will display the summarized text in the text box.

### Selecting Which Model to Use
1. Click the `Settings` button.
2. Select one of many LLM models available on [Hugging Face](https://huggingface.co/models).
3. Click the `Save` button to save the settings.

### Recommended models:
#### ASR Models
| Model Name | Notes                                  |
|----|----------------------------------------|
| `openai/whisper-large-v3` | Best performance.                      |
| `distil-whisper/distil-large-v3` | Reduced VRAM usage, comparable performance. |

#### LLM Models
| Model Name | Notes |
|------------|-------|
| `meta-llama/Llama-3.2-1B-Instruct` | Requires signing up and requesting access. |
| `meta-llama/Llama-3.2-3B-Instruct` | Larger version of the 1B, requires more VRAM to run fast. Requires signing up and requesting access. |
| `Qwen/Qwen2-1.5B-Instruct` | Has issues with hallucinations. |


## Features

---

- Summarize videos from YouTube, Twitch or any other online platform `yt-dlp` supports and local files.
- Switch between different models for transcription and summarization.
- Create custom prompts for the summarization process.

## Contributions

---

Contributions are welcome! Feel free to fork this repository, submit issues, or create pull requests.