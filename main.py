from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLineEdit, QLabel, QComboBox, QPushButton, QMessageBox, QTextEdit, QHBoxLayout
)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QThread, pyqtSignal
from tempfile import mkstemp

import sys
import yt_dlp
import time
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


class VideoDownloadWorker(QThread):
    progress_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)

    def __init__(self, url, output_dir="downloads"):
        super().__init__()
        self.url = url
        self.output_dir = output_dir

    def run(self):
        try:
            ydl_opts = {
                'format': 'bestaudio',
                'outtmpl': mkstemp("wav")[1] + ".wav",
                'progress_hooks': [self.hook]
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([self.url])

        except Exception as e:
            self.error_signal.emit(str(e))

    def hook(self, d):
        if d['status'] == 'downloading':
            progress = d.get('_percent_str', '0%').strip()
            self.progress_signal.emit(f"Downloading... {progress}")
        elif d['status'] == 'finished':
            self.finished_signal.emit(d['filename'])

class AudioASRWorker(QThread):
    error_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)
    def __init__(self, filename):
        super().__init__()
        self.filename = filename

    def run(self):
        self.progress_signal.emit(f"Loading ASR Model...")

        # Initialize the ASR pipeline
        pipe = pipeline("automatic-speech-recognition",
                        model="distil-whisper/distil-large-v3",
                        device="cuda:0",
                        torch_dtype=torch.float16)

        self.progress_signal.emit(f"Model Loaded. Transcribing...")

        start_time = time.perf_counter()
        outputs = pipe(self.filename, chunk_length_s=30, batch_size=1, return_timestamps=False)

        self.progress_signal.emit("Transcription Completed. Time Elapsed: " + str(time.perf_counter() - start_time))
        # Output the results
        self.finished_signal.emit(outputs["text"])

        # Cleanup
        del pipe
        torch.cuda.empty_cache()


class LLMWorker(QThread):
    error_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)

    def __init__(self, transcript, summary_type, custom_query=None, model_name="qwen2.5", device="cuda:0"):
        super().__init__()
        self.transcript = transcript
        self.summary_type = summary_type
        self.custom_query = custom_query
        self.model_name = model_name
        self.device = device

    def run(self):
        try:
            self.progress_signal.emit(f"Loading {self.model_name} model for summarization...")

            # Load model based on the specified type
            if self.model_name == "llama3":
                model_name = "meta-llama/Meta-Llama-3.1-7B"
            elif self.model_name == "qwen2.5":
                model_name = "Qwen/Qwen2.5-0.5B"
            else:
                raise ValueError("Unsupported model name")

            tokenizer = None
            model = None
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name, device_map=self.device)
                model = AutoModelForCausalLM.from_pretrained(model_name, device_map=self.device)
            except torch.OutOfMemoryError:
                if tokenizer != None:
                    del tokenizer
                if model != None:
                    del model
                torch.cuda.empty_cache()
                tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="cpu")
                model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")
                self.device = "cpu"

            # Load the transcript and build the prompt
            if self.summary_type == "Brief Summary":
                prompt = f"Provide a brief summary."
            elif self.summary_type == "Detailed Summary":
                prompt = f"Provide a detailed summary."
            elif self.summary_type == "Key Points Only":
                prompt = f"Extract only the key points."
            elif self.summary_type == "Custom" and self.custom_query:
                prompt = f"{self.custom_query}"
            else:
                self.error_signal.emit("Invalid summary type or custom query not provided.")
                return
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful summary assistant that processes transcripts.",
                },
                {"role": "user", "content": prompt + "\n\nTranscript:" + self.transcript },

            ]
            self.progress_signal.emit("Generating summary...")

            start_time = time.perf_counter()

            model_inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.device)
            input_len = model_inputs.shape[1]
            generated_ids = model.generate(model_inputs, do_sample=True, max_new_tokens=input_len)
            outputs = tokenizer.batch_decode(generated_ids[:, input_len:], skip_special_tokens=True)
            summary = outputs[0]

            elapsed_time = time.perf_counter() - start_time
            self.progress_signal.emit(f"Summary generation completed in {elapsed_time:.2f} seconds.")
            self.finished_signal.emit(summary)
            del tokenizer
            del model_inputs
            del generated_ids
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_signal.emit(f"Error during summary generation: {str(e)}")


class VideoSummaryApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Summary Tool")
        self.setGeometry(200, 200, 400, 350)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Title label
        self.title_label = QLabel("Enter Video URL and Select Summary Type")
        self.title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(self.title_label)

        # URL Input Field
        self.query_layout = QHBoxLayout()

        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Paste the video URL here...")
        self.query_layout.addWidget(self.url_input)

        self.summary_dropdown = QComboBox()
        self.summary_dropdown.addItems([
            "Brief Summary",
            "Detailed Summary",
            "Key Points Only",
            "Custom"
        ])
        self.summary_dropdown.currentIndexChanged.connect(self.summary_type_changed)
        self.query_layout.addWidget(self.summary_dropdown)

        layout.addLayout(self.query_layout)

        self.summary_query_editor = QTextEdit()
        self.summary_query_editor.setPlaceholderText("Type the summary query here...")
        self.summary_query_editor.setMaximumHeight(100)
        self.summary_query_editor.hide()
        layout.addWidget(self.summary_query_editor)

        # Read-only text field for processing output
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.hide()  # Initially hidden
        layout.addWidget(self.result_text)

        # Submit Button
        self.submit_button = QPushButton("Generate Summary")
        self.submit_button.setStyleSheet(
            "background-color: #4CAF50; color: white; font-size: 14px; padding: 10px;"
        )
        self.submit_button.clicked.connect(self.handle_summary_request)
        layout.addWidget(self.submit_button)

        self.setLayout(layout)

    def summary_type_changed(self):
        if self.summary_dropdown.currentText() == "Custom":
            self.summary_query_editor.show()

    def handle_summary_request(self):
        url = self.url_input.text()
        summary_type = self.summary_dropdown.currentText()

        if not url.strip():
            QMessageBox.warning(self, "Input Error", "Please enter a valid video URL.")
            return

        self.result_text.show()
        self.result_text.setText("Initializing download...")
        self.submit_button.setEnabled(False)

        # Start video download
        self.download_worker = VideoDownloadWorker(url)
        self.download_worker.progress_signal.connect(self.update_progress)
        self.download_worker.error_signal.connect(self.handle_error)
        self.download_worker.finished_signal.connect(self.download_complete)
        self.download_worker.start()

    def update_progress(self, progress):
        self.result_text.setText(progress)

    def handle_error(self, error_message):
        QMessageBox.critical(self, "Download Error", error_message)
        self.submit_button.setEnabled(True)
        self.result_text.setText("Error occurred. Please try again.")

    def download_complete(self, filename):
        self.result_text.setText(f"Download complete: {filename}")
        self.do_asr(filename)

    def handle_summary_request(self):
        url = self.url_input.text()
        summary_type = self.summary_dropdown.currentText()

        if not url.strip():
            QMessageBox.warning(self, "Input Error", "Please enter a valid video URL.")
            return

        self.result_text.show()
        self.result_text.setText("Initializing download...")
        self.submit_button.setEnabled(False)

        # Start video download
        self.download_worker = VideoDownloadWorker(url)
        self.download_worker.progress_signal.connect(self.update_progress)
        self.download_worker.error_signal.connect(self.handle_error)
        self.download_worker.finished_signal.connect(self.download_complete)
        self.download_worker.start()

    def do_asr(self, filename):
        self.asr_worker = AudioASRWorker(filename)
        self.asr_worker.error_signal.connect(self.handle_error)
        self.asr_worker.progress_signal.connect(self.update_progress)
        self.asr_worker.finished_signal.connect(self.asr_complete)
        self.asr_worker.start()

    def asr_complete(self, text):
        self.result_text.append("Transcription completed.")
        self.result_text.append("Starting summarization...")

        # Start LLMWorker for summarization
        summary_type = self.summary_dropdown.currentText()
        custom_query = self.summary_query_editor.toPlainText() if summary_type == "Custom" else None

        self.llm_worker = LLMWorker(text, summary_type, custom_query)
        self.llm_worker.error_signal.connect(self.handle_error)
        self.llm_worker.progress_signal.connect(self.update_progress)
        self.llm_worker.finished_signal.connect(self.llm_complete)
        self.llm_worker.start()

    def llm_complete(self, summary):
        self.result_text.append("Summary:")
        self.result_text.append(summary)
        self.submit_button.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoSummaryApp()
    window.show()
    sys.exit(app.exec_())
