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
from transformers import pipeline
import torch



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
    finished_signal = pyqtSignal(dict)
    def __init__(self, filename):
        super().__init__()
        self.filename = filename

    def run(self):
        self.progress_signal.emit(f"Loading ASR Model...")

        # Initialize the ASR pipeline
        pipe = pipeline("automatic-speech-recognition",
                        model="distil-whisper/distil-large-v3",
                        device="cuda:0",
                        torch_dtype=torch.float32)

        self.progress_signal.emit(f"Model Loaded. Transcribing...")

        start_time = time.perf_counter()
        outputs = pipe(self.filename, chunk_length_s=30, batch_size=1, return_timestamps=False)

        self.progress_signal.emit("Transcription Completed. Time Elapsed: " + str(time.perf_counter() - start_time))
        # Output the results
        self.finished_signal.emit(outputs)


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

    def do_asr(self, filename):
        self.asr_worker = AudioASRWorker(filename)
        self.asr_worker.error_signal.connect(self.handle_error)
        self.asr_worker.progress_signal.connect(self.update_progress)
        self.asr_worker.finished_signal.connect(self.asr_complete)
        self.asr_worker.start()

    def asr_complete(self, text):
        print(text)
        self.result_text.append(text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoSummaryApp()
    window.show()
    sys.exit(app.exec_())
