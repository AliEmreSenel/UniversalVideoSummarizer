import hashlib
import tempfile

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLineEdit, QLabel, QComboBox, QPushButton, QMessageBox, QTextEdit, QHBoxLayout,
    QFormLayout, QDialog, QDialogButtonBox, QCheckBox, QFileDialog, QStyle
)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QThread, pyqtSignal
from tempfile import mkstemp
import os
import sys
import yt_dlp
import time
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from configparser import ConfigParser
from pathlib import Path
import logging
from pydub import AudioSegment

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_all_devices():
    devices = {}
    for i in range(torch.cuda.device_count()):
        device = torch.cuda.get_device_name(i)
        devices[f"cuda:{i}"] = device
    if torch.backends.mps.is_available():
        devices["mps"] = "Apple Neural Engine"
    devices["cpu"] = "CPU"
    return devices

available_devices = get_all_devices()

def device_name_to_id(device_name):
    for i in range(torch.cuda.device_count()):
        if torch.cuda.get_device_name(i) == device_name:
            return f"cuda:{i}"
    if device_name == "Apple Neural Engine":
        return "mps"
    if device_name == "CPU":
        return "cpu"
    return "cpu"

DEFAULT_CONFIG = {
    "ASR_Model": ("distil-whisper/distil-large-v3", "Model to be used for ASR"),
    "LLM_Model": ("Qwen/Qwen2-0.5B", "Model to be used for summarization"),
    "Device": ("cpu", "Device to run the models on", available_devices),
    "Fallback_Device": ("cpu", "Device to fall back to if the primary device runs out of memory", available_devices),
}

def get_config_path():
    if sys.platform == "win32":
        base_dir = Path(os.getenv("APPDATA", os.path.expanduser("~\\AppData\\Roaming")))
    elif sys.platform == "darwin":
        base_dir = Path(os.path.expanduser("~/Library/Application Support"))
    else:
        base_dir = Path(os.path.expanduser("~/.config"))
    config_dir = base_dir / "UniversalVideoSummarizer"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "uvs.conf"

def validate_config(config):
    is_valid = True
    for key, (default, _description, *choices) in DEFAULT_CONFIG.items():
        if key in config['Settings']:
            value = config['Settings'][key]

            if isinstance(default, bool):
                try:
                    config['Settings'][key] = value == "true"
                except ValueError:
                    logging.warning(f"Invalid boolean value for '{key}': '{value}'. Using default '{default}'.")
                    config['Settings'][key] = default
                    is_valid = False

            elif choices and value not in choices[0]:
                logging.warning(f"Invalid value for '{key}': '{value}' not in {choices[0]}. Using default '{default}'.")
                config['Settings'][key] = default
                is_valid = False
    return config, is_valid

def load_config():
    config_path = get_config_path()
    config = ConfigParser()

    if config_path.exists():
        config.read(config_path)
        config, valid = validate_config(config)
        if not valid:
            logging.warning("One or more invalid configurations were reset to defaults.")
    else:
        config['Settings'] = {}
        save_config(config)

    for key, (default, _description, *choices) in DEFAULT_CONFIG.items():
        if key not in config['Settings']:
            config['Settings'][key] = default

    return config

def save_config(config):
    config_path = get_config_path()
    new_config = ConfigParser()
    new_config['Settings'] = {}

    for key, (default, *_rest) in DEFAULT_CONFIG.items():
        value = config['Settings'].get(key, default)

        if value != default:
            new_config['Settings'][key] = ("true" if value else "false") if isinstance(value, bool) else value

    with open(config_path, 'w') as configfile:
        new_config.write(configfile)

def is_url(url):
    return url.startswith("http://") or url.startswith("https://")

class SettingsDialog(QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.config = config
        self.setWindowTitle("Settings")
        self.setLayout(QVBoxLayout())

        # Form layout to display settings
        self.form_layout = QFormLayout()
        self.fields = {}
        self.original_values = {}  # To track initial values

        # Populate the form with settings
        for key, (default, description, *choices) in DEFAULT_CONFIG.items():
            label = QLabel(key)
            label.setToolTip(description)  # Show description on hover

            value = self.config['Settings'].get(key, default)
            self.original_values[key] = value  # Save original value

            if isinstance(default, bool):
                # Use checkbox for boolean values
                checkbox = QCheckBox()
                checkbox.setChecked(value == "true")
                checkbox.setToolTip(description)  # Tooltip for checkbox
                checkbox.stateChanged.connect(self.check_for_changes)
                widget = checkbox
            elif choices:  # Use a dropdown for options with predefined choices
                combo_box = QComboBox()
                if "Device" in key:
                    choices[0] = list(available_devices.values())
                    value = available_devices.get(value, "Auto")
                combo_box.addItems(choices[0])  # Add the list of choices
                combo_box.setCurrentText(value)
                combo_box.setToolTip(description)  # Tooltip for dropdown
                combo_box.currentTextChanged.connect(self.check_for_changes)
                widget = combo_box
            else:  # Use a text input for other options
                line_edit = QLineEdit(value)
                line_edit.setToolTip(description)  # Tooltip for input
                line_edit.textChanged.connect(self.check_for_changes)
                widget = line_edit

            self.fields[key] = widget
            self.form_layout.addRow(label, widget)

        self.layout().addLayout(self.form_layout)

        self.button_box = QDialogButtonBox()
        self.save_button = self.button_box.addButton(QDialogButtonBox.Save)
        self.close_button = self.button_box.addButton(QDialogButtonBox.Close)
        self.apply_button = self.button_box.addButton(QDialogButtonBox.Apply)
        self.apply_button.hide()
        self.save_button.setDisabled(True)
        self.close_button.clicked.connect(self.reject)
        self.save_button.clicked.connect(self.apply_and_exit)
        self.apply_button.clicked.connect(self.apply_changes)
        self.layout().addWidget(self.button_box)

    def get_field_value(self, key):
        return self.fields[key].isChecked() if isinstance(self.fields[key], QCheckBox) \
            else self.fields[key].currentText() if isinstance(self.fields[key], QComboBox) \
            else self.fields[key].text()

    def check_for_changes(self):
        """Check if any field value has changed from the original."""
        has_changes = any(
            self.get_field_value(key)
            != self.original_values[key]
            for key in self.fields
        )
        if has_changes:
            print(self.button_box.buttons())
            self.apply_button.show()
            self.save_button.setDisabled(False)
        else:
            self.apply_button.hide()
            self.save_button.setDisabled(True)

    def apply_and_exit(self):
        self.apply_changes()
        self.accept()

    def apply_changes(self):
        """Save updated settings back to config."""
        for key in self.fields:
            self.config['Settings'][key] = self.get_field_value(key)
            if "Device" in key:
                self.config['Settings'][key] = device_name_to_id(self.config['Settings'][key])
        save_config(self.config)
        self.main_window.config = self.config
        print(self.config['Settings'], self.fields, self.original_values)
        self.original_values = {key: self.get_field_value(key) for key in self.fields}
        self.check_for_changes()

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

class FileToAudioWorker(QThread):
    error_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)

    def __init__(self, filename):
        super().__init__()
        self.filename = filename

    def run(self):
        try:
            out_filename = mkstemp("wav")[1] + ".wav"
            audio = AudioSegment.from_file(self.filename)
            audio.export(out_filename, format="wav")
            self.finished_signal.emit(out_filename)
        except Exception as e:
            self.error_signal.emit(str(e))

class AudioASRWorker(QThread):
    error_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)
    def __init__(self, config, filename):
        super().__init__()
        self.config = config
        self.filename = filename

    def run(self):
        self.progress_signal.emit(f"Loading ASR Model...")

        # Initialize the ASR pipeline
        try:
            pipe = pipeline("automatic-speech-recognition",
                            model=self.config["Settings"]["ASR_Model"],
                            device=self.config["Settings"]["Device"],
                            torch_dtype=torch.float16)
        except torch.OutOfMemoryError:
            self.progress_signal.emit(f"Out of memory. Falling back to {available_devices[self.config['Settings']['Fallback_Device']]}")
            pipe = pipeline("automatic-speech-recognition",
                            model=self.config["Settings"]["ASR_Model"],
                            device=self.config["Settings"]["Fallback_Device"],
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

    def __init__(self, config, transcript, summary_type, custom_query=None):
        super().__init__()
        self.transcript = transcript
        self.summary_type = summary_type
        self.custom_query = custom_query
        self.config = config

    def run(self):
        try:

            model_name = self.config["Settings"]["LLM_Model"]

            self.progress_signal.emit(f"Loading {model_name} model for summarization...")

            tokenizer = None
            model = None
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name, device_map=self.config["Settings"]["Device"])
                model = AutoModelForCausalLM.from_pretrained(model_name, device_map=self.config["Settings"]["Device"])
                self.device = self.config["Settings"]["Device"]
            except torch.OutOfMemoryError:
                self.progress_signal.emit(f"Out of memory. Falling back to {available_devices[self.config['Settings']['Fallback_Device']]}")
                tokenizer = AutoTokenizer.from_pretrained(model_name, device_map=self.config["Settings"]["Fallback_Device"])
                model = AutoModelForCausalLM.from_pretrained(model_name, device_map=self.config["Settings"]["Fallback_Device"])
                self.device = self.config["Settings"]["Fallback_Device"]

            # Load the transcript and build the prompt
            if self.summary_type == "Brief Summary":
                prompt = f"Provide a brief summary: "
            elif self.summary_type == "Detailed Summary":
                prompt = f"Provide a detailed summary: "
            elif self.summary_type == "Key Points Only":
                prompt = f"Extract only the key points: "
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
                {"role": "user", "content": prompt + " " + self.transcript },

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
        self.config = load_config()
        self.setWindowTitle("Video Summary Tool")
        self.setGeometry(200, 200, 400, 350)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Settings button at the top right
        self.settings_layout = QHBoxLayout()
        self.settings_layout.addStretch()  # Push button to the right
        self.settings_button = QPushButton("Settings")
        self.settings_button.setIcon(QIcon.fromTheme("settings"))  # Use gear icon
        self.settings_button.clicked.connect(self.open_settings)
        self.settings_layout.addWidget(self.settings_button)
        layout.addLayout(self.settings_layout)

        # Title label
        self.title_label = QLabel("Enter Video URL and Select Summary Type")
        self.title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(self.title_label)

        # URL Input Field
        self.query_layout = QHBoxLayout()

        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Paste the video URL here...")
        self.query_layout.addWidget(self.url_input)

        self.file_chooser = QFileDialog()
        self.file_chooser.setFileMode(QFileDialog.AnyFile)

        self.file_select_button = QPushButton("")
        self.file_select_button.setIcon(self.style().standardIcon(QStyle.SP_DirIcon))
        self.file_select_button.clicked.connect(self.open_file_chooser)
        print(self.file_select_button.size())
        self.query_layout.addWidget(self.file_select_button)

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

    def open_file_chooser(self):
        url = self.url_input.text()
        if url != "":
            # Check if network url or local path
            if not is_url(url):
                self.file_chooser.selectFile(url)
        if self.file_chooser.exec_():
            self.url_input.setText(self.file_chooser.selectedFiles()[0])

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

    def update_progress_append(self, progress):
        self.result_text.append(progress)

    def handle_error(self, error_message):
        QMessageBox.critical(self, "Download Error", error_message)
        self.submit_button.setEnabled(True)
        self.result_text.append("Error occurred. Please try again.")

    def download_complete(self, filename):
        self.result_text.append(f"Download complete: {filename}")
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

        if is_url(url):
            self.url_hash = hashlib.sha512(url.encode()).hexdigest()
        else:
            self.url_hash = hashlib.sha512(open(url, "rb").read()).hexdigest()

        tmpdir = tempfile.gettempdir()
        if os.path.exists(tmpdir + "/" + self.url_hash):
            self.result_text.append("Video already transcribed")
            self.load_transcript(tmpdir + "/" + self.url_hash)
        else:
            if is_url(url):
                # Start video download
                self.download_worker = VideoDownloadWorker(url)
                self.download_worker.progress_signal.connect(self.update_progress)
                self.download_worker.error_signal.connect(self.handle_error)
                self.download_worker.finished_signal.connect(self.download_complete)
                self.download_worker.start()
            else:
                self.file_convert_worker = FileToAudioWorker(url)
                self.file_convert_worker.error_signal.connect(self.handle_error)
                self.file_convert_worker.finished_signal.connect(self.do_asr)
                self.file_convert_worker.start()

    def load_transcript(self, filename):
        f = open(filename)
        text = f.read()
        f.close()
        self.asr_complete(text)

    def do_asr(self, filename):
        self.asr_worker = AudioASRWorker(self.config, filename)
        self.asr_worker.error_signal.connect(self.handle_error)
        self.asr_worker.progress_signal.connect(self.update_progress_append)
        self.asr_worker.finished_signal.connect(self.asr_complete)
        self.asr_worker.start()

    def asr_complete(self, text):
        self.result_text.append("Transcription completed.")

        f = open(tempfile.gettempdir() + "/" + self.url_hash, "w+")
        f.write(text)
        f.close()

        self.result_text.append("Starting summarization...")

        # Start LLMWorker for summarization
        summary_type = self.summary_dropdown.currentText()
        custom_query = self.summary_query_editor.toPlainText() if summary_type == "Custom" else None

        self.llm_worker = LLMWorker(self.config, text, summary_type, custom_query)
        self.llm_worker.error_signal.connect(self.handle_error)
        self.llm_worker.progress_signal.connect(self.update_progress_append)
        self.llm_worker.finished_signal.connect(self.llm_complete)
        self.llm_worker.start()

    def llm_complete(self, summary):
        self.result_text.append("Summary:")
        self.result_text.append(summary)
        self.submit_button.setEnabled(True)

    def open_settings(self):
        # Open the settings dialog
        dialog = SettingsDialog(self.config, self)
        if dialog.exec():
            # Settings were applied
            print("Settings updated:", dict(self.config['Settings']))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoSummaryApp()
    window.show()
    sys.exit(app.exec_())
