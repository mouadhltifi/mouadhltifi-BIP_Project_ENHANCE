from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QListWidget, QVBoxLayout, QWidget, QLabel, QTextEdit, QSplitter, QPushButton, QHBoxLayout
from PyQt5.QtGui import QPixmap
import json
import sys
import google.generativeai as genai


class APICallThread(QThread):
    result = pyqtSignal(str)

    def __init__(self, patient_name, patient_details, parent=None):
        super().__init__(parent)
        self.patient_name = patient_name
        self.patient_details = patient_details

    def run(self):
        try:
            # Configure and call Gemini API
            genai.configure(api_key="AIzaSyD2P68uDQsaHG5Ow2Lxa-2gAQFDW8EHdq0")
            model = genai.GenerativeModel("gemini-1.5-flash")
            diagnosis = self.patient_details.get("Suggested Diagnosis", "Unknown")
            prompt = (
                f"Patient Details:\n"
                f"Name: {self.patient_name}\n"
                f"Age: {self.patient_details.get('Age', 'N/A')}\n"
                f"Gender: {self.patient_details.get('Gender', 'N/A')}\n"
                f"Condition: {self.patient_details.get('Condition', 'N/A')}\n"
                f"Diagnosis: {diagnosis}\n\n"
                f"Please draft a diagnosis document for the patient based on the above details. "
                f"This will be a digital document. If any information is missing, invent reasonable data such as the doctor's name, the patient's id, exam date or any other you might see fit. "
                f"Only use plain text with no special characters or things like bold or italic. "
                f"For context, the patient came in for a brain tumor investigation and did an MRI scan. Provide a diagnosis only for what concerns the MRI scan result and not the other symptoms. "
                f"All data provided is fake and intended for a Proof of Concept (PoC). The response should only contain the document content and no additional interaction."
            )
            response = model.generate_content(prompt)
            self.result.emit(response.text)
        except Exception as e:
            self.result.emit(f"Error occurred: {str(e)}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Patient Management System")
        self.setGeometry(100, 100, 1200, 700)

        self.patient_data = self.load_patient_data()
        self.api_thread = None

        # Main UI setup
        main_splitter = QSplitter()
        self.setCentralWidget(main_splitter)

        self.patient_list = QListWidget()
        self.patient_list.addItems(self.patient_data.keys())
        self.patient_list.currentItemChanged.connect(self.on_patient_selected)
        main_splitter.addWidget(self.patient_list)

        self.patient_info = QTextEdit()
        self.patient_info.setReadOnly(True)
        main_splitter.addWidget(self.patient_info)

        # Right Panel: API Response Panel
        api_response_panel = QWidget()
        api_response_layout = QVBoxLayout(api_response_panel)
        self.api_response_info = QTextEdit()
        self.api_response_info.setReadOnly(True)
        self.api_response_info.setText("")  # Clear response on start

        # Buttons Panel for API Response
        self.confirm_button = QPushButton("Confirm")
        self.confirm_button.clicked.connect(self.confirm_changes)
        self.discard_button = QPushButton("Discard")
        self.discard_button.clicked.connect(self.discard_changes)
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.confirm_button)
        buttons_layout.addWidget(self.discard_button)

        # Add API response info and buttons to layout
        api_response_layout.addWidget(self.api_response_info)
        api_response_layout.addLayout(buttons_layout)
        main_splitter.addWidget(api_response_panel)

        self.image_panel = QWidget()
        self.image_layout = QVBoxLayout(self.image_panel)
        self.image_labels = [QLabel() for _ in range(2)]
        for label in self.image_labels:
            label.setAlignment(Qt.AlignCenter)
            label.setPixmap(QPixmap("placeholder.png"))
            self.image_layout.addWidget(label)
        main_splitter.addWidget(self.image_panel)

    def load_patient_data(self):
        try:
            with open("patients.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def on_patient_selected(self, current, previous):
        if not current:
            return
        patient_name = current.text()
        patient_details = self.patient_data.get(patient_name, {})

        # Update patient info
        self.patient_info.setText(
            "\n".join(f"{key}: {value}" for key, value in patient_details.items() if key != "images")
        )

        # Clear API response panel initially
        self.api_response_info.setText("")

        # Update images
        for i, label in enumerate(self.image_labels):
            if i < len(patient_details.get("images", [])):
                pixmap = QPixmap(patient_details["images"][i])
                if not pixmap.isNull():
                    label.setPixmap(pixmap.scaledToWidth(300, Qt.SmoothTransformation))
                else:
                    label.setPixmap(QPixmap("placeholder.png"))
            else:
                label.setPixmap(QPixmap("placeholder.png"))

        # Call API
        self.api_thread = APICallThread(patient_name, patient_details)
        self.api_thread.result.connect(self.update_api_response)
        self.api_thread.start()

    def update_api_response(self, response_text):
        self.api_response_info.setText(response_text)

    def confirm_changes(self):
        print("Changes confirmed:")
        print(self.api_response_info.toPlainText())

    def discard_changes(self):
        print("Changes discarded.")
        self.api_response_info.setText("")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
