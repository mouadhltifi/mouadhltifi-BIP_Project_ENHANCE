from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QListWidget, QVBoxLayout,
    QWidget, QLabel, QHBoxLayout, QLineEdit, QTextEdit, QSplitter, QLabel, QVBoxLayout, QPushButton
)
from PyQt5.QtGui import QPixmap, QPainter
from PyQt5.QtCore import Qt
import sys
import json
import requests

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Patient Management System")
        self.setGeometry(100, 100, 1200, 700)  # Increased height to accommodate buttons

        # Load patient data from JSON file
        self.patient_data = self.load_patient_data()

        # Main Splitter
        main_splitter = QSplitter()
        self.setCentralWidget(main_splitter)

        # Left Panel: List of Patients
        self.patient_list = QListWidget()
        self.patient_list.addItems(list(self.patient_data.keys()))
        self.patient_list.currentItemChanged.connect(self.display_patient_info)
        main_splitter.addWidget(self.patient_list)

        # Right Panel: Splitter for Patient Details, API Response, and Images
        right_splitter = QSplitter()
        main_splitter.addWidget(right_splitter)

        # Patient Details Panel
        self.patient_info = QTextEdit()
        self.patient_info.setReadOnly(True)
        self.patient_info.setText("Select a patient to see details.")
        right_splitter.addWidget(self.patient_info)

        # API Response Panel
        api_response_panel = QWidget()
        api_response_layout = QVBoxLayout(api_response_panel)
        self.api_response_info = QTextEdit()
        self.api_response_info.setReadOnly(False)  # Make API response editable
        self.api_response_info.setText("API response will be displayed here.")

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
        right_splitter.addWidget(api_response_panel)

        # Images Panel
        self.image_panel = QWidget()
        self.image_layout = QVBoxLayout(self.image_panel)
        self.image_labels = [QLabel() for _ in range(2)]
        for i, image_label in enumerate(self.image_labels):
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setScaledContents(False)  # Disable scaled contents to maintain aspect ratio
            image_label.setPixmap(QPixmap("placeholder.png"))  # Placeholder images
            self.image_layout.addWidget(image_label)
        right_splitter.addWidget(self.image_panel)

        # Set stretch factors for resizable columns
        main_splitter.setStretchFactor(0, 1)  # Patient list
        main_splitter.setStretchFactor(1, 4)  # Right panel
        right_splitter.setStretchFactor(0, 2)  # Patient info (30%)
        right_splitter.setStretchFactor(1, 2)  # API response (30%)
        right_splitter.setStretchFactor(2, 1)  # Images (20%)

    def load_patient_data(self):
        try:
            with open('patients.json', 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            return {}

    def display_patient_info(self, current, previous):
        if current:
            # Make a dummy API call
            response = self.make_dummy_response(current.text())
            patient_details = self.patient_data.get(current.text(), {})
            if isinstance(patient_details, dict):
                details_text = "\n".join([f"{key}: {value}" for key, value in patient_details.items() if key != "images"])
                self.patient_info.setText(details_text)
                self.api_response_info.setText(json.dumps(response, indent=2))
                # Set patient images if available
                image_paths = patient_details.get("images", ["placeholder.png"] * 2)
                for i, image_path in enumerate(image_paths[:2]):
                    pixmap = QPixmap(image_path)
                    if not pixmap.isNull():
                        self.image_labels[i].setPixmap(pixmap.scaled(self.image_labels[i].width(), self.image_labels[i].height(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation))
                    else:
                        self.image_labels[i].setPixmap(QPixmap("placeholder.png").scaled(self.image_labels[i].width(), self.image_labels[i].height(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation))
            else:
                self.patient_info.setText("No data available.")
                self.api_response_info.setText("No data available.")
                for image_label in self.image_labels:
                    image_label.setPixmap(QPixmap("placeholder.png").scaled(image_label.width(), image_label.height(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation))

    def make_api_call(self, patient_name):
        # Dummy API call (replace with real API logic later)
        """
        try:
            response = requests.get('https://jsonplaceholder.typicode.com/posts/1')
            if response.status_code == 200:
                return response.json()
            else:
                return "Failed to fetch data"
        except requests.RequestException as e:
            return f"Error: {str(e)}"
        """
        return "API call is currently commented out. Use make_dummy_response instead."

    def make_dummy_response(self, patient_name):
        # Dummy response based on loaded JSON data
        patient_details = self.patient_data.get(patient_name, {})

        if patient_details:
            response = {
                "patient_name": patient_name,
                "age": patient_details.get("Age", "N/A"),
                "condition": patient_details.get("Condition", "N/A"),
                "prescriptions": patient_details.get("Prescriptions", "N/A"),
                "Imaging Findings": "A recent MRI brain scan reveals a mass located in the left frontal lobe. The lesion appears hyperintense on T2-weighted imaging, with heterogeneous contrast enhancement noted on T1-weighted imaging post-gadolinium. The tumor measures approximately 3.5 cm in maximum diameter. There is evidence of mild perilesional edema, and a slight midline shift of approximately 2 mm is observed. No hemorrhagic components are noted. The radiological characteristics are consistent with a glioma, possibly a Grade II or III, pending further biopsy and histopathological analysis.",
                "Diagnosis and Considerations": "The imaging findings, combined with Jane's clinical symptoms, suggest a diagnosis of glioma, likely a low- to intermediate-grade glioma. Differential diagnoses include oligodendroglioma or astrocytoma. A biopsy is recommended for definitive histological classification and to determine the tumor grade.",
                "Recommended Next Steps": [
                    "Neurological Consultation: Referral to a neuro-oncologist for further evaluation and treatment planning.",
                    "Biopsy: Surgical biopsy to obtain a tissue sample for histological analysis to confirm tumor type and grade.",
                    "Treatment Planning: Depending on biopsy results, treatment options may include surgical resection, radiation therapy, and/or chemotherapy.",
                    "Symptom Management: Start corticosteroids to manage edema and reduce intracranial pressure, and consider anti-seizure medication if clinically indicated."
                ],
                "Prognosis": "The prognosis will depend on the final histopathological grade of the glioma and its molecular profile. Early intervention, including potential surgical resection, is critical in improving outcomes.",
                "Follow-Up": "A follow-up appointment is suggested within the next two weeks to discuss biopsy results and initiate treatment."
            }
            return response
        else:
            return "No data available for the selected patient"

    def confirm_changes(self):
        # Logic for confirming changes
        print("Changes confirmed:")
        print(self.api_response_info.toPlainText())

    def discard_changes(self):
        # Logic for discarding changes
        self.api_response_info.setText("API response will be displayed here.")
        print("Changes discarded.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
