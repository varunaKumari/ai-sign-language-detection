# 🤟 AI Sign Language Detection

A real-time sign language detection system using computer vision and machine learning. The application captures hand gestures through a webcam and classifies them into corresponding sign language words.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Features

- **Real-time hand detection** using webcam
- **Hand gesture classification** for sign language words
- **Data collection module** for creating custom training datasets
- **User-friendly frontend** interface
- Supports multiple sign language gestures including: `Yes`, `No`, `Thank You`, `Hello`, `Please`, `I Love You`

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Core programming language |
| OpenCV | Video capture & image processing |
| CVZone | Hand tracking & detection |
| NumPy | Numerical computations |
| Math | Mathematical operations |
| TensorFlow/Keras | Model training & classification |

---

## 📁 Project Structure
ai-sign-language-detection/
├── data/ # Training dataset (hand gesture images)
│ ├── yes/
│ ├── no/
│ ├── hello/
│ ├── please/
│ ├── thank You/
│ └── I love You/
├── dataCollection.py # Script to collect training data
├── test.py # Script to test the trained model
├── frontend.py # Frontend interface
├── README.md
└── requirements.txt


---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/varunaKumari/ai-sign-language-detection.git
cd ai-sign-language-detection

2. Create a virtual environment
bash
python -m venv .venv
3. Activate the virtual environment
Windows (PowerShell):

bash
.venv\Scripts\Activate.ps1
Mac/Linux:

bash
source .venv/bin/activate
4. Install dependencies
bash
pip install -r requirements.txt
🚀 Usage
Collect Training Data
bash
python dataCollection.py
Opens webcam and captures hand gesture images
Images are saved in the data/ folder organized by gesture
Test the Model
bash
python test.py
Runs the trained model on live webcam feed
Displays predicted sign language gesture in real-time
Run the Frontend
bash
python frontend.py
📊 How It Works
Data Collection — Webcam captures hand images using CVZone's HandDetector
Preprocessing — Images are cropped, resized to 300x300, and normalized
Training — A classification model is trained on the collected gesture images
Prediction — Real-time webcam feed detects hand gestures and classifies them
📸 Supported Gestures
Gesture	Description
✋ Hello	Open palm wave
👍 Yes	Thumbs up
👎 No	Thumbs down
🙏 Thank You	Open hands together
🤞 Please	Crossed fingers
🤟 I Love You	ILY hand sign
🤝 Contributing
Contributions are welcome! Feel free to:

Fork the repository
Create a feature branch (git checkout -b feature/new-feature)
Commit changes (git commit -m 'Add new feature')
Push to branch (git push origin feature/new-feature)
Open a Pull Request
👩‍💻 Author
Varuna Kumari

GitHub: @varunaKumari
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

