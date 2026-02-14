# 🎙️ Voice Sentiment Analyzer

AI-powered web application that analyzes emotions in audio/voice recordings and displays a real-time interactive dashboard with emotion timeline, distribution charts, and detailed timestamps.

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-000000?logo=flask)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface)
![Chart.js](https://img.shields.io/badge/Chart.js-4.x-FF6384?logo=chartdotjs)

---

## ✨ Features

- **Audio Upload** — Drag & drop or browse for `.wav`, `.mp3`, `.ogg`, `.flac` files
- **Speech-to-Text** — Converts spoken words to text using Google Speech Recognition
- **AI Emotion Detection** — Classifies 7 emotions (joy, sadness, anger, fear, surprise, disgust, neutral) using a fine-tuned DistilRoBERTa model
- **Emotion Timeline** — Interactive line chart pinpointing when each emotion occurs
- **Emotion Distribution** — Doughnut chart showing overall emotion breakdown
- **Detailed Timestamps** — Table showing exact minute:second with transcript, detected emotion, and confidence percentage
- **Premium Dark UI** — Animated gradient backgrounds, glassmorphism cards, glow effects, and smooth micro-animations

---

## 🏗️ Architecture

```
User uploads audio
       │
       ▼
┌──────────────────┐
│   Flask Server   │
│   (app.py)       │
└───────┬──────────┘
        │
        ▼
┌──────────────────┐     ┌─────────────────────┐
│  pydub           │────▶│  SpeechRecognition   │
│  (split audio    │     │  (speech → text)     │
│   into 10s       │     └──────────┬───────────┘
│   chunks)        │                │
└──────────────────┘                ▼
                         ┌─────────────────────┐
                         │  HuggingFace Model   │
                         │  (text → emotions)   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  Chart.js Dashboard  │
                         │  (visualize results) │
                         └─────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **ffmpeg** (required by pydub for audio processing)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Sentiment_Analysis.git
cd Sentiment_Analysis

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install flask SpeechRecognition pydub transformers torch

# Install ffmpeg (Windows)
winget install ffmpeg
```

### Run the App

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

> **Note:** On first run, the AI model (~329MB) will be downloaded automatically.

---

## 📁 Project Structure

```
Sentiment_Analysis/
├── app.py                 # Flask backend + audio processing logic
├── templates/
│   └── index.html         # Dashboard UI with Chart.js visualizations
├── static/
│   └── style.css          # Premium dark theme styles
├── uploads/               # Temporary storage for uploaded files
├── requirements.txt       # Python dependencies
└── README.md
```

---

## 🧠 How It Works

1. **Upload** — User uploads an audio file via the web interface
2. **Split** — Audio is split into 10-second chunks using `pydub`
3. **Transcribe** — Each chunk is converted to text using Google Speech Recognition API
4. **Classify** — Text is analyzed by `j-hartmann/emotion-english-distilroberta-base` model which detects 7 emotions
5. **Visualize** — Results are sent to the frontend and rendered as interactive charts

---

## 📊 Detected Emotions

| Emotion | Emoji | Description |
|---------|-------|-------------|
| Joy | 😊 | Happiness, excitement, delight |
| Sadness | 😢 | Sorrow, grief, disappointment |
| Anger | 😡 | Frustration, rage, irritation |
| Fear | 😨 | Anxiety, worry, nervousness |
| Surprise | 😲 | Astonishment, shock, wonder |
| Disgust | 🤢 | Revulsion, distaste, aversion |
| Neutral | 😐 | No strong emotion detected |

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| [Flask](https://flask.palletsprojects.com/) | Web framework |
| [SpeechRecognition](https://pypi.org/project/SpeechRecognition/) | Speech-to-text conversion |
| [pydub](https://github.com/jiaaro/pydub) | Audio file manipulation |
| [HuggingFace Transformers](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) | Emotion classification model |
| [Chart.js](https://www.chartjs.org/) | Interactive chart rendering |
| [ffmpeg](https://ffmpeg.org/) | Audio codec support |

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">Built with ❤️ using Flask, HuggingFace Transformers & Chart.js</p>
