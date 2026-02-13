import os
from flask import Flask, render_template,request, jsonify
from pydub import AudioSegment
import speech_recognition as sr
from transformers import pipeline

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

emotion_classifier = pipeline('text-classification', model = 'j-hartmann/emotion-english-distilroberta-base', top_k = None)

def analyze_audio(filepath):
    audio = AudioSegment.from_file(filepath)
    chunk_length_ms = 10000
    results = []
    recognizer = sr.Recognizer()

    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        start_sec = i // 1000
        end_sec = min((i + chunk_length_ms) // 1000, len(audio) // 1000)
        timestamp = f"{start_sec // 60}:{start_sec % 60:02d} - {end_sec // 60}:{end_sec % 60:02d}"

        chunk_path = os.path.join(app.config['UPLOAD_FOLDER'], "temp_chunk.wav")
        chunk.export(chunk_path, format = "wav")

        with sr.AudioFile(chunk_path) as source:
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data)
            except sr.UnknownValueError:
                text = "[inaudible]"
            except sr.RequestError:
                text = "[speech service unavailable]"

        if text and text not in ["[inaudible]", "[speech service unavailable]"]:
            emotions = emotion_classifier(text)[0]  # Get emotion scores
            top_emotion = max(emotions, key=lambda x: x['score'])  # Highest score
        else:
            emotions = []
            top_emotion = {"label": "unknown", "score": 0}
        
        # Add this chunk's result
        results.append({
            "timestamp": timestamp,
            "start_seconds": start_sec,
            "text": text,
            "top_emotion": top_emotion["label"],
            "confidence": round(top_emotion["score"] * 100, 1),
            "all_emotions": {e["label"]: round(e["score"] * 100, 1) for e in emotions}
        })
    
    # Clean up temp file
    if os.path.exists(chunk_path):
        os.remove(chunk_path)
    
    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods = ['POST'])
def analyze():
    if 'audio' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    results = analyze_audio(filepath)
    os.remove(filepath)

    return jsonify(results)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=8080)