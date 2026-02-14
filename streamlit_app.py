import os
import streamlit as st
from pydub import AudioSegment
import speech_recognition as sr
from transformers import pipeline
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import tempfile

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Voice Sentiment Analyzer",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===== CUSTOM CSS =====
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #1a1145 50%, #0a0e1a 100%);
        font-family: 'Inter', sans-serif;
    }

    .main-title {
        text-align: center;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #818cf8, #c084fc, #f472b6, #fb923c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0;
    }

    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    .chip-row {
        display: flex;
        justify-content: center;
        gap: 0.6rem;
        margin-bottom: 2rem;
        flex-wrap: wrap;
    }

    .chip {
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.25);
        color: #a5b4fc;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 500;
    }

    .metric-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s;
    }

    .metric-card:hover {
        border-color: rgba(99, 102, 241, 0.3);
        transform: translateY(-2px);
    }

    .emotion-emoji { font-size: 2.5rem; }
    .emotion-label {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #cbd5e1;
        margin-top: 0.3rem;
    }
    .emotion-count {
        font-size: 0.85rem;
        color: #64748b;
    }

    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 1rem;
    }

    div[data-testid="stFileUploader"] {
        background: rgba(15, 23, 42, 0.6);
        border: 2px dashed rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 1rem;
    }

    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ===== LOAD AI MODEL (cached so it only loads once) =====
@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None
    )

emotion_classifier = load_model()

# ===== EMOTION CONFIG =====
EMOTION_CONFIG = {
    'anger':    {'color': '#ef4444', 'emoji': '😡'},
    'disgust':  {'color': '#a855f7', 'emoji': '🤢'},
    'fear':     {'color': '#6b7280', 'emoji': '😨'},
    'joy':      {'color': '#22c55e', 'emoji': '😊'},
    'neutral':  {'color': '#3b82f6', 'emoji': '😐'},
    'sadness':  {'color': '#6366f1', 'emoji': '😢'},
    'surprise': {'color': '#f59e0b', 'emoji': '😲'},
    'unknown':  {'color': '#9ca3af', 'emoji': '❓'}
}


# ===== AUDIO ANALYSIS FUNCTION =====
def analyze_audio(audio_file):
    # Save uploaded file to a temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    audio = AudioSegment.from_file(tmp_path)
    chunk_length_ms = 10000
    results = []
    recognizer = sr.Recognizer()

    total_chunks = (len(audio) + chunk_length_ms - 1) // chunk_length_ms
    progress_bar = st.progress(0, text="Processing audio chunks...")

    for idx, i in enumerate(range(0, len(audio), chunk_length_ms)):
        chunk = audio[i:i + chunk_length_ms]

        start_sec = i // 1000
        end_sec = min((i + chunk_length_ms) // 1000, len(audio) // 1000)
        timestamp = f"{start_sec // 60}:{start_sec % 60:02d} - {end_sec // 60}:{end_sec % 60:02d}"

        # Export chunk to temp wav
        chunk_path = tmp_path + "_chunk.wav"
        chunk.export(chunk_path, format="wav")

        # Speech recognition
        with sr.AudioFile(chunk_path) as source:
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data)
            except sr.UnknownValueError:
                text = "[inaudible]"
            except sr.RequestError:
                text = "[speech service unavailable]"

        # Emotion detection
        if text and text not in ["[inaudible]", "[speech service unavailable]"]:
            emotions = emotion_classifier(text)[0]
            top_emotion = max(emotions, key=lambda x: x['score'])
        else:
            emotions = []
            top_emotion = {"label": "unknown", "score": 0}

        results.append({
            "timestamp": timestamp,
            "start_seconds": start_sec,
            "text": text,
            "top_emotion": top_emotion["label"],
            "confidence": round(top_emotion["score"] * 100, 1),
            "all_emotions": {e["label"]: round(e["score"] * 100, 1) for e in emotions}
        })

        # Update progress
        progress_bar.progress(
            (idx + 1) / total_chunks,
            text=f"Analyzing chunk {idx + 1}/{total_chunks}..."
        )

        # Clean up chunk
        if os.path.exists(chunk_path):
            os.remove(chunk_path)

    # Clean up original temp file
    os.remove(tmp_path)
    progress_bar.empty()

    return results


# ===== HEADER =====
st.markdown('<div style="text-align:center;font-size:3rem;">🎙️</div>', unsafe_allow_html=True)
st.markdown('<h1 class="main-title">Voice Sentiment Analyzer</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered emotion detection from speech — pinpoint every mood shift</p>', unsafe_allow_html=True)
st.markdown('''
<div class="chip-row">
    <span class="chip">🧠 Deep Learning</span>
    <span class="chip">📊 Real-time Charts</span>
    <span class="chip">⏱️ Timestamp Tracking</span>
</div>
''', unsafe_allow_html=True)

# ===== FILE UPLOAD =====
uploaded_file = st.file_uploader(
    "Upload an audio file",
    type=["wav", "mp3", "ogg", "flac", "m4a"],
    help="Supported formats: WAV, MP3, OGG, FLAC, M4A"
)

if uploaded_file is not None:
    # Show audio player
    st.audio(uploaded_file)

    # Analyze button
    if st.button("⚡ Analyze Emotions", type="primary", use_container_width=True):
        with st.spinner(""):
            results = analyze_audio(uploaded_file)

        if not results:
            st.error("No results generated. Please try a different audio file.")
        else:
            st.success(f"✅ Analysis complete! {len(results)} segments analyzed.")
            st.divider()

            # ===== SUMMARY CARDS =====
            st.markdown('<h3 class="section-title">📊 Emotion Overview</h3>', unsafe_allow_html=True)

            emotion_counts = {}
            for r in results:
                emotion_counts[r['top_emotion']] = emotion_counts.get(r['top_emotion'], 0) + 1

            cols = st.columns(len(emotion_counts))
            for i, (emotion, count) in enumerate(sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)):
                cfg = EMOTION_CONFIG.get(emotion, EMOTION_CONFIG['unknown'])
                with cols[i]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="emotion-emoji">{cfg['emoji']}</div>
                        <div class="emotion-label">{emotion}</div>
                        <div class="emotion-count">{count} segment{'s' if count > 1 else ''}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.divider()

            # ===== EMOTION TIMELINE CHART =====
            st.markdown('<h3 class="section-title">📈 Emotion Timeline</h3>', unsafe_allow_html=True)

            all_emotions = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

            fig_line = go.Figure()
            for emotion in all_emotions:
                cfg = EMOTION_CONFIG.get(emotion, EMOTION_CONFIG['unknown'])
                y_values = [r['all_emotions'].get(emotion, 0) for r in results]
                fig_line.add_trace(go.Scatter(
                    x=[r['timestamp'] for r in results],
                    y=y_values,
                    mode='lines+markers',
                    name=emotion.capitalize(),
                    line=dict(color=cfg['color'], width=2.5),
                    marker=dict(size=7, color=cfg['color']),
                    hovertemplate=f"{emotion.capitalize()}: %{{y}}%<extra></extra>"
                ))

            fig_line.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=420,
                margin=dict(l=40, r=20, t=30, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                xaxis=dict(title="Time Segment", gridcolor="rgba(148,163,184,0.08)"),
                yaxis=dict(title="Confidence %", range=[0, 100], gridcolor="rgba(148,163,184,0.08)"),
                hovermode="x unified"
            )
            st.plotly_chart(fig_line, use_container_width=True)

            # ===== TWO COLUMNS: PIE + DOMINANT =====
            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<h3 class="section-title">🍩 Distribution</h3>', unsafe_allow_html=True)

                pie_colors = [EMOTION_CONFIG.get(e, EMOTION_CONFIG['unknown'])['color'] for e in emotion_counts.keys()]
                fig_pie = go.Figure(data=[go.Pie(
                    labels=[e.capitalize() for e in emotion_counts.keys()],
                    values=list(emotion_counts.values()),
                    hole=0.55,
                    marker=dict(colors=pie_colors, line=dict(width=2, color='#0a0e1a')),
                    textfont=dict(size=13),
                    hoverinfo="label+percent+value"
                )])
                fig_pie.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=350,
                    margin=dict(l=20, r=20, t=20, b=20),
                    showlegend=True,
                    legend=dict(font=dict(size=12))
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                st.markdown('<h3 class="section-title">🏆 Dominant Emotion</h3>', unsafe_allow_html=True)

                dominant = max(emotion_counts, key=emotion_counts.get)
                dominant_cfg = EMOTION_CONFIG.get(dominant, EMOTION_CONFIG['unknown'])
                dominant_pct = round(emotion_counts[dominant] / len(results) * 100)

                st.markdown(f"""
                <div style="text-align:center; padding:2rem 0;">
                    <div style="font-size:5rem;">{dominant_cfg['emoji']}</div>
                    <div style="font-size:1.8rem; font-weight:700; color:#e2e8f0; text-transform:capitalize; margin-top:0.5rem;">{dominant}</div>
                    <div style="font-size:1.1rem; color:#64748b; margin-top:0.25rem;">{dominant_pct}% of audio</div>
                    <div style="margin-top:1rem;">
                        <span style="background:{dominant_cfg['color']}; color:white; padding:0.4rem 1.2rem; border-radius:20px; font-weight:600; font-size:0.9rem;">
                            {emotion_counts[dominant]} segment{'s' if emotion_counts[dominant] > 1 else ''}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.divider()

            # ===== DETAILED TABLE =====
            st.markdown('<h3 class="section-title">🕐 Detailed Timeline</h3>', unsafe_allow_html=True)
            st.markdown('<p style="color:#475569;font-size:0.9rem;margin-bottom:1rem;">Exact timestamps where each emotion was detected</p>', unsafe_allow_html=True)

            table_data = []
            for r in results:
                cfg = EMOTION_CONFIG.get(r['top_emotion'], EMOTION_CONFIG['unknown'])
                table_data.append({
                    "⏱️ Time": r['timestamp'],
                    "💬 Transcript": r['text'],
                    "😊 Emotion": f"{cfg['emoji']} {r['top_emotion'].capitalize()}",
                    "📊 Confidence": f"{r['confidence']}%"
                })

            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # ===== EMOTION HEATMAP (BONUS) =====
            st.divider()
            st.markdown('<h3 class="section-title">🌡️ Emotion Heatmap</h3>', unsafe_allow_html=True)

            heatmap_data = []
            for r in results:
                for emotion in all_emotions:
                    heatmap_data.append({
                        "Time": r['timestamp'],
                        "Emotion": emotion.capitalize(),
                        "Score": r['all_emotions'].get(emotion, 0)
                    })

            df_heat = pd.DataFrame(heatmap_data)
            df_pivot = df_heat.pivot(index="Emotion", columns="Time", values="Score")

            fig_heat = go.Figure(data=go.Heatmap(
                z=df_pivot.values,
                x=df_pivot.columns,
                y=df_pivot.index,
                colorscale=[[0, '#0a0e1a'], [0.3, '#312e81'], [0.6, '#7c3aed'], [0.8, '#c084fc'], [1.0, '#f472b6']],
                hovertemplate="Time: %{x}<br>Emotion: %{y}<br>Score: %{z}%<extra></extra>"
            ))
            fig_heat.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=320,
                margin=dict(l=80, r=20, t=20, b=40),
                xaxis=dict(title="Time Segment"),
                yaxis=dict(title="")
            )
            st.plotly_chart(fig_heat, use_container_width=True)

# ===== FOOTER =====
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#334155;font-size:0.82rem;">Built with ❤️ using Streamlit, HuggingFace Transformers & Plotly</p>',
    unsafe_allow_html=True
)
