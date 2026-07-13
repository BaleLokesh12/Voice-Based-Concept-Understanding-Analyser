import os
import re
import tempfile
import streamlit as st
from datetime import datetime

# Import local modules
import audio_processor
import transcriber
import evaluator
import report_generator

# Page Config
st.set_page_config(
    page_title="VBCUA | Voice-Based Concept Understanding Analyser",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS for premium design
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
    
    /* Override font family */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Outfit', sans-serif;
    }
    
    .stCodeBlock, code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* Hide default Streamlit headers */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Base background adjustments (dark-mode styling) */
    .stApp {
        background-color: #0B0F19;
        color: #E2E8F0;
    }
    
    /* Header card styling */
    .header-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 30px;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px 0 rgba(0, 0, 0, 0.4);
    }
    
    .header-title {
        font-size: 32px;
        font-weight: 700;
        background: linear-gradient(90deg, #60A5FA 0%, #34D399 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    
    .header-subtitle {
        color: #94A3B8;
        font-size: 15px;
        font-weight: 300;
        line-height: 1.5;
    }
    
    /* Glassmorphic metrics panel cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.25);
    }
    
    .glass-card-title {
        font-size: 13px;
        color: #94A3B8;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-bottom: 10px;
    }
    
    .glass-card-value {
        font-size: 32px;
        font-weight: 700;
        color: #FFFFFF;
    }
    
    /* Glowing main score styling */
    .glow-score-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 35px 20px;
        background: radial-gradient(circle at center, rgba(59, 130, 246, 0.15) 0%, rgba(0, 0, 0, 0) 70%);
        border-radius: 20px;
        border: 1px dashed rgba(59, 130, 246, 0.25);
        margin-bottom: 20px;
    }
    
    .glow-score {
        font-size: 64px;
        font-weight: 800;
        background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(96, 165, 250, 0.3);
        margin: 5px 0;
    }
    
    /* Status Badges */
    .badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 5px;
    }
    
    .badge-strong {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    
    .badge-moderate {
        background-color: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    
    .badge-basic {
        background-color: rgba(59, 130, 246, 0.15);
        color: #60A5FA;
        border: 1px solid rgba(59, 130, 246, 0.4);
    }
    
    .badge-poor {
        background-color: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    
    /* Highlighted filler words inside transcription */
    .filler-highlight {
        background-color: rgba(239, 68, 68, 0.25);
        color: #F87171;
        font-weight: 600;
        padding: 1px 4px;
        border-radius: 4px;
        border-bottom: 2px dotted #EF4444;
    }
    
    .bridge-highlight {
        background-color: rgba(245, 158, 11, 0.25);
        color: #FBBF24;
        font-weight: 600;
        padding: 1px 4px;
        border-radius: 4px;
        border-bottom: 2px dotted #F59E0B;
    }
    
    /* Transcript Box */
    .transcript-box {
        background-color: #0F172A;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        font-size: 16px;
        line-height: 1.6;
        color: #CBD5E1;
        max-height: 220px;
        overflow-y: auto;
        margin-bottom: 20px;
        font-style: italic;
    }
    
    /* Checklist keywords */
    .keyword-badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 500;
        margin: 4px;
    }
    
    .kw-covered {
        background-color: rgba(16, 185, 129, 0.1);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.25);
    }
    
    .kw-missed {
        background-color: rgba(239, 68, 68, 0.08);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.2);
        text-decoration: line-through;
    }
    
    /* Sidebar customization */
    [data-testid="stSidebar"] {
        background-color: #080C14;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# Main Application Banner
st.markdown("""
<div class="header-card">
    <div class="header-title">VBCUA</div>
    <div class="header-subtitle">
        <b>Voice-Based Concept Understanding Analyser</b> — Evaluate your technical articulation.
        Upload an audio explanation or record your response directly. VBCUA runs Speech-to-Text transcription, 
        evaluates vocabulary against concept frameworks using S-BERT semantic embeddings, and analyzes your verbal 
        fluency (filler words, pauses, RMS energy).
    </div>
</div>
""", unsafe_allow_html=True)

# Initialize Session State
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = None
if "audio_analysis" not in st.session_state:
    st.session_state.audio_analysis = None
if "current_audio_path" not in st.session_state:
    st.session_state.current_audio_path = None
if "student_name" not in st.session_state:
    st.session_state.student_name = "Student User"

# ----------------- SIDEBAR -----------------
st.sidebar.title("⚙️ VBCUA Settings")

st.session_state.student_name = st.sidebar.text_input(
    "Student Name", 
    value=st.session_state.student_name,
    placeholder="Enter your name"
)

# Configuration Panel
st.sidebar.header("🔑 AI APIs & Backend")
transcription_backend = st.sidebar.selectbox(
    "Transcription Backend",
    options=["Google Speech Recognition (Free / Local)", "OpenAI Whisper API"],
    index=0
)

api_key = None
if transcription_backend == "OpenAI Whisper API":
    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-proj-...",
        help="Required for Whisper API backend transcription. Your key is not stored."
    )
    if not api_key:
        st.sidebar.warning("⚠️ OpenAI API Key is required to run OpenAI Whisper.")

# Signal Settings
st.sidebar.header("🎙️ Signal Processing")
silence_threshold = st.sidebar.slider(
    "Silence Level Threshold",
    min_value=0.01,
    max_value=0.10,
    value=0.03,
    step=0.01,
    help="Relative volume fraction below which audio is categorized as silence."
)
min_pause_dur = st.sidebar.slider(
    "Min Pause Duration (sec)",
    min_value=0.2,
    max_value=2.0,
    value=0.5,
    step=0.1,
    help="Minimum duration of a silent period to count as a pause/hesitation."
)

# Concept Library Reference view
st.sidebar.header("📚 Predefined Concepts")
selected_ref_concept = st.sidebar.selectbox(
    "Reference Concept Details",
    options=list(evaluator.REFERENCE_CONCEPTS.keys())
)
concept_ref = evaluator.REFERENCE_CONCEPTS[selected_ref_concept]
st.sidebar.markdown(f"**Description:**\n{concept_ref['definition']}")
st.sidebar.markdown(f"**Key vocabulary to include:**\n`{', '.join(concept_ref['keywords'])}`")

# ----------------- MAIN FLOW -----------------

# Form fields for Audio Input and Concept Selection
st.subheader("📝 Step 1: Select Topic & Submit Audio")

col_a, col_b = st.columns([1, 1])

with col_a:
    concept_to_evaluate = st.selectbox(
        "Select Concept to Explain:",
        options=list(evaluator.REFERENCE_CONCEPTS.keys()),
        index=0,
        help="Select the concept you will explain in your voice recording."
    )
    
    st.info(f"💡 **Target Explanation Definition:**\n{evaluator.REFERENCE_CONCEPTS[concept_to_evaluate]['definition']}")

with col_b:
    input_method = st.radio(
        "Select Input Method:",
        options=["🔴 Record Live", "📤 Upload File"],
        horizontal=True
    )
    
    audio_file = None
    if input_method == "🔴 Record Live":
        # Streamlit 1.58.0 native audio input recorder
        audio_file = st.audio_input(
            "Record your verbal explanation:",
            help="Click the microphone, record your explanation, and click stop. Minimum 5 seconds recommended."
        )
    else:
        audio_file = st.file_uploader(
            "Upload an audio file (WAV files recommended for offline execution):",
            type=["wav", "mp3", "m4a", "ogg"],
            help="Upload your recorded oral response."
        )

# Analyze Button
analyze_button = st.button("🔥 Run AI Evaluation Analysis", type="primary", disabled=(audio_file is None))

# ----------------- PROCESSING PIPELINE -----------------
if analyze_button and audio_file is not None:
    # Save audio input to a temporary WAV file
    temp_dir = tempfile.gettempdir()
    temp_audio_path = os.path.join(temp_dir, f"vbcua_user_audio_{int(datetime.now().timestamp())}.wav")
    
    # Save uploaded/recorded audio stream
    with open(temp_audio_path, "wb") as f:
        f.write(audio_file.read())
        
    st.session_state.current_audio_path = temp_audio_path
    
    # Show pipeline indicators
    with st.status("🔮 Analyzing explanation...", expanded=True) as status_bar:
        try:
            status_bar.update(label="1. Loading audio file and processing signal features...", state="running")
            # Run signal processing
            analysis_results = audio_processor.analyze_audio(
                temp_audio_path,
                silence_threshold=silence_threshold,
                min_pause_duration=min_pause_dur
            )
            st.session_state.audio_analysis = analysis_results
            
            # 2. Transcription
            status_bar.update(label="2. Transcribing audio speech to text...", state="running")
            backend_key = "openai" if transcription_backend == "OpenAI Whisper API" else "google"
            transcription = transcriber.transcribe_audio(
                temp_audio_path,
                api_key=api_key,
                backend=backend_key
            )
            
            # 3. Running Semantic & Fluency Grader
            status_bar.update(label="3. Running semantic similarity S-BERT comparison and checking speech fluency...", state="running")
            eval_results = evaluator.evaluate_explanation(
                concept_to_evaluate,
                transcription,
                analysis_results["duration"],
                analysis_results["pause_ratio"]
            )
            st.session_state.evaluation_results = eval_results
            
            st.session_state.analysis_done = True
            status_bar.update(label="✅ Evaluation completed successfully!", state="complete")
            
        except Exception as err:
            status_bar.update(label="❌ Pipeline Error", state="error")
            st.error(f"Analysis failed: {err}")
            st.session_state.analysis_done = False

# ----------------- DASHBOARD DISPLAY -----------------
if st.session_state.analysis_done and st.session_state.evaluation_results is not None:
    eval_res = st.session_state.evaluation_results
    anal_res = st.session_state.audio_analysis
    
    st.success("🎉 Analysis complete! Scroll down to inspect your dashboard report.")
    
    st.markdown("<hr style='border: 1px solid rgba(255, 255, 255, 0.08); margin: 30px 0;'>", unsafe_allow_html=True)
    st.subheader("📊 VBCUA Performance Evaluation Dashboard")
    
    # Grid Row 1: Overall score & Comprehension level
    col1, col2 = st.columns([1, 2.2])
    
    with col1:
        # Glow Overall Score Card
        overall_val = eval_res["overall_score"]
        understanding_lvl = eval_res["understanding_level"]
        
        # Select badge CSS class
        if "Strong" in understanding_lvl:
            badge_class = "badge-strong"
        elif "Moderate" in understanding_lvl:
            badge_class = "badge-moderate"
        elif "Basic" in understanding_lvl:
            badge_class = "badge-basic"
        else:
            badge_class = "badge-poor"
            
        st.markdown(f"""
        <div class="glow-score-container">
            <span style="font-size: 13px; color: #94A3B8; text-transform: uppercase; font-weight: 600;">Overall VBCUA Score</span>
            <div class="glow-score">{overall_val:.1f}</div>
            <span class="badge {badge_class}">{understanding_lvl}</span>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        # Dashboard key metrics widgets in 4 columns
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown(f"""
            <div class="glass-card">
                <div class="glass-card-title">Concept Comprehension</div>
                <div class="glass-card-value">{eval_res['comprehension_score']:.1f}%</div>
                <div style="font-size: 12px; color: #94A3B8; margin-top: 4px;">Semantic match & vocabulary coverage</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_m2:
            st.markdown(f"""
            <div class="glass-card">
                <div class="glass-card-title">Speech Fluency</div>
                <div class="glass-card-value">{eval_res['fluency_score']:.1f}%</div>
                <div style="font-size: 12px; color: #94A3B8; margin-top: 4px;">Pace (WPM), pauses, and verbal fillers</div>
            </div>
            """, unsafe_allow_html=True)
            
        col_m3, col_m4, col_m5 = st.columns(3)
        with col_m3:
            st.markdown(f"""
            <div class="glass-card" style="padding: 12px 18px;">
                <div class="glass-card-title" style="font-size: 10px;">Semantic Similarity</div>
                <div class="glass-card-value" style="font-size: 20px;">{eval_res['semantic_similarity']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m4:
            st.markdown(f"""
            <div class="glass-card" style="padding: 12px 18px;">
                <div class="glass-card-title" style="font-size: 10px;">Keyword Coverage</div>
                <div class="glass-card-value" style="font-size: 20px;">{eval_res['keyword_coverage']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m5:
            st.markdown(f"""
            <div class="glass-card" style="padding: 12px 18px;">
                <div class="glass-card-title" style="font-size: 10px;">Audio Duration</div>
                <div class="glass-card-value" style="font-size: 20px;">{anal_res['duration']:.1f}s</div>
            </div>
            """, unsafe_allow_html=True)

    # Grid Row 2: Speech Signals Waveform Graph
    st.markdown("### 🎙️ Acoustic Feature Envelope & Pause Overlays")
    waveform_fig = audio_processor.create_waveform_plot(anal_res, dark_theme=True)
    st.plotly_chart(waveform_fig, use_container_width=True)
    
    # Grid Row 3: Transcription and Word Analysis
    col_t1, col_t2 = st.columns([1.5, 1])
    
    with col_t1:
        st.markdown("### 📝 Speech-to-Text Transcription Transcript")
        
        # Display transcript text with highlighted filler words
        transcript = eval_res["transcription"]
        
        if not transcript.strip():
            st.markdown("<div class='transcript-box'>No spoken words detected in the audio file.</div>", unsafe_allow_html=True)
        else:
            # Highlight fillers
            filler_map = eval_res["fluency_details"]["filler_map"]
            
            # Format text using HTML spans
            highlighted_transcript = transcript
            
            # Split transcription by spaces and highlight matches (maintaining boundaries)
            # 1. Bold verbal bridge fillers (like, you know) in orange
            # 2. Bold vocal hesitations (um, uh) in red
            for f in filler_map.keys():
                if f == "you know":
                    highlighted_transcript = re.compile(r'\b(you\s+know)\b', re.IGNORECASE).sub(
                        r'<span class="bridge-highlight">\1</span>', highlighted_transcript
                    )
                elif f in ["like", "so", "actually"]:
                    highlighted_transcript = re.compile(r'\b(' + re.escape(f) + r')\b', re.IGNORECASE).sub(
                        r'<span class="bridge-highlight">\1</span>', highlighted_transcript
                    )
                else:
                    highlighted_transcript = re.compile(r'\b(' + re.escape(f) + r')\b', re.IGNORECASE).sub(
                        r'<span class="filler-highlight">\1</span>', highlighted_transcript
                    )
            
            st.markdown(f"""<div class="transcript-box">"{highlighted_transcript}"</div>""", unsafe_allow_html=True)
            
    with col_t2:
        # Speaking Rate pacing gauge
        st.markdown("### ⏱️ Delivery Fluency Details")
        
        fd = eval_res["fluency_details"]
        
        col_fd1, col_fd2, col_fd3 = st.columns(3)
        with col_fd1:
            st.metric("Tempo (WPM)", f"{fd['wpm']:.1f}", 
                      help="Words per Minute. Ideal target: 110-150 WPM.")
        with col_fd2:
            st.metric("Pause Ratio", f"{fd['pause_ratio']*100:.1f}%", 
                      help="Percentage of audio time spent in silence. Ideal: 10-25%.")
        with col_fd3:
            st.metric("Filler Words", f"{fd['filler_count']}", 
                      help="Count of vocal filler words like um, uh, like.")

    # Grid Row 4: Keyword checklist
    st.markdown("### 📚 Key Technical Terminology Coverage Checklist")
    covered_kws = eval_res["covered_keywords"]
    missed_kws = eval_res["missed_keywords"]
    
    kw_badges_html = ""
    for kw in covered_kws:
        kw_badges_html += f'<span class="keyword-badge kw-covered">✔ {kw}</span>'
    for kw in missed_kws:
        kw_badges_html += f'<span class="keyword-badge kw-missed">✖ {kw}</span>'
        
    st.markdown(f'<div style="background-color: #0F172A; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 15px; margin-bottom: 20px;">{kw_badges_html}</div>', unsafe_allow_html=True)

    # Grid Row 5: AI-Generated Feedback & Growth Advice & PDF Report
    col_fb1, col_fb2 = st.columns([2, 1])
    
    with col_fb1:
        st.markdown("### 💡 Articulation Feedback & Guidance")
        feedback_list = eval_res["feedback"]
        for item in feedback_list:
            if "Excellent" in item or "Superb" in item:
                st.info(f"🌟 {item}")
            elif "Consider" in item or "Key terms" in item:
                st.warning(f"🎯 {item}")
            elif "missed" in item or "Poor" in item or "No verbal" in item:
                st.error(f"❌ {item}")
            else:
                st.info(f"🗣️ {item}")
                
    with col_fb2:
        st.markdown("### 📄 Evaluation Report PDF Export")
        st.write("Compile this evaluation, including metrics, transcription, feedback, and the acoustic waveform visualization, into a professional PDF report.")
        
        # Report generator trigger
        try:
            pdf_filepath = report_generator.build_pdf_report(
                eval_res,
                anal_res,
                student_name=st.session_state.student_name
            )
            
            with open(pdf_filepath, "rb") as pdf_file:
                pdf_bytes = pdf_file.read()
                
            st.download_button(
                label="📥 Download Structured PDF Report",
                data=pdf_bytes,
                file_name=f"VBCUA_Assessment_{eval_res['concept_name'].replace(' ', '_')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            
            # Clean up generated file
            if os.path.exists(pdf_filepath):
                try:
                    os.remove(pdf_filepath)
                except Exception:
                    pass
        except Exception as e:
            st.error(f"Could not build PDF report: {e}")
