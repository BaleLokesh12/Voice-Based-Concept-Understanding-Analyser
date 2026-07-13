import numpy as np
import soundfile as sf
import plotly.graph_objects as go
from scipy.io import wavfile

def load_audio(file_path):
    """
    Loads an audio file and converts it to mono if necessary.
    Uses soundfile as primary loader, falls back to scipy.io.wavfile.
    """
    try:
        data, samplerate = sf.read(file_path)
    except Exception as e:
        # Fallback to scipy wavfile if soundfile fails
        try:
            samplerate, data = wavfile.read(file_path)
            # Normalize to [-1.0, 1.0] if integer type
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float32) / 2147483648.0
            elif data.dtype == np.uint8:
                data = (data.astype(np.float32) - 128.0) / 128.0
        except Exception as inner_e:
            raise ValueError(f"Could not load audio file. Supported formats are WAV, MP3 (if system decoders available), OGG. Details: {e} | {inner_e}")

    # Convert to mono if multi-channel
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
        
    return data, samplerate

def analyze_audio(file_path, silence_threshold=0.03, min_pause_duration=0.5):
    """
    Analyzes the audio file to calculate RMS energy envelope,
    detect pauses (hesitations), and compute the pause ratio.
    """
    data, samplerate = load_audio(file_path)
    total_duration = len(data) / samplerate
    
    # Calculate RMS energy envelope manually
    # Window size of 100ms, step size of 25ms
    frame_length = int(samplerate * 0.1) # 100ms
    hop_length = int(samplerate * 0.025) # 25ms
    
    if frame_length == 0 or hop_length == 0:
        return {
            "duration": 0.0,
            "pause_ratio": 0.0,
            "pauses": [],
            "avg_rms": 0.0,
            "rms_envelope": np.array([]),
            "rms_times": np.array([]),
            "waveform": np.array([]),
            "waveform_times": np.array([])
        }
        
    rms = []
    rms_times = []
    
    for i in range(0, len(data) - frame_length, hop_length):
        frame = data[i:i+frame_length]
        rms.append(np.sqrt(np.mean(frame**2)) if len(frame) > 0 else 0.0)
        rms_times.append((i + frame_length/2) / samplerate)
        
    rms = np.array(rms)
    rms_times = np.array(rms_times)
    
    if len(rms) == 0:
        return {
            "duration": total_duration,
            "pause_ratio": 0.0,
            "pauses": [],
            "avg_rms": 0.0,
            "rms_envelope": rms,
            "rms_times": rms_times,
            "waveform": data,
            "waveform_times": np.linspace(0, total_duration, len(data))
        }
        
    # Detect pauses based on silence threshold
    max_rms = np.max(rms)
    if max_rms == 0:
        max_rms = 1.0
        
    is_silent = rms < (silence_threshold * max_rms)
    
    pauses = []
    in_pause = False
    pause_start = 0.0
    
    for j in range(len(is_silent)):
        if is_silent[j] and not in_pause:
            in_pause = True
            pause_start = rms_times[j]
        elif not is_silent[j] and in_pause:
            in_pause = False
            pause_end = rms_times[j]
            duration = pause_end - pause_start
            if duration >= min_pause_duration:
                pauses.append((pause_start, pause_end, duration))
                
    if in_pause:
        pause_end = rms_times[-1]
        duration = pause_end - pause_start
        if duration >= min_pause_duration:
            pauses.append((pause_start, pause_end, duration))
            
    total_pause_duration = sum(p[2] for p in pauses)
    pause_ratio = total_pause_duration / total_duration if total_duration > 0 else 0.0
    avg_rms = np.mean(rms)
    
    # Downsample raw waveform for faster visualization
    # Goal: downsample to around 2000 points
    downsample_factor = max(1, len(data) // 2000)
    waveform_downsampled = data[::downsample_factor]
    waveform_times = np.linspace(0, total_duration, len(waveform_downsampled))
    
    return {
        "duration": total_duration,
        "pause_ratio": pause_ratio,
        "pauses": pauses,
        "avg_rms": avg_rms,
        "rms_envelope": rms,
        "rms_times": rms_times,
        "waveform": waveform_downsampled,
        "waveform_times": waveform_times
    }

def create_waveform_plot(analysis_results, dark_theme=True):
    """
    Creates an interactive Plotly waveform plot.
    Includes the waveform, RMS envelope, and shaded pause areas.
    """
    wf = analysis_results["waveform"]
    wf_t = analysis_results["waveform_times"]
    rms = analysis_results["rms_envelope"]
    rms_t = analysis_results["rms_times"]
    pauses = analysis_results["pauses"]
    
    # Theme configuration
    bg_color = "rgba(17, 24, 39, 0.9)" if dark_theme else "white"
    grid_color = "rgba(75, 85, 99, 0.3)" if dark_theme else "rgba(229, 231, 235, 1)"
    wf_color = "#3B82F6"      # Bright Blue
    rms_color = "#EC4899"     # Pink
    pause_color = "rgba(239, 68, 68, 0.25)" # Soft Red
    
    fig = go.Figure()
    
    # 1. Waveform Line
    fig.add_trace(go.Scatter(
        x=wf_t, y=wf,
        mode="lines",
        name="Waveform",
        line=dict(color=wf_color, width=1),
        opacity=0.75,
        hoverinfo="skip"
    ))
    
    # 2. RMS Energy Envelope
    fig.add_trace(go.Scatter(
        x=rms_t, y=rms,
        mode="lines",
        name="RMS Energy (Volume)",
        line=dict(color=rms_color, width=2),
        opacity=0.9
    ))
    
    # 3. Highlight Pauses
    for idx, (start, end, dur) in enumerate(pauses):
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor=pause_color,
            opacity=0.5,
            layer="below",
            line_width=0,
            annotation_text=f"Pause {idx+1} ({dur:.1f}s)" if idx == 0 or dur > 1.0 else "",
            annotation_position="top left",
            annotation_font=dict(color="#EF4444", size=9)
        )
        
    fig.update_layout(
        title=dict(
            text="Speech Signal Analysis (Waveform, Energy Envelope, and Pauses)",
            font=dict(size=14, color="#E5E7EB" if dark_theme else "#1F2937")
        ),
        xaxis=dict(
            title="Time (seconds)",
            gridcolor=grid_color,
            zerolinecolor=grid_color,
            color="#9CA3AF" if dark_theme else "#4B5563"
        ),
        yaxis=dict(
            title="Amplitude / Energy",
            gridcolor=grid_color,
            zerolinecolor=grid_color,
            color="#9CA3AF" if dark_theme else "#4B5563"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            font=dict(color="#E5E7EB" if dark_theme else "#1F2937"),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=20, t=50, b=40),
        height=320
    )
    
    return fig
