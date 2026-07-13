import os
import wave
import struct
import math
import numpy as np

import audio_processor
import transcriber
import evaluator
import report_generator

def generate_test_wav(filename, duration=5.0, sample_rate=16000):
    """
    Generates a synthetic WAV file containing:
    - 0.0s to 1.5s: 440Hz sine wave tone
    - 1.5s to 2.5s: silence (1.0s duration)
    - 2.5s to 4.0s: 440Hz sine wave tone
    - 4.0s to 5.0s: silence (1.0s duration)
    """
    print(f"Generating test WAV file: {filename}...")
    num_samples = int(duration * sample_rate)
    
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit (2 bytes)
        wav_file.setframerate(sample_rate)
        
        for i in range(num_samples):
            t = i / sample_rate
            # Play tone or write silence
            if (0.0 <= t < 1.5) or (2.5 <= t < 4.0):
                # 440Hz tone at 50% amplitude
                value = int(16384.0 * math.sin(2.0 * math.pi * 440.0 * t))
            else:
                value = 0
            
            packed = struct.pack('<h', value)
            wav_file.writeframesraw(packed)
            
    print("Test WAV file generated.")

def test_pipeline():
    test_wav = "test_speech_profile.wav"
    
    try:
        # 1. Generate audio
        generate_test_wav(test_wav)
        
        # 2. Test Audio Signal Processing
        print("\n--- Testing Audio Processing Module ---")
        analysis = audio_processor.analyze_audio(test_wav, silence_threshold=0.03, min_pause_duration=0.4)
        print(f"Audio Duration: {analysis['duration']:.2f} seconds (Expected: 5.00)")
        print(f"Detected Pauses: {len(analysis['pauses'])}")
        for idx, p in enumerate(analysis['pauses']):
            print(f"  Pause {idx+1}: {p[0]:.2f}s to {p[1]:.2f}s (Duration: {p[2]:.2f}s)")
        print(f"Pause Ratio: {analysis['pause_ratio']:.2%}")
        
        assert abs(analysis['duration'] - 5.0) < 0.1, "Duration mismatch!"
        assert len(analysis['pauses']) >= 2, "Should detect at least 2 silence segments!"
        
        # 3. Test Plotly figure creation (make sure it doesn't crash)
        print("Creating Plotly waveform visualizer...")
        fig = audio_processor.create_waveform_plot(analysis)
        assert fig is not None, "Plotly figure generation failed!"
        
        # 4. Test Transcription (on a sine wave, Google Speech Recognition should return empty text gracefully)
        print("\n--- Testing Transcription Module ---")
        print("Attempting Google Web Speech transcription (free fallback)...")
        try:
            transcription = transcriber.transcribe_audio(test_wav, backend="google")
            print(f"Transcription result: '{transcription}'")
        except Exception as e:
            print(f"Google speech API not reachable (normal if offline/restricted): {e}")
            transcription = ""
            
        # 5. Test Evaluator (Semantic & Fluency Grader)
        print("\n--- Testing Evaluator Module ---")
        mock_transcript = "Machine learning uses data and algorithms to train models for supervised learning predictions."
        print(f"Mock user transcript: '{mock_transcript}'")
        
        eval_results = evaluator.evaluate_explanation(
            "Machine Learning",
            mock_transcript,
            analysis["duration"],
            analysis["pause_ratio"]
        )
        
        print(f"Overall VBCUA Score: {eval_results['overall_score']:.1f}/100")
        print(f"Comprehension Score: {eval_results['comprehension_score']:.1f}%")
        print(f"Fluency Score: {eval_results['fluency_score']:.1f}%")
        print(f"Understanding Level: {eval_results['understanding_level']}")
        print(f"Covered Keywords: {eval_results['covered_keywords']}")
        print("Feedback comments:")
        for fb in eval_results["feedback"]:
            print(f"  - {fb}")
            
        assert eval_results['overall_score'] > 0, "Evaluation score failed!"
        
        # 6. Test PDF Report Generation
        print("\n--- Testing PDF Report Generator ---")
        pdf_path = report_generator.build_pdf_report(
            eval_results,
            analysis,
            student_name="Test Verifier"
        )
        print(f"Report compiled successfully at: {pdf_path}")
        assert os.path.exists(pdf_path), "PDF report not found!"
        
        # Cleanup
        os.remove(pdf_path)
        print("Temporary PDF report cleaned up.")
        
        print("\n[SUCCESS] Pipeline Verification SUCCESSFUL! All modules compiled and integrate properly.")
        
    except Exception as e:
        print(f"\n[FAIL] Pipeline Verification FAILED: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(test_wav):
            os.remove(test_wav)
            print("Temporary audio file cleaned up.")

if __name__ == "__main__":
    test_pipeline()
