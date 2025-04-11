import torch
import whisper
from transformers import pipeline
import gradio as gr
from datetime import datetime
import os
import sys
from docx import Document
import warnings
warnings.filterwarnings("ignore")

# 1. System Configuration
def configure_system():
    """Set up system paths and dependencies"""
    # Install required packages
    try:
        import tensorflow
    except ImportError:
        os.system("pip install tensorflow==2.12 keras<2.13,>=2.12")
    
    # Verify FFmpeg
    try:
        ffmpeg_check = os.system("ffmpeg -version")
        if ffmpeg_check != 0:  # If FFmpeg is not found
            print("FFmpeg not found. Installing...")
            os.system("winget install Gyan.FFmpeg")
            ffmpeg_path = os.path.join(os.environ["PROGRAMFILES"], "FFmpeg", "bin")
            os.environ["PATH"] += os.pathsep + ffmpeg_path
        else:
            print("FFmpeg is already installed.")
    except Exception as e:
        print(f"FFmpeg setup failed: {e}")

# 2. Initialize Models
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Whisper with fallbacks
    for model_size in ["base", "small"]:
        try:
            print(f"Loading Whisper {model_size}...")
            model = whisper.load_model(model_size, device=device)
            break
        except:
            continue
    else:
        model = whisper.load_model("tiny", device="cpu")

    # BART summarizer
    try:
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=0 if device == "cuda" else -1,
            framework="pt"
        )
    except:
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=-1,
            framework="pt"
        )

    return model, summarizer

# 3. Processing Functions
def transcribe_audio(audio_path):
    try:
        if not os.path.exists(audio_path):
            return "Error: File not found"
        
        # Verify audio file
        try:
            import wave
            with wave.open(audio_path, 'rb') as f:
                pass
        except Exception as e:
            return f"Error: Invalid audio file format ({str(e)})"
        
        # Use absolute path for FFmpeg if necessary
        ffmpeg_path = r"C:\Program Files\FFmpeg\bin\ffmpeg.exe"
        os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)
        
        result = whisper_model.transcribe(audio_path, fp16=(device=="cuda"))
        return result["text"]
    except Exception as e:
        return f"Transcription Error: {str(e)}"

def summarize_text(text):
    if not text or "Error" in text:
        return "Cannot summarize - invalid input"
    
    try:
        summary = summarizer(
            text,
            max_length=150,
            min_length=30,
            do_sample=False
        )
        return summary[0]['summary_text']
    except Exception as e:
        return f"Summarization Error: {str(e)}"

def save_summary(text, format="txt"):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"summary_{timestamp}"
        
        if format == "txt":
            with open(f"{filename}.txt", "w", encoding="utf-8") as f:
                f.write(text)
        elif format == "docx":
            doc = Document()
            doc.add_paragraph(text)
            doc.save(f"{filename}.docx")
            
        return f"{filename}.{format}"
    except Exception as e:
        print(f"File save error: {e}")
        return ""

# 4. Initialize
configure_system()
whisper_model, summarizer = load_models()
device = "cuda" if torch.cuda.is_available() else "cpu"

# 5. Gradio Interface
with gr.Blocks(title="Lecture Summarizer") as app:
    gr.Markdown("# 🎤 AI Lecture Summarizer")
    
    with gr.Row():
        audio_input = gr.Audio(
            type="filepath",  # Ensures Gradio passes the file path
            label="Upload Audio/Video"
        )
        
    with gr.Row():
        with gr.Column():
            export_format = gr.Radio(
                ["txt", "docx"],
                value="txt",
                label="Output Format"
            )
            submit_btn = gr.Button("Process", variant="primary")
        
        with gr.Column():
            transcript_out = gr.Textbox(label="Transcript", lines=5)
            summary_out = gr.Textbox(label="Summary", lines=5)
            download_out = gr.File(label="Download")
    
    def process(audio_file, format):
        if not audio_file:
            return "No file uploaded", "No file uploaded", None
        
        # Use the provided file path directly
        audio_path = audio_file if isinstance(audio_file, str) else audio_file.name
        
        # Check if the file exists
        if not os.path.exists(audio_path):
            return f"Error: File not found at {audio_path}", None, None
        
        # Transcribe and summarize
        transcript = transcribe_audio(audio_path)
        summary = summarize_text(transcript)

        # Save summary if transcription succeeded
        if "Error" not in transcript:
            saved_file = save_summary(summary, format)
        else:
            saved_file = None

        return transcript, summary, saved_file

    submit_btn.click(
        fn=process,
        inputs=[audio_input, export_format],
        outputs=[transcript_out, summary_out, download_out]
    )

if __name__ == "__main__":
    app.launch(server_port=7860, share=True)
