pip install torch torchaudio faster-whisper pyannote.audio pydub numpy

# MAC
brew install ffmpeg

# Windows
Download from ffmpeg.org and add to PATH

# Set hugging face access token as an environment variable (e.g. via the Terminal app on Mac):
export HF_TOKEN=your_token

# NOTE:
1. visit hf.co/pyannote/speaker-diarization and accept user conditions
2. visit hf.co/pyannote/segmentation and accept user conditions

# Run
python app.py --video input_video.mp4 --output transcript.txt --model medium
