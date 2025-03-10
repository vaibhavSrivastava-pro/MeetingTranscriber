import os
import subprocess
import tempfile
import torch
import numpy as np
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
from pydub import AudioSegment
import argparse

def extract_audio(video_path, output_audio_path):
    """Extract audio from video file using ffmpeg"""
    print(f"Extracting audio from {video_path}...")
    command = [
        "ffmpeg", "-i", video_path, 
        "-ac", "1", "-ar", "16000",  # Mono audio at 16kHz (optimal for speech recognition)
        "-vn", "-f", "wav", output_audio_path,
        "-y"  # Overwrite output file if it exists
    ]
    subprocess.run(command, check=True)
    print(f"Audio extracted to {output_audio_path}")
    return output_audio_path

# def transcribe_audio(audio_path, model_size="medium", language="en", compute_type="float16"):
#     """Transcribe audio file using faster-whisper with timestamps"""
#     print(f"Transcribing audio using faster-whisper ({model_size} model)...")
    
#     # Set device based on availability
#     if torch.cuda.is_available():
#         device = "cuda"
#         compute_type = compute_type
#     else:
#         device = "cpu"
#         compute_type = "float32"
    
#     # Load the model
#     model = WhisperModel(
#         model_size, 
#         device=device, 
#         compute_type=compute_type,
#         download_root="./models"
#     )
    
#     # Transcribe the audio file
#     segments, info = model.transcribe(
#         audio_path, 
#         language=language,
#         vad_filter=True,  # Filter out non-speech segments
#         word_timestamps=True  # Get timestamps for each word
#     )
    
#     print(f"Detected language: {info.language} with probability {info.language_probability:.2f}")
    
#     # Collect segments with timestamps
#     transcription_data = []
#     for segment in segments:
#         transcription_data.append({
#             'start': segment.start,
#             'end': segment.end,
#             'text': segment.text,
#             'words': segment.words
#         })
    
#     return transcription_data

def transcribe_audio(audio_path, model_size="medium", language="en", compute_type="float16", chunk_duration=30):
    """Transcribe audio file using faster-whisper with timestamps, processing in chunks to avoid hanging"""
    print(f"Transcribing audio using faster-whisper ({model_size} model)...")
    
    # Set device based on availability
    if torch.cuda.is_available():
        device = "cuda"
        compute_type = compute_type
        print(f"Using CUDA with compute_type={compute_type}")
    else:
        device = "cpu"
        compute_type = "float32"
        print("Using CPU for transcription")
    
    try:
        # Load the model
        print(f"Loading faster-whisper model: {model_size}")
        model = WhisperModel(
            model_size, 
            device=device, 
            compute_type=compute_type,
            download_root="./models"
        )
        
        print(f"Model loaded successfully")
        
        # Split audio into manageable chunks to avoid issues with long recordings
        print(f"Loading audio file: {audio_path}")
        audio = AudioSegment.from_file(audio_path)
        duration_ms = len(audio)
        chunk_size_ms = chunk_duration * 1000  # Convert seconds to milliseconds
        
        print(f"Audio duration: {duration_ms/1000:.2f} seconds, processing in {chunk_duration}s chunks")
        
        # Process audio in chunks
        transcription_data = []
        for start_ms in range(0, duration_ms, chunk_size_ms):
            end_ms = min(start_ms + chunk_size_ms, duration_ms)
            print(f"Processing chunk {start_ms/1000:.2f}s to {end_ms/1000:.2f}s...")
            
            # Extract chunk and save to temporary file
            chunk = audio[start_ms:end_ms]
            temp_chunk_path = "temp_chunk.wav"
            chunk.export(temp_chunk_path, format="wav")
            
            # Transcribe this chunk
            result = model.transcribe(
                temp_chunk_path,
                language=language,
                vad_filter=True,
                word_timestamps=True
            )
            
            # Store language info from first chunk
            if start_ms == 0:
                print(f"Detected language: {result[1].language} with probability {result[1].language_probability:.2f}")
            
            # Process segments from this chunk manually
            chunk_segments = list(result[0])  # Force evaluation of iterator
            print(f"  Found {len(chunk_segments)} segments in this chunk")
            
            # Process the segments from this chunk
            for segment in chunk_segments:
                # Adjust timestamps to account for chunk position
                adjusted_start = segment.start + (start_ms / 1000)
                adjusted_end = segment.end + (start_ms / 1000)
                
                # Adjust word timestamps if they exist
                adjusted_words = []
                if hasattr(segment, 'words') and segment.words:
                    for word in segment.words:
                        adjusted_word = {
                            'word': word.word,
                            'start': word.start + (start_ms / 1000),
                            'end': word.end + (start_ms / 1000),
                            'probability': word.probability
                        }
                        adjusted_words.append(adjusted_word)
                
                # Add segment to results
                transcription_data.append({
                    'start': adjusted_start,
                    'end': adjusted_end,
                    'text': segment.text,
                    'words': adjusted_words
                })
            
            # Clean up temp file
            if os.path.exists(temp_chunk_path):
                os.remove(temp_chunk_path)
        
        print(f"Transcription complete. Extracted {len(transcription_data)} segments total.")
        return transcription_data
        
    except Exception as e:
        import traceback
        print(f"Error during transcription: {str(e)}")
        traceback.print_exc()
        return []

def perform_diarization(audio_path, num_speakers=None):
    """Perform speaker diarization to identify different speakers with improved error handling"""
    print("Starting speaker diarization...")
    
    # Check if HF_TOKEN is set
    if "HF_TOKEN" not in os.environ:
        print("ERROR: HF_TOKEN environment variable not set. Please set your Hugging Face token.")
        print("Example: export HF_TOKEN=your_token_here")
        return None
    
    try:
        # Initialize the pipeline
        print("Loading diarization model...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=os.environ["HF_TOKEN"]
        )
        
        # Set the pipeline to run on GPU if available
        if torch.cuda.is_available():
            print("Using GPU for diarization")
            pipeline = pipeline.to(torch.device("cuda"))
        else:
            print("Using CPU for diarization (this may be slow)")
        
        # Apply the pipeline to the audio file
        print(f"Processing audio file: {audio_path}")
        if num_speakers is not None:
            print(f"Using known number of speakers: {num_speakers}")
            diarization = pipeline(audio_path, num_speakers=num_speakers)
        else:
            print("Automatically detecting number of speakers")
            diarization = pipeline(audio_path)
        
        # Extract speaker turns
        speaker_turns = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_turns.append({
                'speaker': speaker,
                'start': turn.start,
                'end': turn.end
            })
        
        print(f"Diarization complete. Found {len(set(turn['speaker'] for turn in speaker_turns))} speakers.")
        return speaker_turns
    
    except ImportError as e:
        print(f"ERROR: Failed to import required modules for diarization: {str(e)}")
        print("Make sure you've installed: pip install pyannote.audio")
        return None
    except ValueError as e:
        if "HuggingFace hub token is invalid" in str(e):
            print("ERROR: Your Hugging Face token is invalid or you don't have access to the model.")
            print("Please visit https://huggingface.co/pyannote/speaker-diarization-3.1 and accept the user agreement.")
        else:
            print(f"ERROR: ValueError during diarization: {str(e)}")
        return None
    except Exception as e:
        print(f"ERROR: Unexpected error during diarization: {str(e)}")
        if "CUDA out of memory" in str(e):
            print("Try processing a shorter audio file or use CPU by setting: os.environ['CUDA_VISIBLE_DEVICES']=''")
        return None

def assign_speakers_to_segments(transcription_data, speaker_turns):
    """Assign speakers to transcription segments based on overlap"""
    print("Assigning speakers to transcribed segments...")
    
    for segment in transcription_data:
        segment_start = segment['start']
        segment_end = segment['end']
        
        # Find the speaker with maximum overlap
        max_overlap = 0
        assigned_speaker = None
        
        for turn in speaker_turns:
            # Calculate overlap duration
            overlap_start = max(segment_start, turn['start'])
            overlap_end = min(segment_end, turn['end'])
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > max_overlap:
                max_overlap = overlap
                assigned_speaker = turn['speaker']
        
        # Assign the speaker with maximum overlap
        segment['speaker'] = assigned_speaker if max_overlap > 0 else "Unknown"
    
    return transcription_data

def format_transcript_with_speakers(segments_with_speakers, output_path):
    """Format and write the transcript with speaker labels to a text file"""
    print(f"Writing formatted transcript to {output_path}...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        current_speaker = None
        
        for segment in segments_with_speakers:
            # Add speaker change indicator
            if segment['speaker'] != current_speaker:
                current_speaker = segment['speaker']
                f.write(f"\n[{current_speaker}]\n")
            
            # Write segment text
            f.write(f"{segment['text'].strip()}\n")
    
    print("Transcript with speaker diarization completed!")
    return output_path

def process_video(video_path, output_path, model_size="medium", language="en", num_speakers=None):
    """Process video through the entire pipeline with better error handling"""
    try:
        # Create a temporary directory for intermediate files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract audio from video
            audio_path = os.path.join(temp_dir, "audio.wav")
            extract_audio(video_path, audio_path)
            
            # Transcribe audio
            print("\n=== STEP 2: TRANSCRIPTION ===")
            transcription_data = transcribe_audio(
                audio_path, 
                model_size=model_size, 
                language=language
            )
            
            if not transcription_data:
                print("Transcription failed or returned no segments.")
                return False
                
            print(f"Transcription completed successfully with {len(transcription_data)} segments.")
            
            # Save transcription without speakers in case diarization fails
            with open(output_path + ".no_speakers.txt", 'w', encoding='utf-8') as f:
                for segment in transcription_data:
                    f.write(f"{segment['start']:.2f} - {segment['end']:.2f}: {segment['text'].strip()}\n")
            
            # Perform speaker diarization
            print("\n=== STEP 3: SPEAKER DIARIZATION ===")
            speaker_turns = perform_diarization(audio_path, num_speakers)
            
            if not speaker_turns:
                print("Diarization failed. Saving transcript without speaker information.")
                return False
            
            # Assign speakers to transcription segments
            print("\n=== STEP 4: SPEAKER ASSIGNMENT ===")
            segments_with_speakers = assign_speakers_to_segments(transcription_data, speaker_turns)
            
            # Format and save transcript with speakers
            print("\n=== STEP 5: FORMAT OUTPUT ===")
            format_transcript_with_speakers(segments_with_speakers, output_path)
            
            print(f"Process completed successfully. Output saved to {output_path}")
            return True
            
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video to Text Transcription with Speaker Diarization")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output", default="transcript.txt", help="Path for output transcript file")
    parser.add_argument("--model", default="medium", choices=["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"], 
                        help="Whisper model size")
    parser.add_argument("--language", default="en", help="Language code (e.g., 'en' for English)")
    parser.add_argument("--speakers", type=int, default=None, help="Number of speakers (if known)")
    
    args = parser.parse_args()
    
    process_video(
        args.video,
        args.output,
        model_size=args.model,
        language=args.language,
        num_speakers=args.speakers
    )