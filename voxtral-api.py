#!/usr/bin/env python3
"""
Voxtral API Transcription Script
Transcribes audio/video files using Mistral's Voxtral API with segment timestamps
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import tempfile
import subprocess
import wave
import warnings
import time
import re
from dataclasses import dataclass

# Suppress pydub warnings about regex escape sequences in Python 3.12+
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pydub")

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Look for .env file in the script's directory
    script_dir = Path(__file__).parent
    env_file = script_dir / '.env'
    if env_file.exists():
        load_dotenv(env_file)
        print(f"Loaded environment variables from {env_file}")
except ImportError:
    # python-dotenv not installed, continue without it
    pass

# Third-party imports
try:
    from pydub import AudioSegment
    from pydub.silence import detect_silence
except ImportError:
    print("Please install required packages:")
    print("pip install pydub")
    sys.exit(1)

# Configuration
MISTRAL_API_KEY = os.environ.get('MISTRAL_API_KEY')
if not MISTRAL_API_KEY:
    print("Error: MISTRAL_API_KEY environment variable not set")
    print("\nPlease set it using one of these methods:")
    print("1. Create a .env file with: MISTRAL_API_KEY=your-api-key-here")
    print("2. Export it: export MISTRAL_API_KEY='your-api-key-here'")
    print("\nSee README.md for detailed setup instructions.")
    sys.exit(1)

# Audio configuration
TARGET_SAMPLE_RATE = 16000  # 16kHz
TARGET_CHANNELS = 1  # Mono
TARGET_SAMPLE_WIDTH = 2  # 16-bit (2 bytes)
MAX_CHUNK_DURATION_MS = 15 * 60 * 1000  # 15 minutes in milliseconds
MIN_SILENCE_LENGTH_MS = 1000  # 1 second of silence
SILENCE_THRESH_DB = -40  # Silence threshold in dB

# Supported languages by Voxtral API (based on documentation)
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'es': 'Spanish', 
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'hi': 'Hindi',
    'nl': 'Dutch',
    'zh': 'Chinese',
    'ja': 'Japanese',
    'ko': 'Korean',
    'ru': 'Russian',
    'ar': 'Arabic',
    'tr': 'Turkish',
    'pl': 'Polish'
}

# Language code mapping (includes variations and full names)
LANGUAGE_CODES = {
    'english': 'en', 'en': 'en',
    'spanish': 'es', 'es': 'es', 
    'french': 'fr', 'fr': 'fr',
    'german': 'de', 'de': 'de',
    'italian': 'it', 'it': 'it',
    'portuguese': 'pt', 'pt': 'pt',
    'hindi': 'hi', 'hi': 'hi',
    'dutch': 'nl', 'nl': 'nl',
    'chinese': 'zh', 'zh': 'zh', 'mandarin': 'zh', 'cantonese': 'zh-HK',
    'japanese': 'ja', 'ja': 'ja',
    'korean': 'ko', 'ko': 'ko',
    'russian': 'ru', 'ru': 'ru',
    'arabic': 'ar', 'ar': 'ar',
    'turkish': 'tr', 'tr': 'tr',
    'polish': 'pl', 'pl': 'pl',
    'swedish': 'sv', 'sv': 'sv',
    'norwegian': 'no', 'no': 'no',
    'danish': 'da', 'da': 'da',
    'finnish': 'fi', 'fi': 'fi',
    'greek': 'el', 'el': 'el',
    'hebrew': 'he', 'he': 'he',
    'czech': 'cs', 'cs': 'cs',
    'hungarian': 'hu', 'hu': 'hu',
    'romanian': 'ro', 'ro': 'ro',
    'bulgarian': 'bg', 'bg': 'bg',
    'croatian': 'hr', 'hr': 'hr',
    'serbian': 'sr', 'sr': 'sr',
    'slovak': 'sk', 'sk': 'sk',
    'slovenian': 'sl', 'sl': 'sl',
    'ukrainian': 'uk', 'uk': 'uk',
    'vietnamese': 'vi', 'vi': 'vi',
    'thai': 'th', 'th': 'th',
    'indonesian': 'id', 'id': 'id',
    'malay': 'ms', 'ms': 'ms',
    'filipino': 'fil', 'fil': 'fil',
    'swahili': 'sw', 'sw': 'sw'
}


@dataclass
class UsageStats:
    """Track usage statistics across all chunks"""
    prompt_audio_seconds: float = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    chunks_processed: int = 0


def check_wav_format(wav_file: str) -> bool:
    """
    Check if a WAV file is already in the correct format (PCM 16-bit, 16kHz, mono)
    """
    try:
        with wave.open(wav_file, 'rb') as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            frame_rate = wav.getframerate()
            
            return (channels == TARGET_CHANNELS and 
                   sample_width == TARGET_SAMPLE_WIDTH and 
                   frame_rate == TARGET_SAMPLE_RATE)
    except:
        return False


def extract_audio_to_wav(input_file: str) -> str:
    """
    Extract audio from any media file and convert to WAV PCM 16-bit 16kHz mono
    Returns path to the WAV file
    """
    input_path = Path(input_file)
    temp_dir = Path(tempfile.gettempdir())
    
    # Create a safe filename for the temporary file (avoid Unicode issues)
    safe_name = "".join(c if c.isascii() and c.isalnum() else "_" for c in input_path.stem)
    if not safe_name:
        safe_name = "audio"
    
    output_wav = temp_dir / f"{safe_name}_audio.wav"
    
    # Check if input is already WAV in correct format
    if input_path.suffix.lower() == '.wav' and check_wav_format(input_file):
        print(f"Input is already in correct WAV format: {input_file}")
        # Copy to temp location for consistency
        import shutil
        shutil.copy2(input_file, output_wav)
        return str(output_wav)
    
    print(f"Converting audio to WAV PCM 16-bit 16kHz mono from {input_file}...")
    
    # Use ffmpeg to extract and convert audio to WAV
    cmd = [
        'ffmpeg', '-i', str(input_file),
        '-vn',  # No video
        '-acodec', 'pcm_s16le',  # PCM 16-bit little-endian
        '-ar', str(TARGET_SAMPLE_RATE),  # 16kHz sample rate
        '-ac', str(TARGET_CHANNELS),  # Mono
        '-y',  # Overwrite output
        str(output_wav)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Audio converted to: {output_wav}")
        
        # Verify the output format
        if not check_wav_format(str(output_wav)):
            print("Warning: Converted file may not be in the expected format")
        
        return str(output_wav)
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e.stderr}")
        print("Make sure ffmpeg is installed: sudo apt-get install ffmpeg")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please install ffmpeg first.")
        print("Ubuntu/Debian: sudo apt-get install ffmpeg")
        print("macOS: brew install ffmpeg")
        print("Windows: Download from https://ffmpeg.org/download.html")
        sys.exit(1)


def find_silence_splits(audio: AudioSegment, max_duration_ms: int) -> List[int]:
    """
    Find optimal points to split audio based on silence detection
    Returns list of timestamps (in ms) where to split
    """
    duration_ms = len(audio)
    
    if duration_ms <= max_duration_ms:
        return []  # No need to split
    
    print("Detecting silence for optimal splitting...")
    
    # Detect all silence periods
    silence_ranges = detect_silence(
        audio, 
        min_silence_len=MIN_SILENCE_LENGTH_MS,
        silence_thresh=SILENCE_THRESH_DB
    )
    
    split_points = []
    current_position = 0
    
    while current_position + max_duration_ms < duration_ms:
        # Find target position
        target_position = current_position + max_duration_ms
        
        # Look for silence around target position (±30 seconds)
        search_start = max(target_position - 30000, current_position + 60000)  # At least 1 min from last split
        search_end = min(target_position + 30000, duration_ms)
        
        best_silence = None
        min_distance = float('inf')
        
        for silence_start, silence_end in silence_ranges:
            silence_middle = (silence_start + silence_end) / 2
            
            if search_start <= silence_middle <= search_end:
                distance = abs(silence_middle - target_position)
                if distance < min_distance:
                    min_distance = distance
                    best_silence = silence_middle
        
        if best_silence:
            split_points.append(int(best_silence))
            current_position = int(best_silence)
        else:
            # No silence found, force split at target
            split_points.append(target_position)
            current_position = target_position
    
    return split_points


def split_audio_intelligently(wav_file: str) -> List[Tuple[str, float, float]]:
    """
    Split audio into chunks at silence points
    Returns list of (chunk_file_path, start_time_seconds, end_time_seconds)
    """
    print(f"Loading audio file: {wav_file}")
    audio = AudioSegment.from_wav(wav_file)
    duration_ms = len(audio)
    
    if duration_ms <= MAX_CHUNK_DURATION_MS:
        print("Audio is shorter than 15 minutes, no splitting needed")
        return [(wav_file, 0.0, duration_ms / 1000.0)]
    
    print(f"Audio duration: {duration_ms / 1000:.1f} seconds, splitting needed")
    
    # Find split points
    split_points = find_silence_splits(audio, MAX_CHUNK_DURATION_MS)
    
    # Create chunks
    chunks = []
    start_ms = 0
    temp_dir = Path(tempfile.gettempdir())
    
    # Use safe filename for chunks
    base_name = Path(wav_file).stem
    safe_base = "".join(c if c.isascii() and c.isalnum() else "_" for c in base_name)
    if not safe_base:
        safe_base = "audio"
    
    for i, split_ms in enumerate(split_points + [duration_ms]):
        chunk_audio = audio[start_ms:split_ms]
        chunk_file = temp_dir / f"{safe_base}_chunk_{i:03d}.wav"
        
        # Export as WAV with same parameters
        chunk_audio.export(
            str(chunk_file), 
            format="wav",
            parameters=["-acodec", "pcm_s16le", "-ar", str(TARGET_SAMPLE_RATE), "-ac", str(TARGET_CHANNELS)]
        )
        chunks.append((str(chunk_file), start_ms / 1000.0, split_ms / 1000.0))
        
        print(f"Created chunk {i}: {start_ms/1000:.1f}s - {split_ms/1000:.1f}s ({(split_ms-start_ms)/1000:.1f}s)")
        start_ms = split_ms
    
    return chunks


def transcribe_chunk(audio_file: str, chunk_index: int = 0, language: str = None, max_retries: int = 3) -> Dict:
    """
    Transcribe a single audio chunk using Voxtral API via curl command
    
    Args:
        audio_file: Path to the audio file to transcribe
        chunk_index: Index of the current chunk (0-based)
        language: Language code (e.g., 'en', 'zh', 'fr') or None for auto-detection
        max_retries: Maximum number of retry attempts
    """
    # Display user-friendly chunk number (1-based)
    display_chunk_num = chunk_index + 1
    print(f"Transcribing chunk {display_chunk_num}...")
    if language:
        print(f"Using specified language: {language}")
    
    for attempt in range(max_retries):
        try:
            # Build the curl command
            curl_cmd = [
                'curl',
                '--location', 'https://api.mistral.ai/v1/audio/transcriptions',
                '--header', f'x-api-key: {MISTRAL_API_KEY}',
                '--form', f'file=@"{audio_file}"',
                '--form', 'model="voxtral-mini-latest"',
                '--form', 'timestamp_granularities="segment"',
                '--form', 'temperature="0.0"'
            ]
            
            # Add language if specified
            if language:
                curl_cmd.extend(['--form', f'language="{language}"'])
            
            # Execute curl command
            print(f"Sending transcription request for chunk {display_chunk_num}...")
            process = subprocess.run(
                curl_cmd,
                capture_output=True,
                text=True,
                check=False  # Don't raise on non-zero exit code
            )
            
            # Check for curl errors
            if process.returncode != 0:
                error_msg = f"Curl command failed with return code {process.returncode}"
                if process.stderr:
                    error_msg += f": {process.stderr}"
                raise Exception(error_msg)
            
            # Parse JSON response
            try:
                result = json.loads(process.stdout)
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON response: {e}")
                print(f"Raw response: {process.stdout[:500]}...")  # Show first 500 chars
                raise
            
            # Check for API errors in the response
            if 'error' in result:
                error_msg = result.get('error', {}).get('message', 'Unknown API error')
                error_code = result.get('error', {}).get('code', 'unknown')
                
                # Check for rate limiting
                if error_code == 'rate_limit_exceeded' or '429' in str(error_code):
                    print("Rate limit detected. Waiting 60 seconds...")
                    time.sleep(60)
                    continue
                    
                raise Exception(f"API error: {error_msg}")
            
            # Debug: Check what fields are in the response
            if chunk_index == 0:  # Only print for first chunk to avoid spam
                if 'usage' in result:
                    print(f"Usage info: {result['usage']}")
                    
                if 'segments' in result:
                    print(f"Number of segments: {len(result.get('segments', []))}")
                    if result.get('segments'):
                        print(f"First segment sample: {result['segments'][0]}")
            
            print(f"Chunk {display_chunk_num} transcribed successfully")
            
            # Ensure segments is a list even if empty
            if 'segments' not in result:
                result['segments'] = []
            
            return result
            
        except Exception as e:
            error_str = str(e)
            print(f"Error on attempt {attempt + 1}/{max_retries} for chunk {display_chunk_num}: {error_str}")
            
            # Check for rate limiting in error message
            if "429" in error_str or "rate" in error_str.lower():
                print("Rate limit detected. Waiting 60 seconds...")
                time.sleep(60)
                continue
                
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            raise


def align_timestamps(segments: List[Dict], offset_seconds: float) -> List[Dict]:
    """
    Adjust timestamps in segments by adding an offset
    """
    aligned_segments = []
    for segment in segments:
        aligned_segment = segment.copy()
        aligned_segment['start'] = segment.get('start', 0) + offset_seconds
        aligned_segment['end'] = segment.get('end', 0) + offset_seconds
        aligned_segments.append(aligned_segment)
    
    return aligned_segments


def format_timestamp_srt(seconds: float) -> str:
    """
    Convert seconds to SRT timestamp format (HH:MM:SS,mmm)
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def create_srt_file(segments: List[Dict], output_file: str) -> None:
    """
    Create an SRT subtitle file from segments
    """
    print(f"Creating SRT file: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(segments, 1):
            start_time = format_timestamp_srt(segment.get('start', 0))
            end_time = format_timestamp_srt(segment.get('end', 0))
            text = segment.get('text', '').strip()
            
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{text}\n")
            f.write("\n")
    
    print(f"SRT file created: {output_file}")


def format_usage_stats(stats: UsageStats) -> str:
    """
    Format usage statistics for display
    """
    output = []
    output.append("=" * 50)
    output.append("USAGE STATISTICS")
    output.append("=" * 50)
    output.append(f"Chunks processed: {stats.chunks_processed}")
    output.append(f"Total audio duration: {stats.prompt_audio_seconds:.1f} seconds ({stats.prompt_audio_seconds/60:.1f} minutes)")
    output.append(f"Prompt tokens: {stats.prompt_tokens:,}")
    output.append(f"Completion tokens: {stats.completion_tokens:,}")
    output.append(f"Total tokens: {stats.total_tokens:,}")
    
    # Calculate approximate cost (using estimated rates - adjust as needed)
    # Voxtral pricing is typically per audio second/minute
    estimated_cost = (stats.prompt_audio_seconds / 60) * 0.001  # Example rate: $0.001 per minute
    output.append(f"Estimated cost: €{estimated_cost:.4f}")
    output.append("=" * 50)
    
    return "\n".join(output)


def main(input_file: str, specified_language: str = None):
    """
    Main function to orchestrate the transcription process
    
    Args:
        input_file: Path to the media file to transcribe
        specified_language: Language code specified by user (optional)
    """
    # Ensure proper Unicode handling in console output
    if sys.platform == 'win32':
        import locale
        locale.setlocale(locale.LC_ALL, '')
    
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"Error: File not found: {input_file}")
        sys.exit(1)
    
    print(f"Processing: {input_file}")
    
    # Initialize usage statistics
    usage_stats = UsageStats()
    
    # Step 1: Extract/convert audio to WAV PCM 16-bit 16kHz
    wav_file = extract_audio_to_wav(input_file)
    
    try:
        # Step 2: Split audio if needed
        chunks = split_audio_intelligently(wav_file)
        
        # Step 3: Transcribe each chunk
        all_segments = []
        full_transcript = ""
        detected_language = specified_language  # Use specified language if provided
        
        for i, (chunk_file, start_offset, end_offset) in enumerate(chunks):
            # Transcribe the chunk
            try:
                result = transcribe_chunk(chunk_file, i, language=specified_language)
                usage_stats.chunks_processed += 1
            except Exception as e:
                print(f"Failed to transcribe chunk {i}: {e}")
                print("Continuing with empty segments for this chunk...")
                result = {'text': '', 'segments': [], 'language': detected_language or 'en'}
            
            # Get segments first
            segments = result.get('segments', [])
            
            # Extract and accumulate usage statistics
            if 'usage' in result:
                usage = result['usage']
                usage_stats.prompt_audio_seconds += usage.get('prompt_audio_seconds', 0)
                usage_stats.prompt_tokens += usage.get('prompt_tokens', 0)
                usage_stats.completion_tokens += usage.get('completion_tokens', 0)
                usage_stats.total_tokens += usage.get('total_tokens', 0)
            
            # Extract language (from first chunk) if not specified by user
            if not specified_language and not detected_language:
                # Check for language in response
                result_language = result.get('language')
                
                if result_language and result_language != 'None':
                    detected_language = result_language
                    print(f"Auto-detected language: {detected_language}")
                else:
                    # Try to detect from the text content
                    text_content = result.get('text', '')
                    if text_content:
                        # Check for Chinese characters
                        if re.search(r'[\u4e00-\u9fff\u3400-\u4dbf]', text_content):
                            detected_language = 'zh'
                            print("Detected Chinese characters in transcript, setting language to 'zh'")
                        # Check for Japanese characters
                        elif re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text_content):
                            detected_language = 'ja'
                            print("Detected Japanese characters in transcript, setting language to 'ja'")
                        # Check for Korean characters
                        elif re.search(r'[\uac00-\ud7af\u1100-\u11ff]', text_content):
                            detected_language = 'ko'
                            print("Detected Korean characters in transcript, setting language to 'ko'")
                        # Check for Arabic characters
                        elif re.search(r'[\u0600-\u06ff\u0750-\u077f]', text_content):
                            detected_language = 'ar'
                            print("Detected Arabic characters in transcript, setting language to 'ar'")
                        # Check for Cyrillic
                        elif re.search(r'[\u0400-\u04ff]', text_content):
                            detected_language = 'ru'
                            print("Detected Cyrillic characters in transcript, setting language to 'ru'")
            
            # Align timestamps if needed
            if i > 0 and segments:  # Adjust timestamps for chunks after the first
                segments = align_timestamps(segments, start_offset)
            
            all_segments.extend(segments)
            
            # Update full transcript
            chunk_text = result.get('text', '')
            if chunk_text:
                full_transcript += " " + chunk_text
                print(f"Chunk {i} text length: {len(chunk_text)} characters")
            
            # Clean up chunk file if it's not the original
            if chunk_file != wav_file and os.path.exists(chunk_file):
                os.remove(chunk_file)
        
        # Step 4: Create SRT file
        # Handle case where language wasn't detected
        if not detected_language:
            # Last attempt: check if we have any text with recognizable characters
            if full_transcript:
                if re.search(r'[\u4e00-\u9fff\u3400-\u4dbf]', full_transcript):
                    detected_language = 'zh'
                    print("Detected Chinese in full transcript")
                else:
                    detected_language = 'unknown'
                    print("Warning: Could not detect language from transcript, using 'unknown'")
            else:
                detected_language = 'unknown'
                print("Warning: No transcript text available for language detection, using 'unknown'")
        
        # Get the appropriate language code
        if detected_language and detected_language.lower() in LANGUAGE_CODES:
            lang_code = LANGUAGE_CODES[detected_language.lower()]
        elif detected_language and len(detected_language) >= 2:
            lang_code = detected_language[:2].lower()
        else:
            lang_code = 'xx'  # ISO 639-1 code for unknown language
        
        output_srt = input_path.parent / f"{input_path.stem}.{lang_code}.srt"
        
        if all_segments:
            create_srt_file(all_segments, str(output_srt))
            
            # Print summary
            print("\n" + "="*50)
            print(f"Transcription complete!")
            if specified_language:
                print(f"Used specified language: {detected_language}")
            else:
                print(f"Auto-detected language: {detected_language if detected_language else 'unknown'}")
            print(f"Total segments: {len(all_segments)}")
            print(f"Output file: {output_srt}")
            
            # Print usage statistics
            print("\n" + format_usage_stats(usage_stats))
        else:
            print("\n" + "="*50)
            print("Warning: No segments were transcribed.")
            print("This might be due to API issues or empty audio.")
            print("="*50)
        
    finally:
        # Clean up temporary WAV if created
        if wav_file != input_file and os.path.exists(wav_file):
            os.remove(wav_file)
            print(f"Cleaned up temporary file: {wav_file}")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Transcribe audio/video files using Mistral's Voxtral API with segment timestamps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect language:
  python voxtral-api.py video.mp4
  
  # Specify Chinese language:
  python voxtral-api.py video.mp4 --language zh
  python voxtral-api.py video.mp4 -l zh
  
  # Specify English language:
  python voxtral-api.py audio.wav -l en
  
Officially supported languages by Voxtral:
  en=English, es=Spanish, fr=French, de=German,
  it=Italian, pt=Portuguese, hi=Hindi, nl=Dutch
  
Additional languages (may work with varying accuracy):
  zh=Chinese, ja=Japanese, ko=Korean, ru=Russian,
  ar=Arabic, pl=Polish, tr=Turkish, etc.
  
Audio format:
  The script converts all audio to WAV PCM 16-bit 16kHz mono
  for optimal compatibility with Voxtral API.
  
Environment:
  Make sure to set your MISTRAL_API_KEY environment variable:
  export MISTRAL_API_KEY='your-api-key-here'
  or create a .env file with: MISTRAL_API_KEY=your-api-key-here
        """
    )
    
    # Add arguments
    parser.add_argument(
        'media_file',
        help='Path to the media file to transcribe'
    )
    
    parser.add_argument(
        '-l', '--language',
        type=str,
        default=None,
        help='Language code (e.g., en, zh, ja, fr, de). Default: auto-detect'
    )
    
    # Show help if no arguments provided
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate language code if provided
    if args.language:
        # Check if it's a valid 2-letter code or in our language mapping
        if len(args.language) == 2:
            print(f"Using specified language code: {args.language}")
            if args.language in SUPPORTED_LANGUAGES:
                print(f"Language: {SUPPORTED_LANGUAGES[args.language]} (officially supported)")
            else:
                print(f"Note: Language '{args.language}' may not be officially supported by Voxtral")
        elif args.language.lower() in LANGUAGE_CODES:
            # Convert full name to code
            args.language = LANGUAGE_CODES[args.language.lower()]
            print(f"Using language code: {args.language}")
            if args.language in SUPPORTED_LANGUAGES:
                print(f"Language: {SUPPORTED_LANGUAGES[args.language]} (officially supported)")
        else:
            print(f"Warning: Unrecognized language '{args.language}', will pass to API as-is")
    else:
        print("Language: Auto-detect mode")
    
    # Run main function
    main(args.media_file, args.language)