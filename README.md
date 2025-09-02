# Voxtral API Transcription Tool - Online CURL Edition

A Python script that transcribes audio and video files using Mistral's Voxtral API with intelligent speech detection, timestamp preservation, and automatic subtitle generation optimized for long-form content.

## Features

- 🎥 **Multi-format Support**: Transcribe any audio/video format supported by FFmpeg
- 🌍 **Multi-language Support**: Auto-detect or specify from 30+ languages
- 📝 **SRT Subtitles**: Automatically generates timestamped subtitle files with segment-level timestamps
- 🎯 **Smart Speech Detection**: Uses Pyannote v3 segmentation model for accurate voice activity detection
- ⏱️ **Accurate Timestamps**: Intelligently splits at natural speech boundaries to preserve timing
- 🔄 **Intelligent Chunking**: Splits long audio at optimal points using VAD-informed boundaries
- 📊 **Usage Statistics**: Track API usage and estimated costs
- 🔁 **Retry Logic**: Built-in retry mechanism with exponential backoff for API resilience
- 🐛 **Debug Mode**: Save raw JSON responses for troubleshooting

## How It Works

### Advanced Speech Processing Pipeline

1. **Audio Extraction**: Converts input media to WAV PCM 16-bit 16kHz mono format
2. **Speech Detection**: Uses Pyannote v3 segmentation model to identify speech/non-speech regions
3. **Intelligent Splitting**: Finds optimal split points near 15-minute boundaries during non-speech segments to chunk longform content
4. **Transcription**: Sends audio chunks to Voxtral API
5. **Timestamp Alignment**: Adjusts timestamps for multi-chunk files to maintain sync across content lengths >15 minutes
6. **SRT Generation**: Creates and assembles a properly formatted subtitle file with accurate timing

This approach ensures clean splits at natural pauses while maintaining accurate timestamps throughout the entire transcription. Additionally, with chunk sizes of up to 15 minutes in length, greater dialogue context is maintained to achieve a higher transcription accuracy. 

## Prerequisites

- Python 3.10 or higher
- FFmpeg installed on your system
- Mistral API key with access to Voxtral API
- HuggingFace account with access to pyannote/segmentation-3.0 model

### Installing FFmpeg

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from [FFmpeg official website](https://ffmpeg.org/download.html)

## Installation

### Step 1: Set up HuggingFace Access (Required for Pyannote VAD)

1. **Create a HuggingFace account** at [huggingface.co](https://huggingface.co/join)

2. **Get your access token**:
   - Go to [HuggingFace Settings > Access Tokens](https://huggingface.co/settings/tokens)
   - Create a new token with "read" permissions
   - Copy the token

3. **Request access to the Pyannote model** (REQUIRED):
   - Visit [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   - Click "Agree and access repository" to accept the model license
   - Wait for approval (usually instant)
   - **Note: Without this step, the script will fail when trying to download the model**

### Step 2: Clone and Set Up the Repository

#### Using uv (Recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package installer and resolver.

1. **Install uv** (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Clone the repository:**
```bash
git clone https://github.com/HansBaumgaertner/voxtral-api
cd voxtral-api
```

3. **Create and activate virtual environment:**
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

4. **Install dependencies:**
```bash
uv pip install -r requirements.txt
```

#### Using pip (Alternative)

```bash
# Clone the repository
git clone https://github.com/HansBaumgaertner/voxtral-api
cd voxtral-api

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Configure API Keys

Create a `.env` file in the project directory with both required tokens:

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your keys:
MISTRAL_API_KEY='your-mistral-api-key-here'
HF_TOKEN='your-huggingface-token-here'
```

**Important**: Both keys are required. The script will not work without:
- `MISTRAL_API_KEY` for transcription
- `HF_TOKEN` for accessing the Pyannote VAD model

### Required Python Packages

```
# Core audio processing
pydub>=0.25.1
python-dotenv>=1.1.1

# Pyannote VAD dependencies
torch>=2.0.0
torchaudio>=2.0.0
numpy>=1.21.0
soundfile>=0.13.1
pyannote.audio>=3.0.0
```

## Usage

### Command Line Options

```bash
python voxtral-api.py [OPTIONS] media_file

Arguments:
  media_file              Path to the media file to transcribe

Options:
  -l, --language LANG     Language code (e.g., en, zh, ja, fr, de)
                         Default: auto-detect
  -d, --debug            Enable debug mode: save raw JSON per chunk
  -h, --help             Show help message and exit
```

### Basic Usage

```bash
# Auto-detect language
python voxtral-api.py path/to/video.mp4

# Specify language
python voxtral-api.py path/to/audio.wav --language en

# Using short option
python voxtral-api.py video.mkv -l zh

# Enable debug mode
python voxtral-api.py movie.mp4 --debug
```

### Examples

```bash
# Transcribe English video
python voxtral-api.py lecture.mp4 -l en

# Transcribe Chinese content with debug output
python voxtral-api.py documentary.mp4 -l zh --debug

# Auto-detect language for podcast
python voxtral-api.py podcast.mp3

# Transcribe French movie (intelligently handles long content)
python voxtral-api.py movie.mkv -l fr

# Debug problematic file
python voxtral-api.py problematic_video.mp4 -d
```

### Output

The script generates:
- **SRT subtitle file**: `{original_filename}.{language_code}.srt`
  - Example: `video.mp4` → `video.en.srt`
- **Debug JSON files** (if --debug / -d enabled): `{original_filename}.chunk{number}.json`
  - Example: `video.chunk01.json`, `video.chunk02.json`

## Supported Languages

### Officially Supported by Voxtral
- English (en)
- Spanish (es)
- French (fr)
- German (de)
- Italian (it)
- Portuguese (pt)
- Hindi (hi)
- Dutch (nl)
- Chinese (zh)
- Japanese (ja)
- Korean (ko)
- Russian (ru)
- Arabic (ar)
- Turkish (tr)
- Polish (pl)

### Additional Languages (Experimental)
Swedish, Norwegian, Danish, Finnish, Greek, Hebrew, Czech, Hungarian, Romanian, Bulgarian, Croatian, Serbian, Slovak, Slovenian, Ukrainian, Vietnamese, Thai, Indonesian, Malay, Filipino, Swahili

## Key Features

### Pyannote v3 VAD Integration
- **State-of-the-art VAD**: Uses Pyannote's segmentation-3.0 model for precise speech detection
- **Natural Split Points**: Finds optimal boundaries during non-speech segments
- **Configurable Parameters**: Fine-tuned for various content types

### Intelligent Chunking Strategy
- **15-minute chunks**: Splits long content at natural pauses near 15-minute boundaries
- **VAD-informed splits**: Uses ±3 minute windows around target boundaries to find silence
- **Fallback handling**: Forces splits if no suitable silence found (last resort)

### Timestamp Preservation
- **Original timing maintained**: No timestamp drift, provided there are no long gaps of non-speech content longer than 60 seconds (a fix for this currently being worked on)
- **Automatic alignment**: Multi-chunk timestamps adjusted seamlessly
- **Accurate SRT output**: Subtitles sync perfectly with original media

## API Usage and Costs

The script provides detailed usage statistics after each transcription:
- Total audio duration processed
- Token usage (prompt and completion)
- Number of chunks processed
- Estimated cost calculation

## Performance Characteristics

- **Chunk Size**: Up to 15 minutes per API call
- **Split Strategy**: VAD-informed boundaries for clean transitions
- **Processing Speed**: Depends on content length and API response time
- **Optimal for**: Long-form content like movies, lectures, podcasts

## Troubleshooting

### Common Issues

1. **HuggingFace Token Issues**
   - **Error**: "HF_TOKEN not found in environment variables"
   - **Solution**: Add `HF_TOKEN='your-token-here'` to `.env` file
   
2. **Pyannote Model Access Denied**
   - **Error**: "401 Unauthorized" when downloading model
   - **Solution**: Visit [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) and click "Agree and access repository"
   - **Note**: You must be logged in to HuggingFace and have accepted the model license

3. **FFmpeg not found**
   - Ensure FFmpeg is installed and in your PATH
   - Test with: `ffmpeg -version`

4. **Mistral API Key not set**
   - Check if MISTRAL_API_KEY is properly set in `.env`
   - Test with: `echo $MISTRAL_API_KEY`

5. **PyTorch/CUDA issues**
   - The script works with CPU-only PyTorch
   - For GPU acceleration, install appropriate CUDA version

6. **Rate limiting**
   - The script automatically handles rate limits with 60-second waits

7. **Model download on first run**
   - First execution will download the Pyannote model
   - Subsequent runs will use the cached model

### Debug Mode

Use the `--debug` flag to:
- Save raw JSON responses from each chunk
- See detailed VAD processing information
- Track chunk creation and split points
- Verify timestamp alignment

```bash
python voxtral-api.py problematic_file.mp4 --debug
```

## Technical Details

### Pyannote VAD Parameters
- **Model**: pyannote/segmentation-3.0
- **Min speech duration**: 0.8 seconds
- **Min silence duration**: 0.4 seconds
- **Window size**: ±3 minutes around split targets
- **Min gap after split**: 30 seconds

### Chunking Strategy
- **Max chunk duration**: 15 minutes
- **Split search window**: 6 minutes (±3 min from target)
- **Required non-speech gap**: 0.3 seconds minimum
- **Fallback**: Force split at target if no silence found

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

Areas for potential improvement:
- Alternative chunking strategies
- FFmpeg filter strategies

## License

Apache 2.0

## Acknowledgments

- Mistral AI for the Voxtral Transcription API
- FFmpeg for audio processing capabilities
- pydub for audio manipulation
- Pyannote team for the segmentation model
- HuggingFace for model hosting

## Support

For issues, questions, or suggestions, please open an issue on GitHub.

## Version History

- **v0.1.1**: Switched to Pyannote v3 VAD for improved chunking strategy
- **v0.1.0**: Initial release with basic chunking at silence points