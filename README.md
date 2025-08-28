# Voxtral API Transcription Tool - CURL Version

A Python script that transcribes audio and video files using Mistral's Voxtral API with segment-level timestamps and automatic subtitle generation.

## Features

- 🎥 **Multi-format Support**: Transcribe any audio/video format supported by FFmpeg
- 🌍 **Multi-language Support**: Auto-detect or specify from 30+ languages
- 📝 **SRT Subtitles**: Automatically generates timestamped subtitle files with segment-level timestamps
- ⏱️ **Segment-level Timestamps**: Returns precise start and end times for each transcribed segment
- 🔄 **Smart Chunking**: For audio longer than 15 minutes, intelligently finds silence points around each 15-minute increment, makes separate API calls for each chunk, and automatically re-aligns timestamps in the final combined SRT file
- 📊 **Usage Statistics**: Track API usage and estimated costs
- 🔁 **Retry Logic**: Built-in retry mechanism with exponential backoff for API resilience

## Prerequisites

- Python 3.10 or higher
- FFmpeg installed on your system
- Mistral API key with access to Voxtral API

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

### Using uv (Recommended)

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

### Using pip (Alternative)

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

## Configuration

### Setting up the Mistral API Key

#### Option 1: Using .env file (Recommended)

1. Create a `.env` file in the project directory:
```bash
echo "MISTRAL_API_KEY=your-api-key-here" > .env
```

2. The script will automatically load the API key from the `.env` file.

#### Option 2: Using direnv (For automatic environment loading)

1. Install direnv:
```bash
# macOS
brew install direnv

# Ubuntu/Debian
sudo apt-get install direnv
```

2. Create `.envrc` file:
```bash
echo "export MISTRAL_API_KEY='your-api-key-here'" > .envrc
direnv allow
```

#### Option 3: Add to virtual environment activation script

Add to `.venv/bin/activate` (Linux/macOS) or `.venv\Scripts\activate.bat` (Windows):
```bash
export MISTRAL_API_KEY="your-api-key-here"
```

#### Option 4: Manual export (Temporary)
```bash
export MISTRAL_API_KEY="your-api-key-here"
```

## Usage

### Basic Usage

```bash
# Auto-detect language
python voxtral-api.py path/to/video.mp4

# Specify language
python voxtral-api.py path/to/audio.wav --language en

# Using short option
python voxtral-api.py video.mkv -l zh
```

### Examples

```bash
# Transcribe English video
python voxtral-api.py lecture.mp4 -l en

# Transcribe Chinese content
python voxtral-api.py documentary.mp4 -l zh

# Auto-detect language for podcast
python voxtral-api.py podcast.mp3

# Transcribe French audio
python voxtral-api.py interview.wav -l fr
```

### Output

The script generates an SRT subtitle file in the same directory as the input file:
- Format: `{original_filename}.{language_code}.srt`
- Example: `video.mp4` → `video.en.srt`

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

## How It Works

1. **Audio Extraction**: Converts input media to WAV PCM 16-bit 16kHz mono format
2. **Smart Chunking**: For files >15 minutes, intelligently splits at silence points
3. **Transcription**: Sends chunks to Voxtral API with retry logic
4. **Alignment**: Adjusts timestamps for multi-chunk transcriptions
5. **SRT Generation**: Creates properly formatted subtitle file

## API Usage and Costs

The script provides detailed usage statistics after each transcription:
- Total audio duration processed
- Token usage (prompt and completion)
- Number of chunks processed
- Estimated cost calculation

## Troubleshooting

### Common Issues

1. **FFmpeg not found**
   - Ensure FFmpeg is installed and in your PATH
   - Test with: `ffmpeg -version`

2. **API Key not set**
   - Check if MISTRAL_API_KEY is properly set
   - Test with: `echo $MISTRAL_API_KEY`

3. **Rate limiting**
   - The script automatically handles rate limits with 60-second waits

### Debug Mode

For verbose output, you can modify the script to increase logging or check the console output for detailed processing information.

## Performance Tips

- **Optimal file size**: Files under 15 minutes process in a single request
- **Language specification**: Specifying language can improve accuracy and speed
- **Audio quality**: Higher quality audio yields better transcription results
- **Silence detection**: Files with clear silence breaks chunk more efficiently

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

Apache 2.0

## Acknowledgments

- Mistral AI for the Voxtral API
- FFmpeg for audio processing capabilities
- pydub for audio manipulation

## Support

For issues, questions, or suggestions, please open an issue on GitHub.
