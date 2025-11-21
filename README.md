# Starter Story Transcript Extractor

This small project downloads audio from YouTube videos listed in `videos_list.json` and transcribes them using AssemblyAI's Python SDK.

Quick start (Ubuntu 24.04):

1. Install system deps:

```bash
sudo apt update
sudo apt install -y ffmpeg python3.11-venv
```

2. Create and activate a venv:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

3. Set your AssemblyAI API key (do NOT commit this key):

```bash
export ASSEMBLYAI_API_KEY="<your_key_here>"
```

4. Edit `videos_list.json` to include Starter Story video URLs (one per entry), then run:

```bash
python transcribe_videos.py
```

Outputs:
- Downloaded audio files go to `downloads/` (created at runtime).
- Transcripts are saved to `transcripts/{video_id}.json` and `transcripts/index.json`.

Notes:
- The repository intentionally does not store your API key. Use environment variables.
- For local/no-cost transcription, see `whisper.cpp` or `faster-whisper` alternatives.
# starter_story_analysis
