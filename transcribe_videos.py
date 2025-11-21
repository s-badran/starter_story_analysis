#!/usr/bin/env python3
"""
Simple scaffold to download YouTube audio and transcribe with AssemblyAI Python SDK (`aai`).

Usage: set `ASSEMBLYAI_API_KEY` in env, put YouTube URLs in `videos_list.json`, then run.
"""
import os
import json
import subprocess
import logging
from urllib.parse import urlparse, parse_qs
from pathlib import Path

try:
    import aai
except Exception:
    aai = None

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
ROOT = Path(__file__).parent
DOWNLOAD_DIR = ROOT / "downloads"
OUT_DIR = ROOT / "transcripts"
DOWNLOAD_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
if not API_KEY:
    raise SystemExit("Set ASSEMBLYAI_API_KEY environment variable before running.")

if aai is None:
    raise SystemExit("`aai` package not found. Install with `pip install aai`.")

# Configure SDK key
aai.settings.api_key = API_KEY


def get_video_id(url: str) -> str:
    if "youtube" in url or "youtu.be" in url:
        parsed = urlparse(url)
        if "youtube" in parsed.netloc:
            qs = parse_qs(parsed.query)
            return qs.get("v", [None])[0]
        else:
            # youtu.be short link
            return parsed.path.strip("/")
    # fallback: last path segment
    return url.rstrip("/\n").split("/")[-1]


def download_audio(url: str) -> str:
    vid = get_video_id(url) or "video"
    out_template = str(DOWNLOAD_DIR / f"{vid}.%(ext)s")
    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format",
        "wav",
        "-o",
        out_template,
        url,
    ]
    logging.info("Downloading audio for %s", url)
    subprocess.run(cmd, check=True)
    # find the produced file
    for f in DOWNLOAD_DIR.iterdir():
        if f.name.startswith(vid + "."):
            return str(f)
    raise FileNotFoundError("Downloaded file not found for video " + url)


def transcribe_with_assemblyai(audio_path: str) -> dict:
    logging.info("Transcribing %s with AssemblyAI SDK", audio_path)
    config = aai.TranscriptionConfig(speech_model=aai.SpeechModel.universal)
    transcriber = aai.Transcriber(config=config)
    # `transcribe` accepts a local path or URL according to SDK sample
    transcript = transcriber.transcribe(audio_path)
    if getattr(transcript, "status", None) == "error":
        raise RuntimeError(f"Transcription failed: {getattr(transcript, 'error', 'unknown')}")
    # Try to build a serializable result
    result = {
        "status": getattr(transcript, "status", None),
        "text": getattr(transcript, "text", None),
        "id": getattr(transcript, "id", None),
    }
    # attach raw if convertible
    try:
        raw = transcript.__dict__
        result["raw"] = raw
    except Exception:
        result["raw"] = None
    return result


def main():
    with open(ROOT / "videos_list.json") as fh:
        urls = json.load(fh)
    index = {}
    for url in urls:
        try:
            vid = get_video_id(url) or None
            audio = download_audio(url)
            out = transcribe_with_assemblyai(audio)
            vid_key = vid or out.get("id") or Path(audio).stem
            outpath = OUT_DIR / f"{vid_key}.json"
            with open(outpath, "w") as of:
                json.dump(out, of, indent=2)
            index[vid_key] = {"url": url, "transcript_file": str(outpath)}
        except Exception as exc:
            logging.exception("Failed to process %s: %s", url, exc)
    with open(OUT_DIR / "index.json", "w") as idxf:
        json.dump(index, idxf, indent=2)
    logging.info("Done. Transcripts in %s", OUT_DIR)


if __name__ == "__main__":
    main()
