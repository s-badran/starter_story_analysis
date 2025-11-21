#!/usr/bin/env python3
"""
Simple scaffold to download YouTube audio and transcribe with AssemblyAI Python SDK (`aai`).

Usage: set `ASSEMBLYAI_API_KEY` in env, put YouTube URLs in `videos_list.json`, then run.
"""
import os
import json
import subprocess
import logging
import shutil
from urllib.parse import urlparse, parse_qs
from pathlib import Path
from dotenv import load_dotenv
import requests
import time
from tqdm import tqdm
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
ROOT = Path(__file__).parent
DOWNLOAD_DIR = ROOT / "downloads"
OUT_DIR = ROOT / "transcripts"
DOWNLOAD_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)
INDEX_PATH = OUT_DIR / "index.json"

# retries/backoff config
DOWNLOAD_RETRIES = int(os.getenv("DOWNLOAD_RETRIES", "3"))
UPLOAD_RETRIES = int(os.getenv("UPLOAD_RETRIES", "3"))
RETRY_BACKOFF = float(os.getenv("RETRY_BACKOFF", "2"))

# limit processed videos (useful for testing channels)
MAX_VIDEOS = int(os.getenv("MAX_VIDEOS", "0"))  # 0 = no limit

# audio format to request from yt-dlp (use compressed by default to save upload time/cost)
AUDIO_FORMAT = os.getenv("AUDIO_FORMAT", "mp3")

# detect which yt-dlp binary will be used (system or pip-installed)
YT_DLP_PATH = shutil.which("yt-dlp")
if YT_DLP_PATH:
    try:
        ver = subprocess.run([YT_DLP_PATH, "--version"], check=True, capture_output=True, text=True)
        logging.info("Using yt-dlp at %s (version: %s)", YT_DLP_PATH, ver.stdout.strip())
    except Exception:
        logging.info("Using yt-dlp at %s", YT_DLP_PATH)
else:
    logging.warning("yt-dlp not found on PATH. Ensure yt-dlp is installed and on PATH.")

# load .env if present so users can keep API keys there
load_dotenv()
API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
if not API_KEY:
    raise SystemExit("Set ASSEMBLYAI_API_KEY environment variable before running.")

# AssemblyAI REST API base
API_BASE = "https://api.assemblyai.com/v2"
HEADERS = {"authorization": API_KEY}

# Polling / retry/backoff config (can be set via env vars)
POLL_INTERVAL = float(os.getenv("POLL_INTERVAL", "5"))
MAX_POLL_TRIES = int(os.getenv("MAX_POLL_TRIES", "120"))
BACKOFF_FACTOR = float(os.getenv("BACKOFF_FACTOR", "1.5"))
MAX_BACKOFF = float(os.getenv("MAX_BACKOFF", "60"))

# diarization config
ENABLE_DIARIZATION = os.getenv("ENABLE_DIARIZATION", "1") in ("1", "true", "True")


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
    # choose yt-dlp binary path if detected
    ytdlp_exec = YT_DLP_PATH or "yt-dlp"
    cmd = [
        ytdlp_exec,
        "-x",
        "--audio-format",
        AUDIO_FORMAT,
        "-o",
        out_template,
        url,
    ]
    logging.info("Downloading audio for %s (format=%s)", url, AUDIO_FORMAT)

    attempt = 0
    while attempt < DOWNLOAD_RETRIES:
        try:
            attempt += 1
            logging.info("Download attempt %d/%d for %s", attempt, DOWNLOAD_RETRIES, url)
            subprocess.run(cmd, check=True)
            # find the produced file
            for f in DOWNLOAD_DIR.iterdir():
                if f.name.startswith(vid + "."):
                    return str(f)
            raise FileNotFoundError("Downloaded file not found for video " + url)
        except Exception as exc:
            logging.warning("Download attempt %d failed for %s: %s", attempt, url, exc)
            if attempt >= DOWNLOAD_RETRIES:
                raise
            sleep_t = RETRY_BACKOFF * (2 ** (attempt - 1))
            logging.info("Retrying download after %.1fs...", sleep_t)
            time.sleep(sleep_t)


def fetch_channel_videos(channel_url: str) -> list:
    """Use yt-dlp to expand a channel /videos page into individual video URLs.

    Returns a list of full watch URLs.
    """
    logging.info("Fetching channel video list from %s", channel_url)
    cmd = ["yt-dlp", "--flat-playlist", "-J", channel_url]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        logging.error("yt-dlp failed to list channel: %s", exc)
        return []
    try:
        data = json.loads(proc.stdout)
    except Exception as exc:
        logging.error("Failed to parse yt-dlp output: %s", exc)
        return []
    entries = data.get("entries") or []
    urls = []
    for e in entries:
        vid = e.get("id") or e.get("url")
        if not vid:
            continue
        # normalize to watch URL
        if vid.startswith("http"):
            urls.append(vid)
        else:
            urls.append(f"https://www.youtube.com/watch?v={vid}")
    logging.info("Found %d videos on channel", len(urls))
    return urls


def get_video_metadata(url: str) -> dict:
    """Return metadata dict for a video URL using yt-dlp -J.

    Returns a dict (may contain 'title', 'duration', etc.) or {} on failure.
    """
    ytdlp_exec = YT_DLP_PATH or "yt-dlp"
    cmd = [ytdlp_exec, "-J", url]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        data = json.loads(proc.stdout)
        return data
    except Exception as exc:
        logging.warning("Failed to fetch metadata for %s: %s", url, exc)
        return {}


def transcribe_with_assemblyai(audio_path: str) -> dict:
    logging.info("Transcribing %s with AssemblyAI REST API", audio_path)

    def _upload_file(path: str) -> str:
        upload_url = f"{API_BASE}/upload"

        def _read_in_chunks(fh, chunk_size=5242880):
            while True:
                chunk = fh.read(chunk_size)
                if not chunk:
                    break
                yield chunk

        attempt = 0
        while attempt < UPLOAD_RETRIES:
            try:
                attempt += 1
                logging.info("Upload attempt %d/%d for %s", attempt, UPLOAD_RETRIES, path)
                with open(path, "rb") as fh:
                    resp = requests.post(upload_url, headers=HEADERS, data=_read_in_chunks(fh))
                resp.raise_for_status()
                return resp.json().get("upload_url")
            except Exception as exc:
                logging.warning("Upload attempt %d failed for %s: %s", attempt, path, exc)
                if attempt >= UPLOAD_RETRIES:
                    raise
                sleep_t = RETRY_BACKOFF * (2 ** (attempt - 1))
                logging.info("Retrying upload after %.1fs...", sleep_t)
                time.sleep(sleep_t)

    def _create_transcript(audio_url: str) -> str:
        endpoint = f"{API_BASE}/transcript"
        payload = {"audio_url": audio_url}
        # enable diarization/speaker labeling if requested
        if ENABLE_DIARIZATION:
            # AssemblyAI supports speaker labels; let the service auto-detect speaker count
            payload["speaker_labels"] = True
        attempt = 0
        while attempt < UPLOAD_RETRIES:
            try:
                attempt += 1
                logging.info("Create transcript attempt %d/%d for %s", attempt, UPLOAD_RETRIES, audio_url)
                resp = requests.post(endpoint, json=payload, headers=HEADERS)
                resp.raise_for_status()
                return resp.json().get("id")
            except Exception as exc:
                logging.warning("Create transcript attempt %d failed for %s: %s", attempt, audio_url, exc)
                if attempt >= UPLOAD_RETRIES:
                    raise
                sleep_t = RETRY_BACKOFF * (2 ** (attempt - 1))
                logging.info("Retrying create transcript after %.1fs...", sleep_t)
                time.sleep(sleep_t)

    def _poll_transcript(tid: str, interval: int = POLL_INTERVAL) -> dict:
        endpoint = f"{API_BASE}/transcript/{tid}"
        attempt = 0
        sleep_time = interval
        while attempt < MAX_POLL_TRIES:
            resp = requests.get(endpoint, headers=HEADERS)
            resp.raise_for_status()
            j = resp.json()
            status = j.get("status")
            if status == "completed":
                return j
            if status == "failed":
                raise RuntimeError(f"Transcription failed: {j}")
            # backoff before next poll
            attempt += 1
            sleep_time = min(interval * (BACKOFF_FACTOR ** attempt), MAX_BACKOFF)
            logging.debug("Poll attempt %d for %s, sleeping %.1fs", attempt, tid, sleep_time)
            time.sleep(sleep_time)
        raise RuntimeError(f"Max polling attempts ({MAX_POLL_TRIES}) exceeded for transcript {tid}")

    upload_url = _upload_file(audio_path)
    if not upload_url:
        raise RuntimeError("Upload failed, no upload_url returned")
    tid = _create_transcript(upload_url)
    result = _poll_transcript(tid)
    out = {
        "status": result.get("status"),
        "text": result.get("text"),
        "id": result.get("id"),
        "raw": result,
    }
    return out


def main():
    # load or initialize index
    if INDEX_PATH.exists():
        try:
            with open(INDEX_PATH, "r") as idxf:
                index = json.load(idxf)
        except Exception:
            index = {}
    else:
        index = {}

    # sync existing transcript files into the index (so we skip them)
    # sync existing transcript directories (new layout) and fallback to flat jsons
    for p in OUT_DIR.iterdir():
        # ignore the index file itself
        if p == INDEX_PATH:
            continue
        if p.is_dir():
            key = p.name
            entry = index.get(key, {})
            entry.setdefault("transcript_path", str(p))
            # detect raw transcript file inside
            raw_files = list(p.glob(f"{key}_raw.json"))
            conv_files = list(p.glob(f"{key}_conversation.json"))
            if raw_files:
                entry.setdefault("transcript_files", {})
                entry["transcript_files"].setdefault("raw", str(raw_files[0]))
                if conv_files:
                    entry["transcript_files"].setdefault("conversation", str(conv_files[0]))
                entry["status"] = "completed"
            index[key] = entry
        elif p.is_file() and p.suffix == ".json":
            # legacy flat json transcripts or raw transcripts named <video_id>_raw.json
            if p.name == INDEX_PATH.name:
                continue
            stem = p.stem
            key = stem
            # handle files named <id>_raw
            if stem.endswith("_raw"):
                key = stem[: -4]
                entry = index.get(key, {})
                entry.setdefault("transcript_files", {})
                entry["transcript_files"].setdefault("raw", str(p))
                entry.setdefault("status", "completed")
                index[key] = entry
            else:
                entry = index.get(key, {})
                entry.setdefault("transcript_file", str(p))
                entry.setdefault("url", entry.get("url", ""))
                entry["status"] = "completed"
                index[key] = entry
    # persist any updates from the scan
    with open(INDEX_PATH, "w") as idxf:
        json.dump(index, idxf, indent=2)

    with open(ROOT / "videos_list.json") as fh:
        urls = json.load(fh)
    # expand any channel /videos page entries using yt-dlp
    expanded = []
    for url in urls:
        if isinstance(url, str) and ("/videos" in url and ("@" in url or "/channel/" in url)):
            found = fetch_channel_videos(url)
            if found:
                expanded.extend(found)
            else:
                logging.warning("No videos expanded for channel URL %s", url)
        else:
            expanded.append(url)
    urls = expanded
    # process sequentially, skipping completed
    processed = 0
    for url in urls:
        if MAX_VIDEOS and processed >= MAX_VIDEOS:
            logging.info("Reached MAX_VIDEOS=%d, stopping.", MAX_VIDEOS)
            break
        try:
            vid = get_video_id(url) or None
            vid_key = vid or None
            # fetch metadata (title, duration) and store in index
            meta = get_video_metadata(url)
            title = meta.get("title")
            duration = meta.get("duration")
            # if we already have a completed transcript, skip
            if vid_key and vid_key in index and index[vid_key].get("status") == "completed":
                logging.info("Skipping %s (already transcribed)", vid_key)
                continue

            # initialize index entry
            temp_key = vid_key or url
            entry = index.get(temp_key, {})
            if title:
                entry["title"] = title
            if duration:
                entry["duration"] = duration
            entry.setdefault("url", url)
            entry.setdefault("created_at", datetime.utcnow().isoformat() + "Z")

            # if already completed (and file exists), skip; if index says completed but file missing, re-run
            if entry.get("status") == "completed" and entry.get("transcript_file") and Path(entry.get("transcript_file")).exists():
                logging.info("Skipping %s (already completed)", temp_key)
                processed += 1
                continue
            elif entry.get("status") == "completed":
                logging.info("Transcript marked completed in index but file missing; will reprocess %s", temp_key)
                # clear completed status so we re-run
                entry.pop("status", None)
                entry.pop("transcript_file", None)

            # DOWNLOAD STEP: only download if we don't already have audio_file
            try:
                if entry.get("audio_file") and Path(entry.get("audio_file")).exists():
                    audio = entry.get("audio_file")
                    logging.info("Using existing audio file for %s: %s", temp_key, audio)
                else:
                    entry["status"] = "downloading"
                    with open(INDEX_PATH, "w") as idxf:
                        json.dump(index, idxf, indent=2)
                    audio = download_audio(url)
                    entry["audio_file"] = str(audio)
                    entry["status"] = "downloaded"
                    with open(INDEX_PATH, "w") as idxf:
                        json.dump(index, idxf, indent=2)
            except Exception as exc:
                logging.exception("Download failed for %s: %s", url, exc)
                entry["status"] = "audio-download-failed"
                entry.setdefault("last_error", str(exc))
                index[temp_key] = entry
                with open(INDEX_PATH, "w") as idxf:
                    json.dump(index, idxf, indent=2)
                continue

            # TRANSCRIBE STEP: upload/create/poll. we retry the specific failing operations inside transcribe_with_assemblyai
            try:
                entry["status"] = "uploading"
                with open(INDEX_PATH, "w") as idxf:
                    json.dump(index, idxf, indent=2)
                out = transcribe_with_assemblyai(audio)
                # save raw transcript as a flat file transcripts/<video_id>_raw.json
                vid_name = (temp_key or out.get('id') or Path(audio).stem)
                raw_fname = f"{vid_name}_raw.json"
                raw_path = OUT_DIR / raw_fname
                with open(raw_path, "w") as of:
                    json.dump(out, of, indent=2)
                # conversation will be produced later by reconstruction step
                entry.setdefault("transcript_files", {})
                entry["transcript_files"]["raw"] = str(raw_path)
                entry["status"] = out.get("status") or "completed"
                entry["transcribed_at"] = datetime.utcnow().isoformat() + "Z"
                index[temp_key] = entry
                with open(INDEX_PATH, "w") as idxf:
                    json.dump(index, idxf, indent=2)
            except Exception as exc:
                logging.exception("Transcription failed for %s: %s", url, exc)
                # determine likely failure stage from exception context? mark as transcript-failed
                entry["status"] = "transcript-failed"
                entry.setdefault("last_error", str(exc))
                index[temp_key] = entry
                with open(INDEX_PATH, "w") as idxf:
                    json.dump(index, idxf, indent=2)
                continue
            processed += 1
        except Exception as exc:
            logging.exception("Failed to process %s: %s", url, exc)
            # mark as failed
            try:
                entry = index.get(vid_key or url, {})
                entry["status"] = "failed"
                entry.setdefault("last_error", str(exc))
                index[vid_key or url] = entry
                with open(INDEX_PATH, "w") as idxf:
                    json.dump(index, idxf, indent=2)
            except Exception:
                pass
    with open(OUT_DIR / "index.json", "w") as idxf:
        json.dump(index, idxf, indent=2)
    logging.info("Done. Transcripts in %s", OUT_DIR)

    # Note: reconstruction of conversation JSON is intentionally kept separate.
    # Run `scripts/run_reconstruct_all.py` after this script completes to build
    # per-video `_conversation.json` files from the raw transcripts.


if __name__ == "__main__":
    main()
