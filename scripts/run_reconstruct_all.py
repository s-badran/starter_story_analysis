#!/usr/bin/env python3
"""
Run the reconstruct_diarization script across all transcripts referenced in `transcripts/index.json`.

This script is separate from the transcription flow: run it after `transcribe_videos.py` completes.
"""
import json
from pathlib import Path
import subprocess
import logging
import sys

ROOT = Path(__file__).resolve().parent.parent
TRANS_DIR = ROOT / "transcripts"
INDEX = TRANS_DIR / "index.json"
RECON = ROOT / "scripts" / "reconstruct_diarization.py"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

if not INDEX.exists():
    logging.error("Index file not found: %s", INDEX)
    sys.exit(2)

with open(INDEX) as f:
    index = json.load(f)

for key, entry in index.items():
    raw_file = None
    conv_file = None
    if entry.get('transcript_path'):
        tpath = Path(entry['transcript_path'])
        raw_file = tpath / f"{tpath.name}_raw.json"
        conv_file = tpath / f"{tpath.name}_conversation.json"
    elif entry.get('transcript_files') and entry['transcript_files'].get('raw'):
        raw_file = Path(entry['transcript_files'].get('raw'))
        conv_file = Path(entry['transcript_files'].get('conversation') or str(raw_file.with_name(f"{raw_file.stem}_conversation.json")))
        # if raw_file lives directly under TRANS_DIR as <key>_raw.json, move it into a per-video dir
        try:
            if raw_file.exists() and raw_file.parent.resolve() == TRANS_DIR.resolve() and raw_file.stem.endswith('_raw'):
                expected_key = raw_file.stem[: -4]
                if expected_key == key:
                    vid_dir = TRANS_DIR / key
                    vid_dir.mkdir(parents=True, exist_ok=True)
                    new_raw = vid_dir / raw_file.name
                    # move file
                    raw_file.replace(new_raw)
                    raw_file = new_raw
                    conv_file = vid_dir / f"{key}_conversation.json"
                    # update index entry to point to new directory
                    entry['transcript_path'] = str(vid_dir)
                    entry.setdefault('transcript_files', {})
                    entry['transcript_files']['raw'] = str(raw_file)
                    entry['transcript_files']['conversation'] = str(conv_file)
                    with open(INDEX, 'w') as f:
                        json.dump(index, f, indent=2)
                    logging.info('Moved raw transcript into %s', vid_dir)
        except Exception as exc:
            logging.warning('Failed to move raw file for %s: %s', key, exc)
    elif entry.get('transcript_file'):
        raw_file = Path(entry['transcript_file'])
        conv_file = raw_file.with_name(f"{raw_file.stem}_conversation.json")
    else:
        logging.debug("No transcript info for %s - skipping", key)
        continue

    if not raw_file.exists():
        logging.debug("Raw transcript missing for %s: %s", key, raw_file)
        continue

    if conv_file.exists():
        logging.info("Conversation already exists for %s: %s", key, conv_file)
        continue

    logging.info("Reconstructing %s -> %s", raw_file, conv_file)
    try:
        subprocess.run([sys.executable, str(RECON), str(raw_file), "--out", str(conv_file)], check=True)
    except Exception as exc:
        logging.warning("Failed to reconstruct for %s: %s", key, exc)

print("Done")
