#!/usr/bin/env python3
"""
Reconstruct a conversation from AssemblyAI transcript JSON.

Outputs:
 - Ordered list of segments: [{"speaker":"A","text":"..."}, ...]

Usage:
  python scripts/reconstruct_diarization.py transcripts/nsn94Ad47GY.json \
      --out transcripts/nsn94Ad47GY_conversation.json

If the transcript has `raw.utterances` it will be used. Otherwise it will attempt
to group contiguous `raw.words` entries with the same speaker.
"""
import argparse
import json
import os
import sys

def load_transcript(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_segments_from_utterances(raw):
    utts = raw.get('utterances') or []
    if not utts:
        return []
    # Ensure sorted by start
    utts_sorted = sorted(utts, key=lambda u: u.get('start', 0))
    segments = []
    for u in utts_sorted:
        speaker = u.get('speaker') or u.get('speaker_label') or 'UNKNOWN'
        text = u.get('text') or ''
        segments.append({'speaker': speaker, 'text': text, 'start': u.get('start'), 'end': u.get('end')})
    return segments


def build_segments_from_words(raw):
    words = raw.get('words') or []
    if not words:
        return []
    # Group contiguous words with same speaker
    segments = []
    cur_speaker = None
    cur_words = []
    cur_start = None
    cur_end = None
    for w in sorted(words, key=lambda x: x.get('start', 0)):
        sp = w.get('speaker') or w.get('speaker_label') or 'UNKNOWN'
        if cur_speaker is None:
            cur_speaker = sp
            cur_words = [w.get('text','')]
            cur_start = w.get('start')
            cur_end = w.get('end')
        elif sp == cur_speaker:
            cur_words.append(w.get('text',''))
            cur_end = w.get('end')
        else:
            segments.append({'speaker': cur_speaker, 'text': ' '.join(cur_words), 'start': cur_start, 'end': cur_end})
            cur_speaker = sp
            cur_words = [w.get('text','')]
            cur_start = w.get('start')
            cur_end = w.get('end')
    # final
    if cur_speaker is not None:
        segments.append({'speaker': cur_speaker, 'text': ' '.join(cur_words), 'start': cur_start, 'end': cur_end})
    return segments


# note: we intentionally do not aggregate per-speaker text here; the ordered
# `segments` structure preserves conversational turns and order.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('transcript', help='Path to transcript JSON (transcripts/{id}.json)')
    parser.add_argument('--out', '-o', help='Output path (JSON). If omitted prints to stdout')
    parser.add_argument('--mode', choices=['both','list','per_speaker'], default='both')
    args = parser.parse_args()

    if not os.path.exists(args.transcript):
        print('Transcript file not found:', args.transcript, file=sys.stderr)
        sys.exit(2)

    obj = load_transcript(args.transcript)
    raw = obj.get('raw', {}) if isinstance(obj, dict) else {}

    segments = build_segments_from_utterances(raw)
    if not segments:
        segments = build_segments_from_words(raw)

    # normalize speaker labels to short form if they come like "Speaker 0" -> "A","B"...
    # but keep them if already A,B etc.
    def normalize_label(lab):
        if not isinstance(lab, str):
            return str(lab)
        lab = lab.strip()
        if lab.lower().startswith('speaker'):
            try:
                n = int(''.join(ch for ch in lab if ch.isdigit()))
                # map 0->A,1->B ...
                return chr(ord('A') + n)
            except Exception:
                return lab
        # keep single-letter labels
        if len(lab) == 1:
            return lab
        return lab

    for s in segments:
        s['speaker'] = normalize_label(s['speaker'])
        s['text'] = s['text'].strip()

    out_obj = {
        'source': args.transcript,
        'segments': [{'speaker': seg['speaker'], 'text': seg['text'], 'start': seg.get('start'), 'end': seg.get('end')} for seg in segments],
    }

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)
        print('Wrote', args.out)
    else:
        print(json.dumps(out_obj, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
