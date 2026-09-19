#!/usr/bin/env python3
"""Find gaps between words in a single transcript JSON.

No truth comparison, unlike drops.py: this only looks at the candidate's
own word timeline and reports where it goes quiet for at least --gap
seconds. A gap is the stretch between one emitted word and the next (the
next word's start minus the previous word's end), so the time between the
start of the audio and the first word is *not* a gap, nor is what follows
the last word -- only the space between two consecutive emitted words.

Words are ordered by start before measuring, so overlapping or touching
windows produce no gap (a negative/zero inter-word stretch is skipped).

Usage:
  python3 gaps.py <transcript.json> [--gap 5.0]

Per gap it prints the silent [end -> start] window, the gap size, and the
emitted word immediately before / after (the seam, the way drops.py shows
the words around a drop).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare as cp


def find_gaps(words, min_gap):
    """Gaps between consecutive emitted words >= min_gap seconds.

    words: as returned by compare.load_words (raw/norm/start/end). Returns
    a list of dicts, in timeline order; each has the window (start, end),
    the gap size, and the words immediately before and after the seam."""
    gaps = []
    if len(words) < 2:
        return gaps
    ws = sorted(words, key=lambda w: (w['start'], w['end']))
    for prev, nxt in zip(ws, ws[1:]):
        gap = nxt['start'] - prev['end']
        if gap >= min_gap:
            gaps.append({'start': prev['end'], 'end': nxt['start'],
                         'gap': gap, 'before': prev, 'after': nxt})
    return gaps


def main():
    ap = argparse.ArgumentParser(
        description='gap between consecutive words in one transcript JSON '
                    '(the head, up to the first word, is not a gap)')
    ap.add_argument('transcript', help='transcript JSON (segments[].words[])')
    ap.add_argument('--gap', type=float, default=5.0,
                    help='report gaps of at least this many seconds (default 5.0)')
    args = ap.parse_args()

    if not os.path.exists(args.transcript):
        sys.exit(f'no such file: {args.transcript}')
    words = cp.load_words(args.transcript)
    gaps = find_gaps(words, args.gap)

    for g in gaps:
        print(f'  [{g["start"]:8.2f} -> {g["end"]:8.2f}]  gap {g["gap"]:6.2f}s')
        print(f'      before: ... {g["before"]["raw"]}')
        print(f'      after:  {g["after"]["raw"]}')
    print(f'{len(gaps)} gap(s) >= {args.gap:.2f}s in {args.transcript} ({len(words)} words)')


if __name__ == '__main__':
    main()
