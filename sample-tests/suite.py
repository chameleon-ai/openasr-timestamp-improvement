#!/usr/bin/env python3
"""Cross-clip comparison table for one (or more) candidate files.

Usage:
  python3 suite.py [cand.json ...]

With no args it uses <model>.json for every model seen under test/*/ (except
the static references), so two binaries differ in *which* files exist and you
can diff those runs by giving the filenames explicitly:

  python3 suite.py whisper-large-v3-turbo.json
  # A/B two saved sets (files kept in their own clip dirs under distinct names)
  python3 suite.py runA_whisper.json runB_whisper.json

For each candidate, over every clip dir that contains it, prints Recall/Prec/
F1/WER/TempErr/AbsStart/AbsEnd/InWin plus the suite MEAN (row at the bottom).
Rows are sorted by TempErr asc so the worst clips surface first.
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare as cp

CLIPS = None  # discover dynamically


def clips_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def discover_clips():
    root = clips_root()
    return sorted(d for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d))
                  and os.path.exists(os.path.join(root, d, 'truth.json')))


def default_candidates(root):
    names = set()
    for clip in discover_clips():
        for p in glob.glob(os.path.join(root, clip, '*.json')):
            n = os.path.basename(p)
            if n != 'truth.json':
                names.add(n)
    return sorted(names)


def main():
    root = clips_root()
    args = sys.argv[1:]
    cands = args if args else default_candidates(root)
    clips = discover_clips()

    header = (f"{'clip':12} {'File':34} {'Recall':>7} {'F1':>7} {'WER':>7} "
              f"{'TempErr':>8} {'AbsStart':>9} {'AbsEnd':>8} {'InWin':>6}")
    print(header)
    print('-' * len(header))

    totals = {c: {k: 0.0 for k in ('temp', 'inwin', 'wer', 'f1', 'rec', 'abss', 'abse')} | {'n': 0} for c in cands}
    for cand in cands:
        rows = []
        for clip in clips:
            path = os.path.join(root, clip, cand)
            if not os.path.exists(path):
                continue
            try:
                truth = cp.load_words(os.path.join(root, clip, 'truth.json'))
                m = cp.compare(truth, cp.load_words(path))
            except Exception as e:
                print(f'error {clip}/{cand}: {e}', file=sys.stderr)
                continue
            rows.append((clip, m))
            t = totals[cand]
            t['temp'] += m['avg_temporal_err']
            t['inwin'] += m['in_window']
            t['wer'] += m['wer']
            t['f1'] += m['f1']
            t['rec'] += m['recall']
            t['abss'] += m['avg_abs_start_err']
            t['abse'] += m['avg_abs_end_err']
            t['n'] += 1
        rows.sort(key=lambda r: r[1]['avg_temporal_err'], reverse=True)
        for clip, m in rows:
            print(f"{clip:12} {cand:34} {m['recall']:7.3f} {m['f1']:7.3f} {m['wer']:7.3f} "
                  f"{m['avg_temporal_err']:8.3f} {m['avg_abs_start_err']:9.3f} {m['avg_abs_end_err']:8.3f} "
                  f"{m['in_window']:6.0%}")
        t = totals[cand]
        n = t['n']
        print(f"{'MEAN':12} {cand:34} {t['rec'] / n:7.3f} {t['f1'] / n:7.3f} {t['wer'] / n:7.3f} "
              f"{t['temp'] / n:8.3f} {t['abss'] / n:9.3f} {t['abse'] / n:8.3f} {t['inwin'] / n:6.0%}  ({n} clips)")
        print()


if __name__ == '__main__':
    main()
