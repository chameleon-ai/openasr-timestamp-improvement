#!/usr/bin/env python3
"""Deep-dive per-window diagnostics for one clip (or a set of clips).

Usage:
  python3 diagnose.py <clip-dir> [cand.json ...]
  python3 diagnose.py test/thriller
  python3 diagnose.py test/thriller whisper-timestamped.json cohere-*.json

Defaults to every candidate in the dir (truth.json excluded).

Prints, per candidate:
  - span/scale: candidate span vs truth span of the matched words and the
    TempErr rescale factor (scale < 1 = candidate over-stretched; > 1 = under).
    This is the knob TempErr's normalization reacts to.
  - signed drift in 15 s buckets of absolute truth time (post-scale,
    truth - candidate): + means the candidate word landed early.
  - full-miss words (candidate window does not overlap the truth window at
    all): count, direction (late/early), and the worst few with both windows.
  - worst-N normalized start-error words, and how much of the TempErr the
    high-error tail accounts for.
"""
import json
import os
import re
import sys
from difflib import SequenceMatcher

BUCKET_SECONDS = 15
WORST_N = 10


def normalize_word(w):
    return re.sub(r'[^\w]', '', w.lower())


def load_words(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    words = []
    for seg in data.get('segments', []):
        for w in seg.get('words', []):
            n = normalize_word(w.get('word', ''))
            if n:
                words.append({
                    'raw': w.get('word', ''),
                    'norm': n,
                    'start': float(w.get('start', 0.0)),
                    'end': float(w.get('end', 0.0)),
                })
    return words


def pairs(truth_words, cand_words):
    s = SequenceMatcher(None, [w['norm'] for w in truth_words], [w['norm'] for w in cand_words])
    return [(i1 + k, j1 + k) for tag, i1, i2, j1, j2 in s.get_opcodes()
            if tag == 'equal' for k in range(i2 - i1)]


def _affine_scale_offset(truth_vals, cand_vals):
    """Least-squares two-param affine fit cand -> truth (see compare.py's
    `_affine_scale_offset`; duplicated here so this tool runs from any cwd).
    Robust to a single misplaced boundary word."""
    n = len(cand_vals)
    if n < 2:
        return 1.0, 0.0
    mx = sum(cand_vals) / n
    my = sum(truth_vals) / n
    sxx = sum((c - mx) ** 2 for c in cand_vals)
    sxy = sum((c - mx) * (t - my) for t, c in zip(truth_vals, cand_vals))
    scale = sxy / sxx if sxx > 1e-9 else 1.0
    offset = my - scale * mx
    return scale, offset


def diagnose(truth_words, cand_words, label):
    mp = pairs(truth_words, cand_words)
    if len(mp) < 2:
        print(f"### {label}: only {len(mp)} matched word(s); nothing to show")
        return

    t_starts = [truth_words[i]['start'] for i, _ in mp]
    c_starts = [cand_words[j]['start'] for _, j in mp]
    # Endpoint span (raw, for human reference) and a least-squares affine
    # scale/offset used for the per-word errors (offset/stretched-invariant,
    # robust to one bad boundary word -- matches compare.py's TempErr).
    t_norm = [x - t_starts[0] for x in t_starts]
    c_norm = [x - c_starts[0] for x in c_starts]
    scale, offset = _affine_scale_offset(t_starts, c_starts)
    abs_err = [abs(t - (scale * c + offset)) for t, c in zip(t_starts, c_starts)]
    total_err = sum(abs_err)

    stretch = 'over-stretched' if scale < 1 else ('under-stretched' if scale > 1 else 'even')
    print(f"### {label}")
    print(f"  matched={len(mp)}/{len(truth_words)}  truthSpan={t_norm[-1]:.1f}s  "
          f"candSpan={c_norm[-1]:.1f}s  fitScale={scale:.3f} ({stretch})  TempStart={total_err / len(mp):.3f}s")

    buckets = {}
    for (i, j), e in zip(mp, abs_err):
        b = int(truth_words[i]['start'] // BUCKET_SECONDS) * BUCKET_SECONDS
        buckets.setdefault(b, []).append(truth_words[i]['start'] - (scale * cand_words[j]['start'] + offset))
    line = []
    for b in sorted(buckets):
        v = buckets[b]
        line.append(f"{b:>3}s:{sum(v) / len(v):+.2f}")
    print(f"  signed drift (truth-cand, post-scale) per {BUCKET_SECONDS}s truth-time bucket: {' '.join(line) or '-'}")

    full_miss = [(i, j) for i, j in mp
                 if cand_words[j]['end'] < truth_words[i]['start'] or cand_words[j]['start'] > truth_words[i]['end']]
    if full_miss:
        late = sum(1 for i, j in full_miss if (cand_words[j]['start'] + cand_words[j]['end']) / 2
                   > (truth_words[i]['start'] + truth_words[i]['end']) / 2)
        print(f"  full-miss (no window overlap): {len(full_miss)} "
              f"({len(full_miss) / len(mp):.0%}), direction late={late} early={len(full_miss) - late}")
        for i, j in full_miss[:5]:
            tw, cw = truth_words[i], cand_words[j]
            print(f"    t[{tw['start']:7.2f},{tw['end']:7.2f}] c[{cw['start']:7.2f},{cw['end']:7.2f}] "
                  f"truth={tw['raw']!r} cand={cw['raw']!r}")

    # Head/tail error: residual (truth - candidate) of the first and last
    # matched words after the affine fit. A large value here is exactly what
    # the old endpoint-pinned TempErr amplified into a fake whole-clip ramp --
    # the boundary word is misanchored and the least-squares fit absorbs it as
    # a slope. Reporting it directly surfaces boundary leaks as a first-class
    # signal instead of shape distortion.
    head_i, head_j = mp[0]
    tail_i, tail_j = mp[-1]
    head_res = truth_words[head_i]['start'] - (scale * cand_words[head_j]['start'] + offset)
    tail_res = truth_words[tail_i]['start'] - (scale * cand_words[tail_j]['start'] + offset)
    print(f"  head/tail start error (post-fit, truth-cand; + = cand early): "
          f"head={head_res:+.2f}s ({truth_words[head_i]['raw']!r}: "
          f"t[{truth_words[head_i]['start']:.2f}] c[{cand_words[head_j]['start']:.2f}])  "
          f"tail={tail_res:+.2f}s ({truth_words[tail_i]['raw']!r}: "
          f"t[{truth_words[tail_i]['start']:.2f}] c[{cand_words[tail_j]['start']:.2f}])")

    order = sorted(range(len(mp)), key=lambda k: -abs_err[k])[:WORST_N]
    big = [k for k in order if abs_err[k] > 0.5]
    tail = sum(abs_err[k] for k in big)
    print(f"  worst-{WORST_N} normalized start errors; >0.5s tail: n={len(big)} "
          f"accounts for {tail / total_err:.0%} of the {total_err:.1f}s error")

    for k in order:
        i, j = mp[k]
        tw, cw = truth_words[i], cand_words[j]
        print(f"    err={abs_err[k]:6.2f}  t[{i:3d}] {tw['start']:7.2f} {tw['raw']!r:18} "
              f"-> c[{j:3d}] {cw['start']:7.2f} {cw['raw']!r}")


def candidates_for(dir_path, args):
    if args:
        files = [a if os.path.isabs(a) or os.path.exists(a) else os.path.join(dir_path, a) for a in args]
    else:
        import glob
        files = sorted(p for p in glob.glob(os.path.join(dir_path, '*.json'))
                       if os.path.basename(p) != 'truth.json')
    out = []
    for p in files:
        if not os.path.exists(p):
            print(f'warning: no such file: {p}', file=sys.stderr)
            continue
        out.append((os.path.basename(p), p))
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    dir_path = sys.argv[1]
    if dir_path.endswith('/'):
        dir_path = dir_path.rstrip('/')
    truth_path = os.path.join(dir_path, 'truth.json')
    if not os.path.exists(truth_path):
        print(f'truth.json not found in {dir_path}')
        return
    truth_words = load_words(truth_path)
    for label, path in candidates_for(dir_path, sys.argv[2:]):
        diagnose(truth_words, load_words(path), label)
        print()


if __name__ == '__main__':
    main()
