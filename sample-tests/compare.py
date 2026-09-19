#!/usr/bin/env python3
import json
import glob
import os
import re
import sys
from difflib import SequenceMatcher

def normalize_word(w):
    w = w.lower()
    w = re.sub(r'[^\w]', '', w)
    return w

def load_words(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    words = []
    for seg in data.get('segments', []):
        for w in seg.get('words', []):
            word_text = w.get('word', '')
            start = w.get('start', 0.0)
            end = w.get('end', 0.0)
            norm = normalize_word(word_text)
            if norm:
                words.append({'raw': word_text, 'norm': norm, 'start': float(start), 'end': float(end)})
    return words

def candidate_files(dir_path, extra=None):
    """Sorted candidate json files in dir_path, truth.json excluded."""
    files = [
        p for p in glob.glob(os.path.join(dir_path, '*.json'))
        if not os.path.basename(p) == 'truth.json'
    ]
    if extra:
        for e in extra:
            p = e if os.path.isabs(e) else os.path.join(dir_path, e)
            if p not in files:
                files.append(p)
    return sorted(files)

def word_pairs(truth_words, cand_words):
    """(truth_idx, cand_idx) pairs for words whose normalized text is equal,
    via a whole-clip SequenceMatcher alignment (the same alignment compare()
    uses, so 'matched' here == compare()'s tp)."""
    s = SequenceMatcher(None, [w['norm'] for w in truth_words], [w['norm'] for w in cand_words])
    return [(i1 + k, j1 + k) for tag, i1, i2, j1, j2 in s.get_opcodes()
            if tag == 'equal' for k in range(i2 - i1)]

def _affine_scale_offset(truth_vals, cand_vals):
    """Least-squares two-param affine fit cand -> truth: `t ~= scale*c + offset`.

    Returns (scale, offset, mean_abs_residual). The least-squares fit is
    invariant to a global offset and a uniform stretch (matching the
    endpoint-pinned intent) but, unlike subtracting the first match and
    scaling to the last, it lets no single boundary word dominate: one
    misplaced first/last word can no longer drag the scale/offset and
    manufacture a fake clip-wide linear ramp."""
    n = len(cand_vals)
    if n < 2:
        return 1.0, 0.0, 0.0
    mx = sum(cand_vals) / n
    my = sum(truth_vals) / n
    sxx = sum((c - mx) ** 2 for c in cand_vals)
    sxy = sum((c - mx) * (t - my) for t, c in zip(truth_vals, cand_vals))
    scale = sxy / sxx if sxx > 1e-9 else 1.0
    offset = my - scale * mx
    mae = sum(abs(t - (scale * c + offset)) for t, c in zip(truth_vals, cand_vals)) / n
    return scale, offset, mae


def compare(truth_words, cand_words):
    truth_norms = [w['norm'] for w in truth_words]
    cand_norms = [w['norm'] for w in cand_words]

    s = SequenceMatcher(None, truth_norms, cand_norms)
    ops = s.get_opcodes()

    tp = 0
    fp = 0
    fn = 0
    matched_pairs = []

    for tag, i1, i2, j1, j2 in ops:
        if tag == 'equal':
            for k in range(i2 - i1):
                tp += 1
                matched_pairs.append((i1 + k, j1 + k))
        elif tag == 'replace':
            n_t = i2 - i1
            n_c = j2 - j1
            fn += n_t
            fp += n_c
        elif tag == 'delete':
            fn += i2 - i1
        elif tag == 'insert':
            fp += j2 - j1

    total_truth = len(truth_words)
    total_cand = len(cand_words)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / total_truth if total_truth > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # normalized temporal error for matched words: a two-param affine
    # (scale + offset) fit over ALL matched words, invariant to a global offset
    # and a uniform stretch, robust to a single misplaced boundary word.
    if len(matched_pairs) >= 2:
        t_starts = [truth_words[i]['start'] for i, _ in matched_pairs]
        c_starts = [cand_words[j]['start'] for _, j in matched_pairs]
        _, _, avg_start_err = _affine_scale_offset(t_starts, c_starts)

        t_ends = [truth_words[i]['end'] for i, _ in matched_pairs]
        c_ends = [cand_words[j]['end'] for _, j in matched_pairs]
        _, _, avg_end_err = _affine_scale_offset(t_ends, c_ends)
    else:
        avg_start_err = 0
        avg_end_err = 0

    avg_temporal_err = (avg_start_err + avg_end_err) / 2

    wer = (fp + fn) / total_truth if total_truth > 0 else 0

    # Absolute, non-normalized temporal metrics: no removal of the global
    # offset and no rescaling to the last word. These measure how well each
    # candidate window lands on the true window in place, which the normalized
    # TempErr above cannot show (it subtracts the first matched start and
    # scales to the last, so a constant late/early bias is invisible).
    if matched_pairs:
        raw_starts = [abs(truth_words[i]['start'] - cand_words[j]['start']) for i, j in matched_pairs]
        raw_ends = [abs(truth_words[i]['end'] - cand_words[j]['end']) for i, j in matched_pairs]
        avg_abs_start_err = sum(raw_starts) / len(raw_starts)
        avg_abs_end_err = sum(raw_ends) / len(raw_ends)
        # A word "misses the truth window entirely" when its candidate window
        # has no overlap with the truth window at all.
        full_miss = sum(
            1 for i, j in matched_pairs
            if cand_words[j]['end'] < truth_words[i]['start']
            or cand_words[j]['start'] > truth_words[i]['end']
        )
        in_window = (len(matched_pairs) - full_miss) / len(matched_pairs)
    else:
        avg_abs_start_err = 0
        avg_abs_end_err = 0
        in_window = 0

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'wer': wer,
        'avg_start_err': avg_start_err,
        'avg_end_err': avg_end_err,
        'avg_temporal_err': avg_temporal_err,
        'avg_abs_start_err': avg_abs_start_err,
        'avg_abs_end_err': avg_abs_end_err,
        'in_window': in_window,
        'matched_words': tp,
        'total_truth': total_truth,
        'total_cand': total_cand,
    }

def _fmt_row(r):
    return (f"{r['file']:34} {r['recall']:7.3f} {r['precision']:7.3f} {r['f1']:7.3f} "
            f"{r['wer']:7.3f} {r['avg_start_err']:9.3f} {r['avg_end_err']:7.3f} "
            f"{r['avg_temporal_err']:8.3f} {r['avg_abs_start_err']:9.3f} {r['avg_abs_end_err']:8.3f} "
            f"{r['in_window']:6.0%}")

def _header():
    print(f"{'File':34} {'Recall':>7} {'Prec':>7} {'F1':>7} {'WER':>7} {'StartErr':>9} {'EndErr':>7} "
          f"{'TempErr':>8} {'AbsStart':>9} {'AbsEnd':>8} {'InWin':>6}")

HELP = """Usage: compare.py <clip-dir> [-c a.json ...] [-o a.json ...]
  <clip-dir>  a dir containing truth.json ('.' if omitted)
  -c  only score the named candidate files
  -o  omit the named candidate files

Scores every *.json in <clip-dir> (except truth.json, and anything filtered
with -c/-o) against truth.json and prints the metric table below."""


def main():
    args = sys.argv[1:]
    dir_path = None
    only = set()
    omit = set()
    mode = None
    for a in args:
        if a in ('-c', '-o'):
            mode = 'only' if a == '-c' else 'omit'
        elif a in ('-h', '--help'):
            print(HELP)
            return
        elif dir_path is None and (os.path.isdir(a) or a == '.'):
            dir_path = a
        else:
            (omit if mode == 'omit' else only).add(a)
    if dir_path is None:
        dir_path = '.'
    truth_path = os.path.join(dir_path, 'truth.json')
    if not os.path.exists(truth_path):
        print('truth.json not found')
        return

    truth_words = load_words(truth_path)
    results = []

    for path in candidate_files(dir_path):
        fname = os.path.basename(path)
        if only and fname not in only:
            continue
        if fname in omit:
            continue
        try:
            cand_words = load_words(path)
            metrics = compare(truth_words, cand_words)
            metrics['file'] = fname
            results.append(metrics)
        except Exception as e:
            print(f'Error processing {fname}: {e}')

    # sort by F1 desc, then in-window coverage desc, then temporal error asc
    results.sort(key=lambda r: (-r['f1'], -r['in_window'], r['avg_temporal_err'], -r['recall']))

    _header()
    for r in results:
        print(_fmt_row(r))

    if only and len(results) == 1:
        # single-candidate mode: skip the "best" footer (it always picks itself)
        return

    best = results[0] if results else None
    if best:
        print('\nBest match by F1 then temporal error:')
        print(best['file'])
        print(json.dumps({k: best[k] for k in ['file','recall','precision','f1','wer','avg_start_err','avg_end_err','avg_temporal_err']}, indent=2))

if __name__ == '__main__':
    main()
