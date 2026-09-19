#!/usr/bin/env python3
"""Detect DROPPED true speech: stretches where truth has words the candidate
transcript does not.

The other tools score *quality* of what the candidate emitted (Recall/F1/TempErr).
This tool finds *omissions*: stretches of time where the truth has real words the
candidate did not emit. That is the class of error a single low metric can hide
(a 3-second dropped phrase is ~7 words out of 400, ~2% recall) yet is the most
user-visible failure -- the transcript is missing what the speaker actually said.

There are two drop shapes:

  GAP      -- a candidate silence interval (a stretch of the timeline no
              candidate word window covers) that the truth filled with a
              contiguous run of real speech (words merged within
              TRUTH_CONTIGUITY).
  STRETCH  -- the candidate's windows cover the timeline, but the candidate
              emitted far fewer words there than the truth has: some word
              windows were *stretched* across the missing speech, so the
              timeline looks covered while the words are not (the classic
              shape is a genuinely repeated phrase where the model emits a
              few copies and one window spans the rest of the timeline).
              The drop size is the *deficit*: truth words inside the covered
              part of a contiguous truth span minus the candidate windows
              overlapping the span. Because the deficit is inferred from word
              counts (the timeline looks covered), it uses a sturdier floor
              than the hole pass: max(-w, 3) words.

Measured purely from timestamps, both passes are immune to SequenceMatcher
mis-pairing on repeated words -- they catch a drop even when the
surrounding alignment is confused. Splitting by silence interval (GAP) and
counting only covered words (STRETCH) means a dropped phrase glued to an
*adjacent captured* phrase reports only the dropped part, and the two passes
never double count the same truth words.

Each dropped region (GAP: the dropped words; STRETCH: the covered words) is
flagged **REPEAT x{reps} (len {L})** when it is dominated by a short token
cycle (len 1-4) -- the signature of the shared degenerate-repeat guard cutting
real repeated speech (a "there we go" x2, a "tap" x7, a "yeah" xN). That flag
points straight at the openasr guard as the prime suspect.

Usage:
  python3 drops.py <clip-dir> [-c a.json b.json ...] [-o a.json ...]
  python3 drops.py --suite [cand]          # every clip; cand = <model>.json or 'all'
      -g <min span seconds>  (default 1.5)  truth words must span >= this long
      -w <min word count>    (default 3)    a drop needs >= this many truth words

Per drop it prints the candidate silence gap, the dropped truth text, and the
candidate words immediately before / after. The SUM line (suite mode) rolls up
drop / word / time / repeat-shaped counts across the corpus.
"""
import sys

import compare as cp


def candidate_files(dir_path):
    return [p for p in cp.candidate_files(dir_path)]


def _repeated_phrase(norms, min_coverage=0.7):
    """Is this span dominated by a short repeated token cycle?

    Returns (ngram_len, reps) of the best cycle, or (0, 0). For each candidate
    cycle length L (1..4) we test whether the span is *periodic* with period L
    (norms[k] == norms[k % L] for enough k). A span is flagged when some cycle
    meets a per-length repetition floor AND covers >= min_coverage of the span.

    The per-L floor mirrors the shared degenerate-repeat guard
    (seq2seq_greedy_decode.default_max_consecutive_ngram_repeats: 8 for 1-token,
    6 for 2-token, 4 for longer) so this flag points at the same failure the
    guard produces -- a short phrase emitted back to back until the decoder is
    cut short. A normal phrase that merely *contains* a repeated word (e.g.
    "I mean I mean to say") does not tile the whole span, so it never reaches
    the coverage floor."""
    # minimum number of complete cycle copies for a span to be repeat-shaped.
    # 1-token ("yeah yeah yeah...") and 2-token are allowed fewer because
    # single-word stutters are common and short.
    FLOOR = {1: 4, 2: 3}
    L_DEFAULT = 2

    n = len(norms)
    if n < 4:
        return (0, 0)
    best = (0, 0)
    best_score = 0.0
    for L in range(1, 5):
        if n < L * 2:
            continue
        # fraction of positions that obey period L (only from L onward, since
        # the first period is the definition of the phase).
        period = sum(1 for k in range(L, n) if norms[k] == norms[k % L])
        coverage = period / max(1, n - L)
        reps = n // L
        if reps < FLOOR.get(L, L_DEFAULT):
            continue
        if coverage < min_coverage:
            continue
        score = reps * coverage  # reward long, clean cycles
        if score > best_score:
            best = (L, reps)
            best_score = score
    return best


TRUTH_CONTIGUITY = 0.6  # merge adjacent truth words within this many seconds


def _build_silence(cand_sorted, clip_duration):
    """Disjoint silence intervals: parts of the timeline no candidate word
    window covers (clipped to [0, clip_duration]). cand_sorted is sorted by
    start; overlapping candidate windows are merged into a single covered
    interval before taking the complement."""
    if not cand_sorted:
        return [(0.0, clip_duration)] if clip_duration > 0 else []
    covered = [[cand_sorted[0]['start'], cand_sorted[0]['end']]]
    for w in cand_sorted[1:]:
        if w['start'] <= covered[-1][1]:
            covered[-1][1] = max(covered[-1][1], w['end'])
        else:
            covered.append([w['start'], w['end']])
    silence = []
    prev_end = 0.0
    for start, end in covered:
        if start > prev_end:
            silence.append((prev_end, start))
        prev_end = max(prev_end, end)
    if prev_end < clip_duration:
        silence.append((prev_end, clip_duration))
    return silence


def find_drops(truth_words, cand_words, min_gap=1.0, min_words=2):
    """Yield drop regions: stretches of the candidate's word-timeline the
    candidate emitted no words in, where the truth has real speech.

    For each candidate silence interval (the complement of the merged candidate
    word windows), the truth words inside it are merged into contiguous speech
    spans (gap <= TRUTH_CONTIGUITY). Each span that is >= min_gap long and has
    >= min_words words is a drop. Splitting by silence interval means a dropped
    phrase glued to an *adjacent captured* phrase reports only the dropped
    part (the captured part is inside a different, covered interval)."""
    drops = []
    truth_sorted = sorted(truth_words, key=lambda w: w['start'])
    if not truth_sorted:
        return drops
    clip_duration = max(w['end'] for w in truth_sorted)
    if cand_words:
        clip_duration = max(clip_duration, max(w['end'] for w in cand_words))
    cand_sorted = sorted(cand_words, key=lambda w: w['start'])
    silence = _build_silence(cand_sorted, clip_duration)

    for gap_start, gap_end in silence:
        # truth words whose midpoint falls in this silence interval.
        inside = [w for w in truth_sorted
                  if gap_start - 1e-6 <= (w['start'] + w['end']) / 2.0 <= gap_end + 1e-6]
        if len(inside) < min_words:
            continue
        # merge into contiguous speech spans.
        spans = []
        for w in inside:
            if spans and (w['start'] - spans[-1]['end']) <= TRUTH_CONTIGUITY:
                spans[-1]['words'].append(w)
                spans[-1]['end'] = max(spans[-1]['end'], w['end'])
            else:
                spans.append({'start': w['start'], 'end': w['end'], 'words': [w]})
        for span in spans:
            words = span['words']
            if len(words) < min_words:
                continue
            span_len = span['end'] - span['start']
            if span_len < min_gap:
                continue
            norms = [w['norm'] for w in words]
            L, reps = _repeated_phrase(norms)
            prev_cand = [c for c in cand_sorted if c['end'] <= span['start'] + 1e-6]
            next_cand = [c for c in cand_sorted if c['start'] >= span['end'] - 1e-6]
            drops.append({
                'span_start': span['start'],
                'span_end': span['end'],
                'gap_start': gap_start,
                'gap_end': gap_end,
                'gap': gap_end - gap_start,
                'truth_time': span['end'] - span['start'],
                'words': words,
                'n_words': len(words),
                'repeat': L > 0,
                'repeat_len': L,
                'repeat_count': reps,
                'ctx_before': prev_cand[-1]['raw'] if prev_cand else '<start>',
                'ctx_after': next_cand[0]['raw'] if next_cand else '<end>',
            })
    return drops


def _in_silence(t, silence):
    return any(g0 - 1e-6 <= t <= g1 + 1e-6 for g0, g1 in silence)


def find_undercounts(truth_words, cand_words, min_gap, min_words):
    """Yield STRETCH undercount regions: contiguous truth spans the candidate
    timeline *covers* but the candidate transcribed with far fewer words than
    the truth has there, so word windows were stretched over the missing
    speech and the GAP pass above sees no silence.

    For each TRUTH_CONTIGUITY-merged truth span (over the whole clip) we count
    the truth words whose midpoint is covered (not in a candidate silence
    interval) and the candidate windows overlapping the span; the deficit is
    the excess of the former over the later. A span is an undercount drop when
    it is >= min_gap long and the deficit is >= min_words. Only covered truth
    words enter the deficit, so hole portions of a span stay with find_drops
    and the two passes never double count."""
    out = []
    truth_sorted = sorted(truth_words, key=lambda w: w['start'])
    if not truth_sorted:
        return out
    cand_sorted = sorted(cand_words, key=lambda w: w['start'])
    silence = _build_silence(cand_sorted,
                             max(max(w['end'] for w in truth_sorted),
                                 max((w['end'] for w in cand_sorted), default=0.0)))
    spans = []
    for w in truth_sorted:
        if spans and (w['start'] - spans[-1]['end']) <= TRUTH_CONTIGUITY:
            spans[-1]['words'].append(w)
            spans[-1]['end'] = max(spans[-1]['end'], w['end'])
        else:
            spans.append({'start': w['start'], 'end': w['end'], 'words': [w]})
    for span in spans:
        span_len = span['end'] - span['start']
        if span_len < min_gap:
            continue
        covered = [w for w in span['words']
                   if not _in_silence((w['start'] + w['end']) / 2.0, silence)]
        if not covered:
            continue
        K = sum(1 for c in cand_sorted
                if c['start'] < span['end'] and c['end'] > span['start'])
        deficit = len(covered) - K
        # The deficit is inferred from word counts (the timeline looks
        # covered), so it gets a sturdier floor than the hole pass: a
        # two-word undercount in a covered stretch is usually filler /
        # truth judgment noise, not a drop (a hole of the same size has
        # zero candidate windows and is certain).
        if deficit < max(min_words, 3):
            continue
        # length of the covered portion of the span (drop the hole parts).
        hole_len = 0.0
        for g0, g1 in silence:
            if g0 < span['end'] and g1 > span['start']:
                hole_len += min(span['end'], g1) - max(span['start'], g0)
        norms = [w['norm'] for w in covered]
        L, reps = _repeated_phrase(norms)
        prev_cand = [c for c in cand_sorted if c['end'] <= span['start'] + 1e-6]
        next_cand = [c for c in cand_sorted if c['start'] >= span['end'] - 1e-6]
        out.append({
            'span_start': span['start'],
            'span_end': span['end'],
            'gap_start': span['start'],
            'gap_end': span['end'],
            'gap': span_len - hole_len,
            'truth_time': span_len - hole_len,
            'words': covered,
            'n_words': deficit,
            'covered': len(covered),
            'cand': K,
            'stretch': True,
            'repeat': L > 0,
            'repeat_len': L,
            'repeat_count': reps,
            'ctx_before': prev_cand[-1]['raw'] if prev_cand else '<start>',
            'ctx_after': next_cand[0]['raw'] if next_cand else '<end>',
        })
    return out


def _fmt_drop(d):
    flag = f"  REPEAT x{d['repeat_count']} (len {d['repeat_len']})" if d['repeat'] else ''
    if d.get('stretch'):
        line = (f"  [{d['gap_start']:7.2f} -> {d['gap_end']:7.2f}]  "
                f"STRETCH  {d['n_words']:>2}w deficit over {d['gap']:5.2f}s covered"
                f" ({d['covered']} truth words vs {d['cand']} candidate windows){flag}")
        print(line)
        # the covered region, truncated: the deficit is a subset of it.
        tw = d['words']
        shown = ' '.join(w['raw'] for w in tw[:20])
        if len(tw) > 20:
            shown += f' ... (+{len(tw) - 20} more words)'
        print(f"      region:  {shown}")
    else:
        line = (f"  [{d['gap_start']:7.2f} -> {d['gap_end']:7.2f}]  "
                f"gap {d['gap']:5.2f}s  {d['n_words']:>2}w{flag}")
        print(line)
        print(f"      dropped: {' '.join(w['raw'] for w in d['words'])}")
    print(f"      after:   ...{d['ctx_before']}")
    print(f"      resumes: {d['ctx_after']}...")


def score_clip(truth_words, cand_words, min_gap, min_words):
    drops = find_drops(truth_words, cand_words, min_gap, min_words)
    drops += find_undercounts(truth_words, cand_words, min_gap, min_words)
    drops.sort(key=lambda d: -d['n_words'])
    dropped_words = sum(d['n_words'] for d in drops)
    dropped_time = sum(d['truth_time'] for d in drops)
    return drops, dropped_words, dropped_time


def main():
    import os
    import glob
    args = sys.argv[1:]
    min_gap = 1.5
    min_words = 3
    suite = False
    suite_filter = 'all'
    root = '.'
    only = set()
    omit = set()
    positionals = []
    i = 0
    while i < len(args):
        a = args[i]
        if a in ('-g',):
            min_gap = float(args[i + 1]); i += 2
        elif a in ('-w',):
            min_words = int(args[i + 1]); i += 2
        elif a in ('-h', '--help'):
            print(__doc__)
            return
        elif a in ('--suite', '-s'):
            suite = True
            if i + 1 < len(args) and args[i + 1] not in ('-g', '-w', '-c', '-o', '-h', '--help', '.'):
                suite_filter = args[i + 1]
                i += 1
            i += 1
        elif a in ('-c', '-o'):
            mode = 'only' if a == '-c' else 'omit'
            i += 1
            while i < len(args) and args[i] not in ('-g', '-w', '-c', '-o', '-s', '--suite', '-h', '--help'):
                (only if mode == 'only' else omit).add(args[i]); i += 1
        else:
            positionals.append(a); i += 1

    # Positionals: if any is a clip dir (has truth.json) it is the root; the
    # rest act as candidate filters. In --suite mode root defaults to '.' (the
    # suite root, e.g. test/) and positionals are candidate filters.
    if not suite:
        for p in positionals:
            if os.path.isdir(p) and os.path.exists(os.path.join(p, 'truth.json')):
                root = p
                break
        else:
            if positionals and os.path.isdir(positionals[0]):
                root = positionals[0]
        if positionals:
            only |= set(p for p in positionals if p != root)

    if suite:
        root = os.path.abspath(root)
        clips = sorted(p for p in glob.glob(os.path.join(root, '*'))
                       if os.path.isdir(p) and os.path.exists(os.path.join(p, 'truth.json')))
    elif os.path.exists(os.path.join(root, 'truth.json')):
        clips = [root]
    else:
        clips = sorted(p for p in glob.glob(os.path.join(root, '*'))
                       if os.path.isdir(p) and os.path.exists(os.path.join(p, 'truth.json')))

    grand_drops = 0
    grand_words = 0
    grand_time = 0.0
    grand_repeat = 0
    for clip in clips:
        truth_path = os.path.join(clip, 'truth.json')
        if not os.path.exists(truth_path):
            continue
        truth_words = cp.load_words(truth_path)
        files = [p for p in candidate_files(clip)]
        fbase = [os.path.basename(p) for p in files]
        if only:
            files = [p for p, b in zip(files, fbase) if b in only]
        if omit:
            files = [p for p, b in zip(files, fbase) if b not in omit]
        if suite:
            files = [p for p, b in zip(files, fbase)
                     if suite_filter == 'all'
                     or b == suite_filter
                     or b == suite_filter + '.json'
                     or b.endswith('_' + suite_filter + '.json')]
        if not files:
            continue
        print(f'=== {os.path.relpath(clip)} ===')
        for path in files:
            cand_words = cp.load_words(path)
            drops, dw, dt = score_clip(truth_words, cand_words, min_gap, min_words)
            grand_drops += len(drops)
            grand_words += dw
            grand_time += dt
            grand_repeat += sum(1 for d in drops if d['repeat'])
            name = os.path.basename(path)
            if not drops:
                print(f'  {name}: no drops')
                continue
            print(f'  {name}: {len(drops)} drop(s), {dw} word(s), {dt:.1f}s of truth '
                  f'({dw / len(truth_words):.1%} of clip)')
            for d in drops:
                _fmt_drop(d)
        print()

    if suite:
        print(f'SUM over {len(clips)} clip(s): {grand_drops} drop(s), '
              f'{grand_words} word(s), {grand_time:.1f}s '
              f'({grand_repeat} repeat-shaped)')


if __name__ == '__main__':
    main()
