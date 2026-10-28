"""Summarise results/losses.csv and results/generation.csv.
Usage: python3 analyze.py  -> prints tables and writes results/summary.md"""
import csv, statistics as st, random, os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results')
rows = list(csv.DictReader(open(os.path.join(R, 'losses.csv'))))
gen = list(csv.DictReader(open(os.path.join(R, 'generation.csv'))))
COND = ['A_repetition', 'B_variety', 'C_balanced', 'D_same_person', 'C_short', 'E_titles', 'F_syllables', 'G1_length1500', 'G3_length6000']
d = defaultdict(lambda: defaultdict(dict))  # d[cond][step][seed] = row
for r in rows: d[r['condition']][int(r['step'])][int(r['seed'])] = r
steps = sorted({int(r['step']) for r in rows})
out = []
def p(s=''): print(s); out.append(s)

def ms(vals): return (st.mean(vals), st.stdev(vals) if len(vals) > 1 else 0.0)

p('## Test loss (mean of T1–T3), mean ± sd over seeds')
p('| condition | ' + ' | '.join(str(s) for s in steps) + ' | best step |')
p('|---|' + '---|' * (len(steps) + 1))
for c in COND:
    if c not in d: continue
    cells, best = [], (9, None)
    for s in steps:
        v = [float(x['test_all']) for x in d[c][s].values()]
        if not v: cells.append(''); continue
        m, sd = ms(v); cells.append(f'{m:.3f} ± {sd:.3f}')
        if m < best[0]: best = (m, s)
    p(f'| {c} | ' + ' | '.join(cells) + f' | {best[1]} |')

for sub in ['T1', 'T2', 'T3', 'train_loss']:
    p(f'\n## {sub} at 1000 / 2000 / 4000 steps')
    p('| condition | 1000 | 2000 | 4000 |'); p('|---|---|---|---|')
    for c in COND:
        if c not in d: continue
        cells = []
        for s in [1000, 2000, 4000]:
            v = [float(x[sub]) for x in d[c].get(s, {}).values()]
            cells.append(f'{st.mean(v):.3f} ± {st.stdev(v) if len(v) > 1 else 0:.3f}' if v else '')
        p(f'| {c} | ' + ' | '.join(cells) + ' |')

def paired(a, b, step, key='test_all', reps=5000):
    """difference a - b per seed; 95% CI by bootstrap over seeds"""
    seeds = sorted(set(d[a][step]) & set(d[b][step]))
    diffs = [float(d[a][step][s][key]) - float(d[b][step][s][key]) for s in seeds]
    if len(diffs) < 2: return None
    rnd = random.Random(1); boots = sorted(st.mean(rnd.choices(diffs, k=len(diffs))) for _ in range(reps))
    return st.mean(diffs), boots[int(.025 * reps)], boots[int(.975 * reps)], sum(x < 0 for x in diffs), len(diffs)

p('\n## Pre-registered comparisons (difference in test loss, negative = first is better)')
p('| comparison | step | diff | 95% CI | seeds where first is better |'); p('|---|---|---|---|---|')
for a, b, s, label in [('C_balanced', 'A_repetition', 2000, 'H1 C vs A'), ('C_balanced', 'A_repetition', 4000, 'H1 C vs A'),
                       ('B_variety', 'C_balanced', 2000, 'B vs C'), ('B_variety', 'C_balanced', 4000, 'B vs C'),
                       ('E_titles', 'C_short', 2000, 'H2 E vs C_short'), ('F_syllables', 'C_short', 2000, 'H2 F vs C_short'),
                       ('E_titles', 'C_short', 4000, 'H2 E vs C_short'), ('F_syllables', 'C_short', 4000, 'H2 F vs C_short'),
                       ('G3_length6000', 'C_balanced', 4000, 'H3 6000 vs 3000'), ('C_balanced', 'G1_length1500', 4000, 'H3 3000 vs 1500'),
                       ('D_same_person', 'C_balanced', 2000, 'H4 D vs C'), ('D_same_person', 'C_balanced', 4000, 'H4 D vs C')]:
    r = paired(a, b, s)
    if r: p(f'| {label} | {s} | {r[0]:+.3f} | [{r[1]:+.3f}, {r[2]:+.3f}] | {r[3]}/{r[4]} |')

p('\n## Generation at 4000 steps (2000 characters, temperature 0.8), mean over seeds')
p('| condition | valid words | copied sentences | new valid sentences | attention to previous char |'); p('|---|---|---|---|---|')
g = defaultdict(list)
for r in gen:
    if int(r['step']) == max(steps): g[r['condition']].append(r)
for c in COND:
    if not g[c]: continue
    f = lambda k: st.mean(float(x[k]) for x in g[c])
    p(f"| {c} | {f('valid_words'):.0%} | {f('copy_rate'):.0%} | {f('novel_valid_rate'):.0%} | {f('attn_1'):.0%} |")

open(os.path.join(R, 'summary.md'), 'w').write('\n'.join(out) + '\n')
