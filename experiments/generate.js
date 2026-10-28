// Builds the nine training texts and the fixed test set from grammar.js.
// Usage: node generate.js   (writes texts/*.txt, texts/test_T1..T3.txt, texts/stats.csv)
const fs = require('fs');
const path = require('path');
const G = require('./grammar.js');
const { mulberry32 } = require('./tinygpt.js');

const OUT = path.join(__dirname, 'texts');
fs.mkdirSync(OUT, { recursive: true });
const rng = mulberry32(20261028);
const pick = (a) => a[Math.floor(rng() * a.length)];
const shuffle = (a) => { a = a.slice(); for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };

const LANG = G.language();
const PATTERNS = [...new Set(LANG.map((x) => x.pattern))];
const byPattern = Object.fromEntries(PATTERNS.map((p) => [p, LANG.filter((x) => x.pattern === p)]));

// ---------- 1. test set (made first, then kept out of every training text) ----------
// T2: ten person–object pairs that never appear together in any training sentence.
const RESERVED = [['Nil', 'kalkan'], ['Kaan', 'nane'], ['Nalan', 'kil'], ['Kenan', 'kekik'], ['Lale', 'elek'],
  ['Ali', 'ekin'], ['Ela', 'kalkan'], ['nine', 'kil'], ['anne', 'elek'], ['Kaan', 'keklik']];
const hasReserved = (x) => RESERVED.some(([a, b]) => x.w.includes(a) && x.w.includes(b));
const T2 = []; // spread across patterns, so no single pattern dominates
{
  const cands = shuffle(LANG.filter(hasReserved));
  while (T2.length < 20) for (const p of shuffle(PATTERNS)) { const c = cands.find((x) => x.pattern === p && !T2.includes(x)); if (c && T2.length < 20) T2.push(c); }
}
const T3 = shuffle([...G.longSentences()].filter((x) => !hasReserved(x))).slice(0, 20);
const T1 = [];
{ // one or two sentences from as many patterns as possible, none with a reserved pair
  const cands = shuffle(LANG.filter((x) => !hasReserved(x)));
  for (const p of shuffle(PATTERNS)) { const c = cands.find((x) => x.pattern === p && !T1.includes(x)); if (c && T1.length < 20) T1.push(c); }
  for (const c of cands) { if (T1.length >= 20) break; if (!T1.includes(c)) T1.push(c); }
}
const testSet = new Set([...T1, ...T2, ...T3].map((x) => x.s));
const POOL = LANG.filter((x) => !hasReserved(x) && !testSet.has(x.s));
const poolBy = Object.fromEntries(PATTERNS.map((p) => [p, POOL.filter((x) => x.pattern === p)]));

// ---------- 2. helpers ----------
const ALL_WORDS = [...new Set(LANG.flatMap((x) => x.w))];
function coverSet() { // a small set of sentences that uses every pattern and every word at least once
  const chosen = []; const seenW = new Set(); const seenP = new Set();
  for (const p of shuffle(PATTERNS)) { const c = pick(poolBy[p]); chosen.push(c); c.w.forEach((w) => seenW.add(w)); seenP.add(p); }
  for (const w of shuffle(ALL_WORDS)) {
    if (seenW.has(w)) continue;
    const c = shuffle(POOL).find((x) => x.w.includes(w));
    if (c) { chosen.push(c); c.w.forEach((v) => seenW.add(v)); }
  }
  return chosen;
}
function sampleUnique(n, start = []) { // pattern-balanced sampling without replacement
  const chosen = start.slice(); const used = new Set(chosen.map((x) => x.s));
  while (chosen.length < n) {
    const p = pick(PATTERNS); const left = poolBy[p].filter((x) => !used.has(x.s));
    if (!left.length) continue;
    const c = pick(left); chosen.push(c); used.add(c.s);
  }
  return chosen;
}
const len = (sents) => sents.reduce((a, x) => a + x.s.length + 1, 0);
function fillTo(unique, target) { // repeat a unique set, shuffled, until the target length
  const out = []; let i = 0; let order = shuffle(unique);
  while (len(out) < target) { if (i >= order.length) { order = shuffle(unique); i = 0; } out.push(order[i++]); }
  return out;
}
function paragraphs(sents, per = 5) {
  const ps = []; for (let i = 0; i < sents.length; i += per) ps.push(sents.slice(i, i + per)); return ps;
}
const render = (ps, head) => ps.map((p, i) => (head ? head(i) + '\n' : '') + p.map((x) => x.s).join('\n')).join('\n\n') + '\n';
function trimTo(text, target) { // cut at a paragraph or line break so the text ends cleanly near the target
  if (text.length <= target) return text;
  const cut = text.lastIndexOf('\n', target); return text.slice(0, cut) + '\n';
}
// stories: the same multiset of sentences, regrouped so each paragraph follows one person
function storyOrder(sents) {
  const groups = {};
  for (const x of sents) { const p = x.w.find((w) => G.persons.includes(w)) || 'other'; (groups[p] = groups[p] || []).push(x); }
  const chunks = [];
  for (const g of Object.values(groups)) {
    // round-robin over this person's distinct sentences, so copies of a sentence land in different paragraphs
    const uniq = [...new Map(g.map((x) => [x.s, x])).values()];
    const count = Object.fromEntries(uniq.map((x) => [x.s, g.filter((y) => y.s === x.s).length]));
    const seq = [];
    while (seq.length < g.length) for (const x of uniq) if (count[x.s]-- > 0) seq.push(x);
    const k = Math.min(5, uniq.length);
    for (let i = 0; i < seq.length; i += k) chunks.push(seq.slice(i, i + k));
  }
  return shuffle(chunks);
}

// ---------- 3. conditions ----------
const T = 3000;
const avg = len(POOL.slice(0, 400)) / 400;
const cover = coverSet();
const texts = {};

texts.A_repetition = trimTo(render(paragraphs(fillTo(cover, T))), T);
texts.B_variety = trimTo(render(paragraphs(shuffle(sampleUnique(Math.ceil(T / avg) + 5, cover)))), T);
const uniqueC = sampleUnique(Math.round(T / avg / 2.5), cover);
const sentsC = fillTo(uniqueC, T);
texts.C_balanced = trimTo(render(paragraphs(sentsC)), T);
texts.D_same_person = trimTo(render(storyOrder(sentsC)), T);

// E and F: the same shortened C, plus titles or syllable lines; C_short is its control
const shortSents = sentsC.slice(0, Math.floor(sentsC.length * 0.88));
const shortParas = paragraphs(shortSents);
texts.C_short = render(shortParas);
texts.E_titles = render(shortParas, (i) => G.lexicon.titles[i % G.lexicon.titles.length]);
texts.F_syllables = render(shortParas, (i) => G.lexicon.syllables[i % G.lexicon.syllables.length]);

texts.G1_length1500 = trimTo(render(paragraphs(fillTo(sampleUnique(Math.round(1500 / avg / 2.5), cover), 1500))), 1500);
texts.G3_length6000 = trimTo(render(paragraphs(fillTo(sampleUnique(Math.round(6000 / avg / 2.5), cover), 6000))), 6000);

// ---------- 4. checks and stats ----------
const allowed = new Set([...'elakinELAKİN .,?!\n']);
const rows = [['condition', 'chars', 'sentences', 'unique_sentences', 'words', 'unique_words', 'avg_sentence_chars', 'distinct_chars', 'char_bigram_entropy_bits']];
for (const [name, t] of Object.entries(texts)) {
  for (const c of t) if (!allowed.has(c)) throw new Error(`${name}: bad char ${c}`);
  for (const s of testSet) if (t.includes(s)) throw new Error(`${name}: contains test sentence ${s}`);
  const lines = t.split('\n').filter((l) => /[.?!]$/.test(l));
  const words = t.match(/[^\s.,?!]+/g) || [];
  const bg = {}; for (let i = 0; i + 1 < t.length; i++) bg[t.slice(i, i + 2)] = (bg[t.slice(i, i + 2)] || 0) + 1;
  const n = t.length - 1; const H = -Object.values(bg).reduce((a, c) => a + (c / n) * Math.log2(c / n), 0);
  rows.push([name, t.length, lines.length, new Set(lines).size, words.length, new Set(words).size,
    (lines.reduce((a, l) => a + l.length, 0) / lines.length).toFixed(1), new Set(t).size, H.toFixed(2)]);
  fs.writeFileSync(path.join(OUT, name + '.txt'), t);
}
fs.writeFileSync(path.join(OUT, 'test_T1_near.txt'), T1.map((x) => x.s).join('\n') + '\n');
fs.writeFileSync(path.join(OUT, 'test_T2_new_pairs.txt'), T2.map((x) => x.s).join('\n') + '\n');
fs.writeFileSync(path.join(OUT, 'test_T3_long.txt'), T3.map((x) => x.s).join('\n') + '\n');
fs.writeFileSync(path.join(OUT, 'stats.csv'), rows.map((r) => r.join(',')).join('\n') + '\n');
console.table(rows.slice(1).map((r) => Object.fromEntries(rows[0].map((k, i) => [k, r[i]]))));
console.log('language size', LANG.length, '| pool', POOL.length, '| cover set', cover.length, '| unique C', uniqueC.length);
