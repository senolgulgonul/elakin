// Trains every condition with several seeds and writes results/*.csv.
// Usage: node run.js [steps=4000] [seeds=5] [conditions=all, comma separated]
const fs = require('fs');
const path = require('path');
const TinyGPT = require('./tinygpt.js');
const G = require('./grammar.js');

const STEPS = +(process.argv[2] || 4000);
const SEEDS = +(process.argv[3] || 5);
const CHECK = [250, 500, 1000, 2000, 3000, 4000].filter((s) => s <= STEPS);
const TXT = path.join(__dirname, 'texts');
const RES = path.join(__dirname, 'results');
fs.mkdirSync(RES, { recursive: true });
const ALL = ['A_repetition', 'B_variety', 'C_balanced', 'D_same_person', 'C_short', 'E_titles', 'F_syllables', 'G1_length1500', 'G3_length6000'];
const CONDS = process.argv[4] ? process.argv[4].split(',') : ALL;
const CFG = { T: 32, d: 32, f: 64, B: 12, lr: 0.003 };

// one fixed vocabulary for every condition: the 18 characters of the ELAKİN language + ¤ (unknown)
const CHARS = [...new Set('\n !,.?AEKLNaeiklnİ')].sort((a, b) => a.localeCompare(b));
CHARS.push('¤');
const STOI = new Map(CHARS.map((c, i) => [c, i]));
const enc = (s) => Int32Array.from([...s].map((c) => (STOI.has(c) ? STOI.get(c) : STOI.get('¤'))));
const V = CHARS.length;

const tests = Object.fromEntries(['T1_near', 'T2_new_pairs', 'T3_long'].map((k) => {
  const f = fs.readdirSync(TXT).find((x) => x.startsWith('test_' + k.slice(0, 2)));
  return [k.slice(0, 2), fs.readFileSync(path.join(TXT, f), 'utf8').trim().split('\n')];
}));
const VALID = new Set([...G.language().map((x) => x.s), ...[...G.longSentences()].map((x) => x.s)]);
const WORDS = new Set([...VALID].flatMap((s) => s.match(/[^\s.,?!]+/g)));

// mean next-character loss over a text, read in 32-character chunks
function lossOf(m, ids) {
  let loss = 0, n = 0;
  for (let s = 0; s + 1 < ids.length; s += CFG.T) {
    const e = Math.min(s + CFG.T, ids.length - 1); const c = m.forward(ids.subarray(s, e));
    for (let i = 0; i < e - s; i++) {
      let mx = -Infinity; for (let j = 0; j < V; j++) mx = Math.max(mx, c.L[i * V + j]);
      let sum = 0; for (let j = 0; j < V; j++) sum += Math.exp(c.L[i * V + j] - mx);
      loss += Math.log(sum) + mx - c.L[i * V + ids[s + 1 + i]]; n++;
    }
  }
  return loss / n;
}
// each test sentence is scored on its own line, after a newline, as the model would see it in a text
const testLoss = (m, sents) => sents.reduce((a, s) => a + lossOf(m, enc('\n' + s + '\n')), 0) / sents.length;

function generate(m, n, rng) {
  const ctx = [STOI.get('\n')]; let out = '';
  for (let i = 0; i < n; i++) {
    const p = m.nextProbs(Int32Array.from(ctx.slice(-CFG.T)), 0.8);
    let r = rng(), a = 0, id = V - 1; for (let j = 0; j < V; j++) { a += p[j]; if (r < a) { id = j; break; } }
    ctx.push(id); out += CHARS[id];
  }
  return out;
}
function genStats(text, train) {
  const words = text.match(/[^\s.,?!]+/g) || [];
  const sents = text.split('\n').map((s) => s.trim()).filter((s) => /[.?!]$/.test(s));
  const trainSet = new Set(train.split('\n'));
  const copy = sents.filter((s) => trainSet.has(s)).length;
  const novelValid = sents.filter((s) => !trainSet.has(s) && VALID.has(s)).length;
  return { valid_words: words.filter((w) => WORDS.has(w)).length / Math.max(1, words.length), sentences: sents.length,
    copy_rate: copy / Math.max(1, sents.length), novel_valid_rate: novelValid / Math.max(1, sents.length) };
}
function attnByDistance(m, ids, rng) {
  const acc = new Float64Array(9); let rows = 0;
  for (let w = 0; w < 30; w++) {
    const st = Math.floor(rng() * (ids.length - CFG.T - 1)); const c = m.forward(ids.subarray(st, st + CFG.T));
    for (let i = 8; i < CFG.T; i++) { rows++; for (let j = 0; j <= i; j++) acc[Math.min(i - j, 8)] += c.A[i * CFG.T + j]; }
  }
  return [...acc].map((v) => v / rows);
}

const lossCsv = path.join(RES, 'losses.csv'), genCsv = path.join(RES, 'generation.csv');
if (!fs.existsSync(lossCsv)) fs.writeFileSync(lossCsv, 'condition,seed,step,train_loss,T1,T2,T3,test_all\n');
if (!fs.existsSync(genCsv)) fs.writeFileSync(genCsv, 'condition,seed,step,valid_words,copy_rate,novel_valid_rate,sentences,attn_self,attn_1,attn_2,attn_3,attn_4,attn_5,attn_6,attn_7,attn_8plus\n');
const done = new Set(fs.readFileSync(lossCsv, 'utf8').trim().split('\n').slice(1).map((l) => l.split(',').slice(0, 3).join(',')));

for (const cond of CONDS) {
  const text = fs.readFileSync(path.join(TXT, cond + '.txt'), 'utf8');
  const ids = enc(text);
  for (let seed = 1; seed <= SEEDS; seed++) {
    if (done.has(`${cond},${seed},${STEPS}`)) { console.log('skip', cond, seed); continue; }
    const m = new TinyGPT({ V, T: CFG.T, d: CFG.d, f: CFG.f, seed: 1000 + seed });
    const rng = TinyGPT.mulberry32(2000 + seed);
    const t0 = Date.now();
    for (let step = 1; step <= STEPS; step++) {
      m.trainStep(ids, CFG.B, CFG.lr, rng);
      if (CHECK.includes(step)) {
        const T1 = testLoss(m, tests.T1), T2 = testLoss(m, tests.T2), T3 = testLoss(m, tests.T3);
        fs.appendFileSync(lossCsv, [cond, seed, step, lossOf(m, ids).toFixed(4), T1.toFixed(4), T2.toFixed(4), T3.toFixed(4), ((T1 + T2 + T3) / 3).toFixed(4)].join(',') + '\n');
        if (step === 1000 || step === STEPS) {
          const gs = genStats(generate(m, 2000, TinyGPT.mulberry32(3000 + seed)), text);
          const at = attnByDistance(m, ids, TinyGPT.mulberry32(4000 + seed));
          fs.appendFileSync(genCsv, [cond, seed, step, gs.valid_words.toFixed(4), gs.copy_rate.toFixed(4), gs.novel_valid_rate.toFixed(4), gs.sentences, ...at.map((v) => v.toFixed(4))].join(',') + '\n');
        }
      }
    }
    console.log(cond, 'seed', seed, `${((Date.now() - t0) / 1000).toFixed(0)}s`);
  }
}
