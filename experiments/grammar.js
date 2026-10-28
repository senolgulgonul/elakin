// Sentence patterns for the ELAKİN language. Each pattern lists every sentence it can make.
const L = require('./lexicon.json');

const cap = (s) => (s[0] === 'i' ? 'İ' : s[0].toLocaleUpperCase('tr')) + s.slice(1);
const persons = [...L.names, ...L.family];
const places = Object.values(L.place);
const countable = L.take.filter((x) => x !== 'ekin'); // "iki ekin" (two crops) reads oddly

function* pairs(a, b, distinct) { for (const x of a) for (const y of b) if (!distinct || x !== y) yield [x, y]; }

// each entry: [id, generator of {s: sentence, w: content words}]
const patterns = {
  P1_take_comma: function* () { for (const [p, x] of pairs(persons, L.take)) yield { s: `${cap(p)}, ${x} al.`, w: [p, x] }; },
  P2_take: function* () { for (const [p, x] of pairs(persons, L.take)) yield { s: `${cap(p)} ${x} al.`, w: [p, x] }; },
  P3_take_two: function* () { for (const [p, x] of pairs(persons, countable)) yield { s: `${cap(p)} iki ${x} al.`, w: [p, x] }; },
  P4_plant: function* () { for (const [p, x] of pairs(persons, L.plant)) yield { s: `${cap(p)} ${x} ek.`, w: [p, x] }; },
  P5_plant_field: function* () { for (const [p, x] of pairs(persons, L.plant)) yield { s: `${cap(p)} alana ${x} ek.`, w: [p, x] }; },
  P6_spice: function* () { for (const [p, x] of pairs(persons, L.spice)) yield { s: `${cap(p)} keke ${x} ekle.`, w: [p, x] }; },
  P7_hand_in_hand: function* () { for (const [p, q] of pairs(persons, persons, true)) yield { s: `${cap(p)} ile ${q} el ele.`, w: [p, q] }; },
  P8_intransitive: function* () { for (const [p, v] of pairs(persons, L.intransitive)) for (const e of ['.', '!']) yield { s: `${cap(p)}, ${v}${e}`, w: [p, v] }; },
  P9_go_down: function* () { for (const [p, y] of pairs(persons, places)) yield { s: `${cap(p)}, ${y} in.`, w: [p, y] }; },
  P10_made_with: function* () { for (const [o, m] of L.made_with) yield { s: `${cap(o)} ne ile? ${cap(m)} ile.`, w: [o, m] }; },
  P11_described: function* () { for (const [n, a] of L.described) yield { s: `${cap(n)} ${a}.`, w: [n, a] }; },
  P12_by_hand: function* () { for (const [p, x] of pairs(persons, L.take)) yield { s: `${cap(p)} elle ${x} al.`, w: [p, x] }; },
  P13_while: function* () { for (const [p, q] of pairs(persons, persons, true)) for (const x of L.take) yield { s: `${cap(p)} ${x} alken ${q} kal.`, w: [p, x, q] }; },
  P14_possessive: function* () { for (const g of Object.values(L.possessive)) for (const [o, a] of L.owned) yield { s: `${cap(g)} ${o} ${a}.`, w: [g, o, a] }; },
  P15_with_animal: function* () { for (const [p, z] of pairs(persons, L.animal)) yield { s: `${cap(p)}, ${z} ile kal.`, w: [p, z] }; },
  P16_sift: function* () { for (const p of persons) { yield { s: `${cap(p)} ekin ele.`, w: [p, 'ekin'] }; yield { s: `${cap(p)} elekle ekin ele.`, w: [p, 'ekin', 'elek'] }; } },
  P17_two_and_two: function* () { for (const [x, y] of pairs(countable, countable, true)) yield { s: `İki ${x}, iki ${y}.`, w: [x, y] }; }
};

// Long two-clause sentences, used only in test subset T3.
function* longSentences() {
  for (const [p, q] of pairs(persons, persons, true)) for (const x of L.take) for (const y of L.plant)
    yield { s: `${cap(p)} ${x} alken ${q} alana ${y} ek.`, w: [p, x, q, y], pattern: 'LONG' };
}

function language() {
  const out = [];
  for (const [id, gen] of Object.entries(patterns)) for (const item of gen()) out.push({ ...item, pattern: id });
  return out;
}

module.exports = { language, longSentences, patterns, persons, cap, lexicon: L };
