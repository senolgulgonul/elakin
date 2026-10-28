# Elakin

**Train a tiny language model on your own text, in your browser, and watch every step.**

**Live page:** https://senolgulgonul.github.io/elakin/

Elakin is a single web page that shows how a large language model works, using a model small enough to look inside. You paste or upload a text, and the page walks through every stage: cutting the text into tokens, building a vocabulary, turning tokens into embeddings, computing attention, training, and finally generating new text.

The name comes from **e, l, a, k, i, n**: the first six letters Turkish children learn to read. The built-in Turkish sample text uses only those letters.

## The eight steps

1. **Your text** – paste text, load a `.txt` file, or use one of the two samples (Turkish primer, English story). The last 10% is held out to test the model on text it has never seen.
2. **Tokens** – the same text cut five ways: characters (what this model uses), BPE trained on your text, and the GPT-2, GPT-4 and GPT-4o tokenizers.
3. **Vocabulary** – every character with its ID and frequency, and your sentence as a list of numbers.
4. **Embeddings** – token and position embeddings as heatmaps.
5. **Attention** – raw scores, the causal mask and softmax weights, row by row.
6. **Training** – live training and held-out loss curves, plus samples written as the model learns. Training pauses by itself at 500, 1,000 and 2,000 steps.
7. **What attention learned** – attention before and after training, and how far back the model looks on average.
8. **Generate** – write new text with an adjustable temperature, and see the odds for the next character.

The copy icon copies all results as plain text, ready to paste into an AI chat for an explanation.

## The model

- One Transformer block: a single causal attention head, an MLP with ReLU, residual connections, no layer norm.
- Character-level tokens, context of 32 characters, embedding size 32, MLP size 64 (about 10,000–13,000 parameters, depending on the vocabulary).
- Forward and backward passes written from scratch in plain JavaScript (no ML library), trained with Adam.
- Fixed random seeds: the same text and settings give the same training run on every computer.

## Running it

Open `index.html` in a browser. No build step and no server needed.

To publish with GitHub Pages: put `index.html` in the root of a repository, then enable Pages under *Settings → Pages*.

Everything runs in the browser and your text is never uploaded. The page loads only two things from the internet:

- the GPT-2, GPT-4 and GPT-4o tokenizers from [gpt-tokenizer](https://www.npmjs.com/package/gpt-tokenizer) 2.9.0 via jsDelivr, only when you open those tabs;
- the Instrument Sans and JetBrains Mono fonts from Google Fonts (system fonts are used if they don't load).

## About the Turkish sample

The Turkish sample text was written for this page. It follows the letter order of Turkish sound-based early reading instruction, where the first group of letters is E, L, A, K, İ, N, and uses only those letters, with short, repetitive sentences.

## License

Elakin is released under the [MIT License](LICENSE). The tokenizers it loads come from [gpt-tokenizer](https://github.com/niieani/gpt-tokenizer) (MIT License); the fonts are from Google Fonts (SIL Open Font License).

---

### Türkçe özet

Elakin, büyük dil modellerinin nasıl çalıştığını, içine bakılabilecek kadar küçük bir modelle gösteren tek sayfalık bir web uygulamasıdır. Metninizi yükleyin; sayfa tokenizasyondan metin üretimine kadar her adımı gösterir. Her şey tarayıcınızda çalışır, metniniz hiçbir yere gönderilmez. İsim, Türkçe ilk okuma-yazmada öğretilen ilk altı harften gelir: e, l, a, k, i, n.

---

Elakin v1.1 · Şenol Gülgönül · MIT License
