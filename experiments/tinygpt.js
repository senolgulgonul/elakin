// TinyGPT: one-block, single-head, character-level transformer with manual backprop.
(function (root) {
  function randn(rng) {
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  // C[n x m] = A[n x k] * B[k x m]   (row-major)
  function mm(A, B, n, k, m, C) {
    C = C || new Float32Array(n * m);
    C.fill(0);
    for (let i = 0; i < n; i++) {
      const ci = i * m, ai = i * k;
      for (let p = 0; p < k; p++) {
        const a = A[ai + p]; if (a === 0) continue;
        const bp = p * m;
        for (let j = 0; j < m; j++) C[ci + j] += a * B[bp + j];
      }
    }
    return C;
  }
  // G[k x m] += A^T[k x n] * D[n x m]
  function mmTA(A, D, n, k, m, G) {
    for (let i = 0; i < n; i++) {
      const ai = i * k, di = i * m;
      for (let p = 0; p < k; p++) {
        const a = A[ai + p]; if (a === 0) continue;
        const gp = p * m;
        for (let j = 0; j < m; j++) G[gp + j] += a * D[di + j];
      }
    }
  }
  // C[n x k] = D[n x m] * B^T  where B is [k x m]
  function mmTB(D, B, n, m, k, C) {
    C = C || new Float32Array(n * k);
    for (let i = 0; i < n; i++) {
      const di = i * m, ci = i * k;
      for (let p = 0; p < k; p++) {
        const bp = p * m; let s = 0;
        for (let j = 0; j < m; j++) s += D[di + j] * B[bp + j];
        C[ci + p] = s;
      }
    }
    return C;
  }

  function TinyGPT(cfg) {
    this.V = cfg.V; this.T = cfg.T; this.d = cfg.d; this.f = cfg.f;
    const rng = mulberry32(cfg.seed || 42);
    const V = this.V, T = this.T, d = this.d, f = this.f;
    const init = (n, s) => { const a = new Float32Array(n); for (let i = 0; i < n; i++) a[i] = randn(rng) * s; return a; };
    this.p = {
      E: init(V * d, 0.5), P: init(T * d, 0.2),
      Wq: init(d * d, 1 / Math.sqrt(d)), Wk: init(d * d, 1 / Math.sqrt(d)),
      Wv: init(d * d, 1 / Math.sqrt(d)), Wo: init(d * d, 0.5 / Math.sqrt(d)),
      W1: init(d * f, 1 / Math.sqrt(d)), b1: new Float32Array(f),
      W2: init(f * d, 0.5 / Math.sqrt(f)), b2: new Float32Array(d),
      Wu: init(d * V, 1 / Math.sqrt(d)), bu: new Float32Array(V)
    };
    this.g = {}; this.m = {}; this.v = {};
    for (const k in this.p) {
      const n = this.p[k].length;
      this.g[k] = new Float32Array(n); this.m[k] = new Float32Array(n); this.v[k] = new Float32Array(n);
    }
    this.t = 0;
  }

  TinyGPT.prototype.forward = function (x) {
    const n = x.length, d = this.d, f = this.f, V = this.V, p = this.p;
    const X = new Float32Array(n * d);
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) X[i * d + j] = p.E[x[i] * d + j] + p.P[i * d + j];
    const Q = mm(X, p.Wq, n, d, d), K = mm(X, p.Wk, n, d, d), Vv = mm(X, p.Wv, n, d, d);
    const sc = 1 / Math.sqrt(d);
    const S = mmTB(Q, K, n, d, n); // n x n raw scores
    for (let i = 0; i < n * n; i++) S[i] *= sc;
    const A = new Float32Array(n * n);
    for (let i = 0; i < n; i++) {
      let mx = -Infinity;
      for (let j = 0; j <= i; j++) mx = Math.max(mx, S[i * n + j]);
      let sum = 0;
      for (let j = 0; j <= i; j++) { const e = Math.exp(S[i * n + j] - mx); A[i * n + j] = e; sum += e; }
      for (let j = 0; j <= i; j++) A[i * n + j] /= sum;
    }
    const Z = mm(A, Vv, n, n, d);
    const H1 = mm(Z, p.Wo, n, d, d);
    for (let i = 0; i < n * d; i++) H1[i] += X[i];
    const U = mm(H1, p.W1, n, d, f);
    const R = new Float32Array(n * f);
    for (let i = 0; i < n; i++) for (let j = 0; j < f; j++) { const u = U[i * f + j] + p.b1[j]; U[i * f + j] = u; R[i * f + j] = u > 0 ? u : 0; }
    const H2 = mm(R, p.W2, n, f, d);
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) H2[i * d + j] += H1[i * d + j] + p.b2[j];
    const L = mm(H2, p.Wu, n, d, V);
    for (let i = 0; i < n; i++) for (let j = 0; j < V; j++) L[i * V + j] += p.bu[j];
    return { n, x, X, Q, K, Vv, S, A, Z, H1, U, R, H2, L };
  };

  // returns loss; accumulates grads scaled by `scale`
  TinyGPT.prototype.backward = function (c, y, scale) {
    const n = c.n, d = this.d, f = this.f, V = this.V, p = this.p, g = this.g;
    const dL = new Float32Array(n * V);
    let loss = 0;
    for (let i = 0; i < n; i++) {
      let mx = -Infinity;
      for (let j = 0; j < V; j++) mx = Math.max(mx, c.L[i * V + j]);
      let s = 0;
      for (let j = 0; j < V; j++) { const e = Math.exp(c.L[i * V + j] - mx); dL[i * V + j] = e; s += e; }
      for (let j = 0; j < V; j++) dL[i * V + j] /= s;
      loss += -Math.log(dL[i * V + y[i]] + 1e-12);
      dL[i * V + y[i]] -= 1;
      for (let j = 0; j < V; j++) dL[i * V + j] *= scale;
    }
    mmTA(c.H2, dL, n, d, V, g.Wu);
    for (let i = 0; i < n; i++) for (let j = 0; j < V; j++) g.bu[j] += dL[i * V + j];
    const dH2 = mmTB(dL, p.Wu, n, V, d);
    const dH1 = Float32Array.from(dH2);
    mmTA(c.R, dH2, n, f, d, g.W2);
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) g.b2[j] += dH2[i * d + j];
    const dU = mmTB(dH2, p.W2, n, d, f);
    for (let i = 0; i < n * f; i++) if (c.U[i] <= 0) dU[i] = 0;
    mmTA(c.H1, dU, n, d, f, g.W1);
    for (let i = 0; i < n; i++) for (let j = 0; j < f; j++) g.b1[j] += dU[i * f + j];
    const t1 = mmTB(dU, p.W1, n, f, d);
    for (let i = 0; i < n * d; i++) dH1[i] += t1[i];
    const dX = Float32Array.from(dH1);
    mmTA(c.Z, dH1, n, d, d, g.Wo);
    const dZ = mmTB(dH1, p.Wo, n, d, d);
    const dA = mmTB(dZ, c.Vv, n, d, n);
    const dVv = new Float32Array(n * d);
    mmTA(c.A, dZ, n, n, d, dVv);
    const sc = 1 / Math.sqrt(d);
    const dS = new Float32Array(n * n);
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0; j <= i; j++) s += dA[i * n + j] * c.A[i * n + j];
      for (let j = 0; j <= i; j++) dS[i * n + j] = c.A[i * n + j] * (dA[i * n + j] - s) * sc;
    }
    const dQ = mm(dS, c.K, n, n, d);
    const dK = new Float32Array(n * d);
    mmTA(dS, c.Q, n, n, d, dK);
    mmTA(c.X, dQ, n, d, d, g.Wq);
    mmTA(c.X, dK, n, d, d, g.Wk);
    mmTA(c.X, dVv, n, d, d, g.Wv);
    const a1 = mmTB(dQ, p.Wq, n, d, d), a2 = mmTB(dK, p.Wk, n, d, d), a3 = mmTB(dVv, p.Wv, n, d, d);
    for (let i = 0; i < n * d; i++) dX[i] += a1[i] + a2[i] + a3[i];
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) {
      g.E[c.x[i] * d + j] += dX[i * d + j];
      g.P[i * d + j] += dX[i * d + j];
    }
    return loss;
  };

  TinyGPT.prototype.zeroGrad = function () { for (const k in this.g) this.g[k].fill(0); };

  TinyGPT.prototype.adam = function (lr) {
    this.t++;
    const b1 = 0.9, b2 = 0.999, eps = 1e-8, t = this.t;
    const c1 = 1 - Math.pow(b1, t), c2 = 1 - Math.pow(b2, t);
    // global grad-norm clipping
    let norm = 0;
    for (const k in this.g) { const G = this.g[k]; for (let i = 0; i < G.length; i++) norm += G[i] * G[i]; }
    norm = Math.sqrt(norm);
    const clip = norm > 1 ? 1 / norm : 1;
    for (const k in this.p) {
      const P = this.p[k], G = this.g[k], M = this.m[k], S = this.v[k];
      for (let i = 0; i < P.length; i++) {
        const gr = G[i] * clip;
        M[i] = b1 * M[i] + (1 - b1) * gr;
        S[i] = b2 * S[i] + (1 - b2) * gr * gr;
        P[i] -= lr * (M[i] / c1) / (Math.sqrt(S[i] / c2) + eps);
      }
    }
  };

  // one optimisation step on a batch of windows taken from ids
  TinyGPT.prototype.trainStep = function (ids, B, lr, rng) {
    const T = this.T;
    this.zeroGrad();
    let loss = 0;
    const scale = 1 / (B * T);
    for (let b = 0; b < B; b++) {
      const s = Math.floor(rng() * (ids.length - T - 1));
      const x = ids.subarray(s, s + T), y = ids.subarray(s + 1, s + T + 1);
      const c = this.forward(x);
      loss += this.backward(c, y, scale);
    }
    this.adam(lr);
    return loss / (B * T);
  };

  TinyGPT.prototype.evalLoss = function (ids, nWin, rng) {
    const T = this.T; let loss = 0;
    for (let b = 0; b < nWin; b++) {
      const s = Math.floor(rng() * (ids.length - T - 1));
      const c = this.forward(ids.subarray(s, s + T));
      const y = ids.subarray(s + 1, s + T + 1), V = this.V;
      for (let i = 0; i < T; i++) {
        let mx = -Infinity; for (let j = 0; j < V; j++) mx = Math.max(mx, c.L[i * V + j]);
        let sum = 0; for (let j = 0; j < V; j++) sum += Math.exp(c.L[i * V + j] - mx);
        loss += Math.log(sum) + mx - c.L[i * V + y[i]];
      }
    }
    return loss / (nWin * T);
  };

  TinyGPT.prototype.nextProbs = function (ctx, temp) {
    const x = ctx.slice(-this.T);
    const c = this.forward(x), V = this.V, n = x.length;
    const out = new Float32Array(V);
    let mx = -Infinity;
    for (let j = 0; j < V; j++) mx = Math.max(mx, c.L[(n - 1) * V + j] / temp);
    let s = 0;
    for (let j = 0; j < V; j++) { out[j] = Math.exp(c.L[(n - 1) * V + j] / temp - mx); s += out[j]; }
    for (let j = 0; j < V; j++) out[j] /= s;
    return out;
  };

  TinyGPT.mulberry32 = mulberry32;
  if (typeof module !== 'undefined') module.exports = Object.assign(TinyGPT, { mulberry32 }); else root.TinyGPT = TinyGPT;
})(this);
