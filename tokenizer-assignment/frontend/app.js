"use strict";

// Try several locations so it works locally AND on Netlify whether you deploy the
// repo root or drop the data next to index.html.
const DATA_PATHS = ["../results/results.json", "./results.json", "results/results.json"];
const tokenPaths = (k) => [`../results/tokens_${k}.json`, `./tokens_${k}.json`, `results/tokens_${k}.json`];
const KEY = { English: "en", Hindi: "hi", Telugu: "te", Marathi: "mr" };
const VAR = { English: "--lang-en", Hindi: "--lang-hi", Telugu: "--lang-te", Marathi: "--lang-mr" };
const CHIP_COLORS = ["#4cc9f0", "#7c5cff", "#06d6a0", "#ffd166", "#ff8c42", "#ef476f", "#b088ff", "#59d2fe"];

async function fetchFirst(paths) {
  for (const p of paths) {
    try { const r = await fetch(p, { cache: "no-store" }); if (r.ok) return await r.json(); }
    catch (_) { /* next */ }
  }
  throw new Error("not found: " + paths.join(", "));
}

const esc = (s) => String(s).replace(/[&<>"]/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const showWS = (t) => esc(t).replace(/ /g, '<span class="ws">·</span>');
const lvar = (name) => `var(${VAR[name] || "--accent"})`;

/* ---------------- hero ---------------- */
function hero(d) {
  const s = d.scoring, v = d.vocab_stats;
  const eqScore = v.equal_weight_gap ? Math.round(1000 / v.equal_weight_gap) : null;
  const badge = (v.equal_weight_gap && v.tuned_gap)
    ? `<div class="improve">Fertility gap tuned <b>${v.equal_weight_gap} → ${v.tuned_gap}</b>
        &nbsp;·&nbsp; score <b>${eqScore?.toLocaleString()} → ${s.score?.toLocaleString()}</b></div>`
    : "";
  return `<section class="hero">
    <p class="eyebrow">Language-Balanced BPE · 10,000 tokens · EN · HI · TE · MR</p>
    <h1>India — Wikipedia in 4 Languages</h1>
    <div class="byline">by <b>Tejaskumar Reddy J</b></div>
    <div class="score">${s.score?.toLocaleString() ?? "—"}</div>
    <div class="score-label">Balance Score = 1000 / (X_max − X_min)</div>
    <div class="score-formula">= 1000 / (${s.max_ratio} − ${s.min_ratio}) = 1000 / ${s.delta}</div>
    ${badge}
  </section>`;
}

/* ---------------- methodology (main page explainer) ---------------- */
function methodology(d) {
  const v = d.vocab_stats;
  const w = v.allocation_weights || {};
  const steps = [
    ["Fetch &amp; clean", "Pulled the <b>India</b> article from English, Hindi, Telugu and Marathi Wikipedia. Stripped HTML, infoboxes, tables, navigation and citation markers down to plain article prose."],
    ["4 monolingual tokenizers", "Trained a separate BPE tokenizer on each language with <i>identical</i> preprocessing: Unicode NFC normalization, Whitespace pre-tokenization, no case-folding, and the full per-language alphabet seeded so rare Indic conjuncts are never dropped as <code>&lt;unk&gt;</code>."],
    ["Weighted merge &amp; interleave", "Interleaved the four languages' merge rules in a <b>weighted round-robin</b> and replayed them into one unified <b>10,000-token</b> vocabulary — so English's larger byte volume can't monopolise the merges."],
    ["Tune for fairness", `Searched the merge-budget weights and kept the split with the smallest <b>fertility gap</b>. Chosen weights <code>en:${w.en} · hi:${w.hi} · te:${w.te} · mr:${w.mr}</code> cut the gap from <b>${v.equal_weight_gap}</b> (equal split) to <b>${v.tuned_gap}</b>. No held-out data touched — only the vocab budget is balanced.`],
    ["Evaluate &amp; score", "Tokenized each full article with the one merged tokenizer, measured <b>fertility = tokens ÷ words</b> per language, sorted them, and scored <code>1000 / (X_max − X_min)</code>. A smaller gap ⇒ fairer tokenizer ⇒ higher score."],
  ];
  const li = steps.map(([h, b], i) =>
    `<div class="step"><div class="n">${i + 1}</div><div><h4>${h}</h4><p>${b}</p></div></div>`).join("");
  return `<h2 class="section">How we built this</h2>
    <div class="method">${li}</div>`;
}

/* ---------------- top-level nav ---------------- */
const NAV = [["overview", "Overview"], ["how", "How it works"], ["languages", "Languages"], ["vocab", "Vocabulary"]];
function nav() {
  return `<div class="navbar">` + NAV.map(([id, label], i) =>
    `<button class="nav-btn${i === 0 ? " active" : ""}" data-view="${id}">${label}</button>`).join("") + `</div>`;
}
const view = (id, html, active) => `<div class="view${active ? " active" : ""}" data-view="${id}">${html}</div>`;

/* ---------------- code structure & flow ---------------- */
function codeFlow() {
  const tree =
`tokenizer-assignment/
├── data/
│   ├── raw/            raw Wikipedia HTML per language
│   └── clean/          cleaned plain-text article per language
├── scripts/
│   ├── fetch_data.py       fetch + clean the 4 Wikipedia pages
│   ├── train_tokenizer.py  train 4 mono BPEs, weighted-merge to 10k, tune
│   ├── evaluate.py         fertility ratios + score  ->  results.json
│   └── utils.py            shared config, NFC, word split
├── tokenizer/
│   ├── merged_vocab.json   final 10k tokenizer (inspectable)
│   └── mono_*.json         the 4 monolingual tokenizers
├── results/
│   ├── results.json        structured output the frontend reads
│   └── tokens_*.json       full ordered token stream per language
└── frontend/           this static site (HTML/CSS/JS, Netlify-ready)`;
  const flow = ["fetch_data.py", "clean text", "train_tokenizer.py", "4 monolingual BPEs",
    "weighted round-robin merge", "tune allocation → min gap", "merged_vocab.json",
    "evaluate.py", "results.json", "frontend"];
  const chips = flow.map((f) => `<span class="flow-node">${esc(f)}</span>`).join('<span class="flow-arrow">→</span>');

  // the three commands that reproduce everything, shown as a mac terminal session
  const term = macTerm("tejaskumar — python — 80×24", [
    `<span class="c-cmt"># reproduce the whole pipeline end to end</span>`,
    `<span class="c-usr">tejaskumar@era-v5</span>:<span class="c-dir">~/tokenizer-assignment</span>$ python scripts/fetch_data.py`,
    `<span class="c-out">English (en): 14,019 words · Hindi 7,534 · Telugu 2,646 · Marathi 4,529</span>`,
    `<span class="c-usr">tejaskumar@era-v5</span>:<span class="c-dir">~/tokenizer-assignment</span>$ python scripts/train_tokenizer.py`,
    `<span class="c-out">Best-weight gap: 0.1137  score=8795.9  weights={en:4, hi:1, te:3, mr:3}</span>`,
    `<span class="c-usr">tejaskumar@era-v5</span>:<span class="c-dir">~/tokenizer-assignment</span>$ python scripts/evaluate.py</span>`,
    `<span class="c-out">X_max=1.6928  X_min=1.5792  delta=0.1136  <span class="c-ok">SCORE=8802.82</span></span>`,
    `<span class="c-usr">tejaskumar@era-v5</span>:<span class="c-dir">~/tokenizer-assignment</span>$ <span class="cursor">▋</span>`,
  ].join("\n"));

  const codeSnips = [
    ["train_tokenizer.py — weighted round-robin merge (the balancing core)",
`while len(vocab) < TOTAL_VOCAB:
    for k in keys:                 # en, hi, te, mr
        for _ in range(weights[k]):    # more turns = more merges = lower fertility
            a, b = next_applicable_merge(k)
            vocab[a + b] = len(vocab)  # grow the shared 10k vocab`],
    ["evaluate.py — fertility &amp; score",
`ratio = token_count / word_count        # fertility: tokens per word
sorted_ratios = sorted(ratios)
score = 1000 / (max(ratios) - min(ratios))   # smaller gap -> higher score`],
  ].map(([t, code]) =>
    `<div class="subhead" style="margin-top:22px">${t}</div>` + macTerm(t.split(" —")[0], `<span class="code">${esc(code)}</span>`, true)).join("");

  return `<h2 class="section">Code structure &amp; flow</h2>
    <div class="chart">
      <div class="subhead">Pipeline — data in, score &amp; widget out</div>
      <div class="flow">${chips}</div>
      <div class="subhead" style="margin-top:24px">Run it</div>
      ${term}
      <div class="subhead" style="margin-top:24px">Project layout</div>
      ${macTerm("tokenizer-assignment", `<span class="code">${esc(tree)}</span>`, true)}
      ${codeSnips}
    </div>`;
}

// mac-style terminal window chrome
function macTerm(title, bodyHTML, plain) {
  return `<div class="mac-term${plain ? " plain" : ""}">
    <div class="mac-bar">
      <span class="dot r"></span><span class="dot y"></span><span class="dot g"></span>
      <span class="mac-title">${esc(title)}</span>
    </div>
    <pre class="mac-body">${bodyHTML}</pre>
  </div>`;
}

/* ---------------- sorted fertility chart ---------------- */
function chart(d) {
  const ranked = d.scoring.sorted_ratios;
  const maxLang = ranked[ranked.length - 1].lang, minLang = ranked[0].lang;
  const scaleMax = Math.max(...ranked.map((r) => r.ratio)) * 1.08;
  const rows = ranked.map((r) => {
    const pct = (r.ratio / scaleMax) * 100;
    const tag = r.lang === maxLang ? '<span class="tag max">X_max</span>'
      : r.lang === minLang ? '<span class="tag min">X_min</span>' : "";
    return `<div class="bar-row">
      <div class="lab">${esc(r.lang)}${tag}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${pct}%;background:${lvar(r.lang)}"></div></div>
      <div class="val">${r.ratio.toFixed(4)}</div>
    </div>`;
  }).join("");
  return `<h2 class="section">Fertility (tokens / word) — sorted</h2>
    <div class="chart">${rows}
    <div class="caption">Lower fertility = fewer tokens per word = more efficient.
    All four now sit between ${ranked[0].ratio} and ${ranked[ranked.length - 1].ratio} — a tight spread
    (X_max ${esc(maxLang)}, X_min ${esc(minLang)}) is exactly what drives a high score.</div></div>`;
}

/* ---------------- per-language tabs ---------------- */
function tabs(d) {
  const btns = d.languages.map((l, i) =>
    `<button class="tab-btn${i === 0 ? " active" : ""}" data-lang="${esc(l.name)}"
       style="--lc:${lvar(l.name)}">${esc(l.name)}</button>`).join("");
  const panels = d.languages.map((l, i) => langPanel(l, i === 0)).join("");
  return `<h2 class="section">Per-language deep dive</h2>
    <div class="tabbar">${btns}</div>
    <div id="panels">${panels}</div>`;
}

function langPanel(l, active) {
  const s = l.sample_tokenized_sentence;
  const chips = s.tokens.map((t, i) =>
    `<span class="chip" style="background:${CHIP_COLORS[i % CHIP_COLORS.length]}22;
      border-color:${CHIP_COLORS[i % CHIP_COLORS.length]}55">${showWS(t)}</span>`).join("");
  const top = l.top_tokens.map((t) =>
    `<span class="toptok">${showWS(t.token)}<b>${t.count}</b></span>`).join("");
  const metric = (v, k) => `<div class="metric"><div class="v">${v}</div><div class="k">${k}</div></div>`;
  return `<section class="panel${active ? " active" : ""}" data-lang="${esc(l.name)}"
      style="--lc:${lvar(l.name)}">
    <div class="panel-head">
      <h3>${esc(l.name)}</h3>
      <a class="wiki" href="${esc(l.wiki_url)}" target="_blank" rel="noopener">source article ↗</a>
    </div>
    <div class="metrics wide">
      ${metric(l.word_count.toLocaleString(), "words")}
      ${metric(l.token_count.toLocaleString(), "tokens")}
      ${metric(l.ratio.toFixed(4), "fertility (t/w)")}
      ${metric((l.final_vocab_allocated ?? l.vocab_allocated).toLocaleString(), "vocab in final 10k")}
      ${metric(l.own_tokens.toLocaleString(), "own-vocab tokens")}
      ${metric(l.borrowed_tokens.toLocaleString(), "borrowed tokens")}
    </div>
    <div class="subhead">Sample sentence — tokenized (${s.tokens.length} tokens)</div>
    <div class="orig">"${esc(s.original)}"</div>
    <div class="chips">${chips}</div>
    <div class="subhead">Top 30 tokens in this article</div>
    <div class="toptokens">${top}</div>
    <div class="subhead">Full tokenized article — every token in order (${l.token_count.toLocaleString()})</div>
    <input class="vsearch stream-search" data-lang="${esc(l.name)}" type="search"
      placeholder="Search this article's tokens…" />
    <div class="caption stream-count" data-lang="${esc(l.name)}">loading…</div>
    <div class="vgrid stream-grid" data-lang="${esc(l.name)}"></div>
  </section>`;
}

// lazy-load + render one language's full token stream, with search
async function loadStream(name) {
  const grid = document.querySelector(`.stream-grid[data-lang="${cssq(name)}"]`);
  if (grid.dataset.loaded) return;
  grid.dataset.loaded = "1";
  const count = document.querySelector(`.stream-count[data-lang="${cssq(name)}"]`);
  const search = document.querySelector(`.stream-search[data-lang="${cssq(name)}"]`);
  let toks;
  try { toks = await fetchFirst(tokenPaths(KEY[name])); }
  catch (e) { count.textContent = "could not load full stream (" + e.message + ")"; return; }
  const draw = () => {
    const q = search.value.trim().toLowerCase();
    const hits = q ? toks.filter((t) => t.toLowerCase().includes(q)) : toks;
    grid.innerHTML = hits.map((t, i) =>
      `<span class="chip" style="background:${CHIP_COLORS[i % CHIP_COLORS.length]}18;
        border-color:${CHIP_COLORS[i % CHIP_COLORS.length]}44">${showWS(t)}</span>`).join("");
    count.textContent = q
      ? `${hits.length.toLocaleString()} match "${q}"`
      : `${toks.length.toLocaleString()} tokens total — all shown, in order; search to filter`;
  };
  search.addEventListener("input", draw);
  draw();
}

const cssq = (s) => s.replace(/"/g, '\\"');

/* ---------------- global vocab viewer ---------------- */
function vocabList(d) {
  const all = d.vocab_stats.all_tokens || [];
  return `<h2 class="section">Full tokenizer vocabulary — ${all.length.toLocaleString()} tokens</h2>
    <div class="chart">
      <input id="vocab-search" class="vsearch" type="search"
        placeholder="Search all ${all.length.toLocaleString()} tokens (e.g. भारत, the, ా)…" />
      <div id="vocab-count" class="caption"></div>
      <div id="vocab-grid" class="vgrid"></div>
    </div>`;
}
function renderVocab(all) {
  const grid = document.getElementById("vocab-grid");
  const count = document.getElementById("vocab-count");
  const search = document.getElementById("vocab-search");
  const draw = () => {
    const q = search.value.trim().toLowerCase();
    const hits = q ? all.filter((t) => t.toLowerCase().includes(q)) : all;
    grid.innerHTML = hits.map((t) => `<span class="vtok">${showWS(t)}</span>`).join("");
    count.textContent = q
      ? `${hits.length.toLocaleString()} match "${q}"`
      : `all ${all.length.toLocaleString()} tokens shown — search to find any token`;
  };
  search.addEventListener("input", draw);
  draw();
}

/* ---------------- notes ---------------- */
function notes(d) {
  const v = d.vocab_stats;
  const li = d.notes.map((n) => `<li>${esc(n)}</li>`).join("");
  const alloc = Object.entries(v.per_language_final_allocation).map(([k, n]) => `${k}: ${n}`).join(" · ");
  return `<h2 class="section">Methodology &amp; caveats</h2>
    <div class="notes"><ul>${li}</ul>
    <div class="vocab-line">Final unified vocab: <b>${v.total_vocab_size.toLocaleString()}</b> tokens ·
      overlap shared by all 4 languages: <b>${v.overlap_tokens}</b> ·
      final contribution — ${alloc}</div></div>`;
}

/* ---------------- wire up ---------------- */
async function main() {
  const app = document.getElementById("app");
  try {
    const d = await load();
    app.innerHTML = hero(d) + nav() +
      view("overview", chart(d) + notes(d), true) +
      view("how", methodology(d) + codeFlow(), false) +
      view("languages", tabs(d), false) +
      view("vocab", vocabList(d), false);
    renderVocab(d.vocab_stats.all_tokens || []);

    // top-level view switching
    app.querySelectorAll(".nav-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const id = btn.dataset.view;
        app.querySelectorAll(".nav-btn").forEach((b) => b.classList.toggle("active", b === btn));
        app.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.dataset.view === id));
        window.scrollTo({ top: 0, behavior: "smooth" });
        if (id === "languages") loadStream(d.languages[0].name);
      });
    });

    // per-language tab switching + lazy stream load
    loadStream(d.languages[0].name);
    app.querySelectorAll(".tab-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const name = btn.dataset.lang;
        app.querySelectorAll(".tab-btn").forEach((b) => b.classList.toggle("active", b === btn));
        app.querySelectorAll(".panel").forEach((p) =>
          p.classList.toggle("active", p.dataset.lang === name));
        loadStream(name);
      });
    });
  } catch (e) {
    app.innerHTML = `<div class="error">Could not load data.<br><code>${esc(e.message)}</code>
      <br><br>Run <code>python scripts/evaluate.py</code>, then serve this folder.</div>`;
  }
}
const load = () => fetchFirst(DATA_PATHS);
main();
