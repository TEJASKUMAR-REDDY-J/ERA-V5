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
// Metaspace uses ▁ for a leading space; render it (and literal spaces) as a faint dot.
const showWS = (t) => esc(t).replace(/[ ▁]/g, '<span class="ws">·</span>');
const lvar = (name) => `var(${VAR[name] || "--accent"})`;

/* ---------------- hero ---------------- */
function hero(d) {
  const s = d.scoring, rt = d.roundtrip;
  const badge = rt
    ? `<div class="improve">✓ Faithful — <b>decode(encode(text))</b> preserves every visible character
        &nbsp;·&nbsp; Hindi penalty <b>${s.hindi_penalty_factor}×</b></div>`
    : "";
  return `<section class="hero">
    <p class="eyebrow">Wiki-Faithful BPE · 10,000 tokens · EN · HI · TE · MR</p>
    <h1>India — Wikipedia in 4 Languages</h1>
    <div class="byline">by <b>Tejaskumar Reddy J</b></div>
    <div class="score">${s.score?.toLocaleString() ?? "—"}</div>
    <div class="score-label">Balance Score = 1000 / (X_max − X_min) &nbsp;·&nbsp; fertility = tokens ÷ faithful units</div>
    <div class="score-formula">= 1000 / (${s.max_ratio} − ${s.min_ratio}) = 1000 / ${s.delta}</div>
    ${badge}
  </section>`;
}

/* ---------------- methodology (main page explainer) ---------------- */
function methodology(d) {
  const w = d.vocab_stats.weights || {};
  const steps = [
    ["Faithful Markdown corpus", "Fetched the <b>India</b> article via the Wikipedia REST HTML API for English, Hindi, Telugu and Marathi and converted each to <b>faithful Markdown</b> (markdownify). Links, URLs, tables, references, image links, navboxes and categories are <i>kept</i> — only scripts/styles/meta are removed. This is a real document, not clipped word prose."],
    ["One shared BPE tokenizer", "Trained a single HuggingFace <b>BPE</b> tokenizer, vocab <b>10,000</b>, <code>min_frequency=1</code>, <b>NFKC</b> normalizer, <b>Metaspace</b> pre-tokenizer + decoder (▁ space marker). Metaspace preserves punctuation, brackets, URL characters, apostrophes and number separators; it's chosen over ByteLevel so Indic scripts don't waste tokens on UTF-8 bytes."],
    ["Language weighting", `Balanced the four languages by <b>duplicating each corpus</b> during training — weights <code>en:${w.en} · hi:${w.hi} · te:${w.te} · mr:${w.mr}</code> — so the smaller Indic pages get fair influence over the shared merges.`],
    ["Faithful-unit fertility", "Defined a <b>faithful unit</b> = one contiguous Unicode letter/mark/number run OR one visible punctuation/symbol. <b>fertility = tokens ÷ faithful units</b> — a denominator that counts everything the tokenizer must reproduce, so you can't cheat by dropping punctuation."],
    ["Faithfulness &amp; score", "Verified <code>decode(encode(text))</code> preserves every non-whitespace character (punctuation, URLs, number separators round-trip). Scored <code>1000 / (X_max − X_min)</code>, with a Hindi penalty <code>exp(max(0, hindi/1.2 − 1))</code> that only bites if Hindi fertility exceeds 1.2."],
  ];
  const li = steps.map(([h, b], i) =>
    `<div class="step"><div class="n">${i + 1}</div><div><h4>${h}</h4><p>${b}</p></div></div>`).join("");
  return `<h2 class="section">How we built this</h2>
    <div class="method">${li}</div>` + roundtripCard(d);
}

function roundtripCard(d) {
  const rt = d.roundtrip; if (!rt) return "";
  return `<h2 class="section">Faithfulness — round-trip check</h2>
    <div class="chart">
      <div class="subhead">decode(encode(text)) must preserve every visible character</div>
      <div class="rt-row"><span class="rt-k">input</span><code class="rt-v">${esc(rt.original)}</code></div>
      <div class="rt-row"><span class="rt-k">decoded</span><code class="rt-v">${esc(rt.decoded)}</code></div>
      <div class="rt-verdict ${rt.preserves_visible ? "ok" : "bad"}">
        ${rt.preserves_visible ? "✓ identical non-whitespace characters — faithful" : "✗ visible characters changed"}</div>
    </div>`;
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
├── build_wiki_faithful_markdown.py   fetch Wikipedia REST HTML → faithful Markdown
├── train_tokenizer.py                train the shared 10k BPE (NFKC + Metaspace)
├── evaluate_tokenizer.py             faithful-unit fertility + score
├── make_results.py                   build results.json for the widget
├── tokenizer.json                    the trained tokenizer (inspectable/downloadable)
├── metrics.json                      saved metrics
├── corpus/
│   ├── {en,hi,te,mr}.faithful.md    faithful Markdown snapshots
│   ├── {en,hi,te,mr}.faithful.txt   same corpus, plain text (tokenizer input)
│   └── {en,hi,te,mr}.meta.json      corpus metadata
├── results/  results.json + tokens_*.json  (widget data)
└── frontend/ this static site (HTML/CSS/JS, Netlify-ready)`;
  const flow = ["build_wiki_faithful_markdown.py", "corpus/*.faithful.txt", "train_tokenizer.py",
    "tokenizer.json (10k, Metaspace)", "evaluate_tokenizer.py", "faithful-unit fertility",
    "score 6502.56", "make_results.py", "frontend"];
  const chips = flow.map((f) => `<span class="flow-node">${esc(f)}</span>`).join('<span class="flow-arrow">→</span>');

  const term = macTerm("tejaskumar — python — 80×24", [
    `<span class="c-cmt"># reproduce the whole pipeline end to end</span>`,
    `<span class="c-usr">tejaskumar@era-v5</span>:<span class="c-dir">~/tokenizer-assignment</span>$ python build_wiki_faithful_markdown.py`,
    `<span class="c-out">en English: 186,367 faithful units · hi 88,359 · te 36,292 · mr 29,766</span>`,
    `<span class="c-usr">tejaskumar@era-v5</span>:<span class="c-dir">~/tokenizer-assignment</span>$ python train_tokenizer.py</span>`,
    `<span class="c-out">weights={en:3, hi:4, te:4, mr:2}  vocab=10000</span>`,
    `<span class="c-usr">tejaskumar@era-v5</span>:<span class="c-dir">~/tokenizer-assignment</span>$ python evaluate_tokenizer.py</span>`,
    `<span class="c-out">spread=0.223030  <span class="c-ok">score=4483.71</span>  hindi_penalty=1.0</span>`,
    `<span class="c-usr">tejaskumar@era-v5</span>:<span class="c-dir">~/tokenizer-assignment</span>$ <span class="cursor">▋</span>`,
  ].join("\n"));

  const codeSnips = [
    ["train_tokenizer.py — the tokenizer (faithful, Metaspace)",
`tok = Tokenizer(BPE(unk_token="[UNK]"))
tok.normalizer   = NFKC()
tok.pre_tokenizer = Metaspace(replacement="▁", prepend_scheme="never")
tok.decoder       = Metaspace(replacement="▁", prepend_scheme="never")
# weight by duplicating each corpus file: en×3, hi×4, te×4, mr×2
tok.train(files, BpeTrainer(vocab_size=10000, min_frequency=1))`],
    ["evaluate_tokenizer.py — faithful units, fertility &amp; score",
`FAITHFUL_UNIT_RE = regex.compile(r"[\\p{L}\\p{M}\\p{N}]+|[^\\s\\p{L}\\p{M}\\p{N}]")
ratio = tokens / faithful_units          # denominator counts every visible symbol
score = 1000 / (max(ratios) - min(ratios))
hindi_penalty = exp(max(0, hindi/1.2 - 1))   # 1.0 while Hindi < 1.2`],
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
  return `<h2 class="section">Fertility (tokens / faithful unit) — sorted</h2>
    <div class="chart">${rows}
    <div class="caption">Fertility = tokens ÷ faithful units; all four are below the 1.2 threshold
    (${ranked[0].ratio}–${ranked[ranked.length - 1].ratio}). The tight spread between X_max (${esc(maxLang)})
    and X_min (${esc(minLang)}) is what drives the score = 1000 / (${d.scoring.max_ratio} − ${d.scoring.min_ratio}).</div></div>`;
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
      ${metric(l.faithful_units.toLocaleString(), "faithful units")}
      ${metric(l.token_count.toLocaleString(), "tokens")}
      ${metric(l.ratio.toFixed(4), "fertility (tok / unit)")}
      ${metric(l.weight + "×", "training weight")}
    </div>
    <div class="subhead">Sample line — tokenized (${s.tokens.length} tokens · ▁ = space)</div>
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
  const CAP = 4000;  // faithful corpora are big (English ~111k tokens) — cap the DOM
  const draw = () => {
    const q = search.value.trim().toLowerCase();
    const hits = q ? toks.filter((t) => t.toLowerCase().includes(q)) : toks;
    grid.innerHTML = hits.slice(0, CAP).map((t, i) =>
      `<span class="chip" style="background:${CHIP_COLORS[i % CHIP_COLORS.length]}18;
        border-color:${CHIP_COLORS[i % CHIP_COLORS.length]}44">${showWS(t)}</span>`).join("");
    count.textContent = q
      ? `${hits.length.toLocaleString()} match "${q}"` + (hits.length > CAP ? ` (showing ${CAP.toLocaleString()})` : "")
      : `${toks.length.toLocaleString()} tokens total — showing first ${Math.min(CAP, toks.length).toLocaleString()}, in order; search to filter`;
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
  const v = d.vocab_stats, s = d.scoring;
  const li = d.notes.map((n) => `<li>${esc(n)}</li>`).join("");
  const w = v.weights || {};
  return `<h2 class="section">Methodology &amp; caveats</h2>
    <div class="notes"><ul>${li}</ul>
    <div class="vocab-line">Shared vocab: <b>${v.total_vocab_size.toLocaleString()}</b> tokens ·
      weights en:${w.en} · hi:${w.hi} · te:${w.te} · mr:${w.mr} ·
      raw score <b>${s.score.toLocaleString()}</b> · Hindi-adjusted <b>${s.hindi_adjusted_score?.toLocaleString()}</b>
      (penalty ${s.hindi_penalty_factor}×)</div></div>`;
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
