// The Neurons view: a secondary screen for single MLP neurons ("L14 N8949"). A card says what a
// neuron does (logit lens, steer lens at ±dose, where it fires, what feeds it and what it feeds);
// a steer set doses one or more neurons with the recipe and answers a prompt at several dose levels
// to verify, or steers the chat itself (brainscan/recorder/neurons.py). Any neuron named in the app
// opens here: a [data-open-neuron] link anywhere, or the brainscan:neuron event.
import { defineBar, noteBar, refreshBar } from "/ui/actionbar.js";

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const vis = (s) => (s ?? "").replace(/\n/g, "⏎").replace(/\t/g, "⇥").replace(/^ /, "·") || "∅";
const n2 = (x, d = 2) => (x == null ? "–" : Number(x).toFixed(d));
const sign = (x, d = 2) => `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(d)}`;
async function api(path, opts) {
  const res = await fetch(path, opts);
  const text = await res.text();
  let body = null;
  try { body = text ? JSON.parse(text) : null; } catch { /* not JSON */ }
  if (!res.ok) throw new Error(body?.detail || text || `${res.status}`);
  return body;
}
const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body ?? {}) });

const N = { name: null, dose: 1, card: null, loading: null, history: [], set: [], sweep: null, busy: false, steering: [],
  index: null, results: null, searching: null, timer: 0 };
const SET_KEY = "brainscan-neuron-set";

function message(text, tone = "") {
  const el = $("#nv-msg");
  if (el) { el.className = `wb-msg ${tone}`; el.textContent = text || ""; }
}

// ---------------------------------------------------------------- the card
async function open(name, { push = true } = {}) {
  name = String(name || "").trim();
  if (!/^L\d+\s*N\d+$/i.test(name)) { message("Name a neuron like L14 N8949.", "error"); return; }
  name = name.toUpperCase().replace(/\s*N/, " N");
  if (push && N.name && N.name !== name) N.history.push(N.name);
  N.name = name;
  N.dose = Number($("#nv-dose").value) || 1;
  $("#nv-q").value = name;
  N.loading = name;
  N.card = null;
  render();
  try {
    const card = await api(`/brainscan/neurons/${encodeURIComponent(name)}?dose=${N.dose}`);
    if (N.loading !== name) return;
    N.card = card;
    message("");
  } catch (err) {
    if (N.loading === name) message(err.message, "error");
  }
  N.loading = null;
  render();
}

function chips(list, cls) {
  return list.map((t) => `<span class="nv-chip ${cls}" title="${esc(t.text)}: ${t.v}">${esc(vis(t.text))}</span>`).join("");
}
function links(list, key, fmt) {
  return list.length ? list.map((x) => `<button type="button" class="nv-link" data-open-neuron="${x.neuron}" title="Open ${x.neuron}">${x.neuron} <span>${fmt(x[key])}</span></button>`).join("")
    : `<span class="muted small">none</span>`;
}

function render() {
  refreshBar("neurons");
  renderSet();
  const body = $("#nv-body");
  if (!body) return;
  if (N.loading) {
    body.innerHTML = `<p class="muted">Reading ${esc(N.loading)} on the calibration texts: its lenses, where it fires, and what it feeds and is fed by…</p>`;
    return;
  }
  const c = N.card;
  if (!c) {
    body.innerHTML = `<p class="muted">Open a neuron by name, or from a cluster's neurons in Trace. A neuron is one channel of a layer's MLP output: it writes its activation × its output weights into the residual stream.</p>`;
    return;
  }
  const sl = c.steer_lens, a = c.activity, fb = c.fed_by, fd = c.feeds;
  body.innerHTML = `
    <div class="nv-head">
      <h3>${esc(c.neuron)}</h3>
      ${c.unit ? `<button type="button" class="link" data-trace-unit="${c.unit.g}">in cluster ${esc(c.unit.name)} (Trace)</button>` : ""}
      <span class="muted small">‖W_down‖ ${n2(c.write_norm, 3)} · median residual ${n2(c.median_norm, 1)} → dose ${c.dose} writes ${n2(c.dose * c.median_norm, 1)}</span>
    </div>
    <section class="nv-sec"><h4>Logit lens <span class="dim">what its write pushes, read straight out (weights only)</span></h4>
      <div class="nv-row"><span class="nv-lbl">up</span>${chips(c.logit_lens.up, "up")}</div>
      <div class="nv-row"><span class="nv-lbl">down</span>${chips(c.logit_lens.down, "down")}</div></section>
    <section class="nv-sec"><h4>Steer lens <span class="dim">what steering it does to the predictions, averaged over the calibration texts</span></h4>
      ${[sl.plus, sl.minus].map((s) => `<div class="nv-steer"><b>dose ${sign(s.dose, 2)}</b><span class="muted small">KL ${n2(s.kl, 3)}</span></div>
        <div class="nv-row"><span class="nv-lbl">more</span>${chips(s.up, "up")}</div>
        <div class="nv-row"><span class="nv-lbl">less</span>${chips(s.down, "down")}</div>`).join("")}</section>
    ${c.corpus ? `<section class="nv-sec"><h4>Where it fires <span class="dim">its top activations in the neuron index (${c.corpus.tokens.toLocaleString()} tokens): mean ${n2(c.corpus.mean)}, spread ${n2(c.corpus.std)}</span></h4>
      <ul class="nv-ctx">${c.corpus.contexts.map(ctxRow).join("")}</ul></section>` : ""}
    <section class="nv-sec"><h4>${c.corpus ? "On the calibration texts" : "Where it fires"} <span class="dim">mean ${n2(a.mean)}, max ${n2(a.max)}, ${a.positions} tokens</span></h4>
      <ul class="nv-ctx">${a.contexts.map(ctxRow).join("")}</ul>
      ${c.corpus ? "" : `<p class="hint">A small, fixed set of texts. Build the neuron index (Find neurons, above) for its top activations across a corpus.</p>`}</section>
    <section class="nv-sec two"><div><h4>Feeds <span class="dim">later neurons it moves when steered (z)</span></h4>
        <div class="nv-sub">toward</div><div class="nv-links">${links(fd.toward, "z", (z) => sign(z, 1))}</div>
        <div class="nv-sub">away</div><div class="nv-links">${links(fd.away, "z", (z) => sign(z, 1))}</div></div>
      <div><h4>Fed by <span class="dim">earlier neurons driving it where it fires</span></h4>
        <div class="nv-sub">drive it up</div><div class="nv-links">${links(fb.drives_up, "c", (v) => sign(v, 3))}</div>
        <div class="nv-sub">drive it down</div><div class="nv-links">${links(fb.drives_down, "c", (v) => sign(v, 3))}</div></div></section>`;
}

const ctxRow = (x) => `<li><span class="v">${n2(x.act)}</span><span class="t">…${esc(x.before)}<mark>${esc(x.token)}</mark>${esc(x.after)}</span></li>`;
const snip = (c) => (c ? `<span class="t">…${esc(c.before.slice(-60))}<mark>${esc(c.token)}</mark>${esc(c.after)}</span>` : "");

// ---------------------------------------------------------------- the index and search
async function loadIndex() {
  try { N.index = await api("/brainscan/neurons/index"); } catch { N.index = null; }
  renderIndex();
  if (N.index?.status === "building") { clearTimeout(N.timer); N.timer = setTimeout(loadIndex, 1500); }
}

function renderIndex() {
  const el = $("#nv-index");
  if (!el) return;
  refreshBar("neurons");
  const x = N.index;
  if (!x) { el.textContent = ""; return; }
  const m = x.meta;
  const built = m ? `Index: ${m.tokens.toLocaleString()} tokens from ${m.sources.join(", ")}; built ${new Date(m.built * 1000).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })} in ${m.seconds}s.` : "";
  el.innerHTML = x.status === "building" ? `Building the neuron index: ${esc(x.step || "")} ${Math.round((x.progress || 0) * 100)}%. The model is busy until it's done.`
    : x.status === "error" ? `<span class="warn-text">The index build failed: ${esc(x.error)}</span> ${esc(built)}`
    : m ? `${esc(built)} <button type="button" class="link" data-build>Rebuild</button>`
    : `No neuron index for this model yet. <button type="button" class="link" data-build>Build it</button>: one read of the corpus (the starter texts, your recorded runs, or BRAINSCAN_CORPUS) keeps every neuron's top contexts and lens.`;
}

async function buildIndex() {
  try { N.index = await post("/brainscan/neurons/index", {}); } catch (err) { message(err.message, "error"); }
  loadIndex();
}

async function search(q) {
  q = String(q || "").trim();
  if (q.length < 2) { message("Search for at least two characters.", "error"); return; }
  N.searching = q; N.results = null; renderResults();
  try { N.results = await api(`/brainscan/neurons/search?q=${encodeURIComponent(q)}&limit=12`); message(""); }
  catch (err) { message(err.message, "error"); }
  N.searching = null; renderResults();
}

function renderResults() {
  const box = $("#nv-results");
  if (!box) return;
  refreshBar("neurons");
  if (N.searching) { box.innerHTML = `<p class="muted small">Searching for “${esc(N.searching)}”: reading it where it occurs…</p>`; return; }
  const r = N.results;
  if (!r) { box.innerHTML = ""; return; }
  const col = (title, sub, rows, fmt) => `<div class="nv-col"><h4>${title} <span class="dim">${sub}</span></h4>
    <ul>${rows.map(fmt).join("") || `<li class="muted small">none</li>`}</ul></div>`;
  const link = (x, extra) => `<button type="button" class="nv-link" data-open-neuron="${x.neuron}">${x.neuron} <span>${extra}</span></button>`;
  box.innerHTML = `<p class="small">“${esc(r.query)}” ${r.in_corpus ? `occurs ${r.matches} time${r.matches === 1 ? "" : "s"} in the index's ${r.tokens.toLocaleString()} tokens` : "isn't in the corpus: neurons that fire on it are read on the query itself"}.</p>
    <div class="nv-cols">
      ${col("Fire on it", "z, where it occurs", r.fires_on, (x) => `<li>${link(x, `z ${n2(x.z, 1)}`)}</li>`)}
      ${col("Push it", "a logit-lens top token", r.pushes, (x) => `<li>${link(x, `#${x.rank + 1}`)}<span class="nv-toks">${x.tokens.map((t) => esc(vis(t))).join(" ")}</span></li>`)}
      ${col("Have it in their top contexts", "hits", r.contexts, (x) => `<li>${link(x, `${x.hits}× · ${n2(x.act)}`)}${snip(x.context)}</li>`)}
    </div>`;
}

// ---------------------------------------------------------------- steer and verify
function saveSet() { try { localStorage.setItem(SET_KEY, JSON.stringify(N.set)); } catch { /* storage unavailable */ } }

function renderSet() {
  const ul = $("#nv-set");
  if (!ul) return;
  ul.innerHTML = N.set.map((s, i) => `<li data-i="${i}"><button type="button" class="link mono" data-open-neuron="${s.neuron}">${s.neuron}</button>
      <label>dose <input type="number" step="0.25" min="-16" max="16" value="${s.dose}" data-dose></label>
      <button type="button" data-remove aria-label="Remove ${s.neuron} from the steer set">✕</button></li>`).join("")
    || `<li class="muted" style="border:none;background:none;padding:0">No neurons yet. Open one and press Add to steer set, or type several, like L14 N8949 +1 and L13 N7956 −1.</li>`;
  const live = N.steering.length;
  $("#nv-live").textContent = live ? `Steering the chat with ${live} neuron${live === 1 ? "" : "s"}: ${N.steering.map((iv) => `${iv.direction} ${sign(iv.strength, 2)}`).join(", ")}` : "";
}

function addToSet(neuron = N.name, dose = N.dose) {
  if (!neuron) return;
  const at = N.set.findIndex((s) => s.neuron === neuron);
  if (at >= 0) N.set[at].dose = dose; else N.set.push({ neuron, dose });
  saveSet(); render();
}

/** "L14 N8949 +1, L13 N7956 -1" → the steer set. */
function addTyped(text) {
  const found = [...String(text).matchAll(/L(\d+)\s*N(\d+)\s*([+−-]?\s*\d*\.?\d+)?/gi)];
  for (const m of found) addToSet(`L${m[1]} N${m[2]}`, m[3] ? Number(m[3].replace("−", "-").replace(/\s/g, "")) : 1);
  return found.length;
}

async function runSweep() {
  if (!N.set.length) { message("Add a neuron to the steer set first.", "error"); return; }
  const prompt = $("#nv-prompt").value.trim();
  if (!prompt) { message("Give a prompt to answer.", "error"); return; }
  const factors = $("#nv-factors").value.split(/[,\s]+/).map(Number).filter((x) => Number.isFinite(x)).slice(0, 8);
  N.busy = true; N.sweep = null; renderSweep(); refreshBar("neurons");
  try {
    N.sweep = await post("/brainscan/neurons/sweep", { prompt, steers: N.set, factors: factors.length ? factors : [0, 1], max_tokens: Number($("#nv-max").value) || 60 });
    message("");
  } catch (err) { message(err.message, "error"); }
  N.busy = false; renderSweep(); refreshBar("neurons");
}

function renderSweep() {
  const box = $("#nv-sweep");
  if (!box) return;
  if (N.busy) { box.innerHTML = `<p class="muted small">Answering at each dose level (greedy)…</p>`; return; }
  if (!N.sweep) { box.innerHTML = ""; return; }
  box.innerHTML = N.sweep.map((r) => `<div class="nv-out${r.factor === 0 ? " base" : ""}"><div class="nv-out-head"><b>× ${r.factor}</b>
      <span class="muted small">${r.factor === 0 ? "unsteered" : r.doses.map((d, i) => `${N.set[i]?.neuron ?? ""} ${sign(d, 2)}`).join(", ")}</span></div>
      <p>${esc(r.text)}</p></div>`).join("");
}

async function refreshSteering() {
  try { N.steering = (await api("/brainscan/neurons")).steering || []; } catch { N.steering = []; }
  renderSet(); refreshBar("neurons");
}

async function steerChat() {
  if (!N.set.length) return;
  try {
    for (const s of N.set) await post("/brainscan/neurons/steer", { neuron: s.neuron, dose: s.dose, phase: "all" });
    await refreshSteering();
    noteBar("neurons", "Every new request is steered with the set, prompt and reply alike, and marked ▲steered");
  } catch (err) { message(err.message, "error"); }
}

async function unsteer() {
  for (const iv of N.steering) await api(`/brainscan/interventions/${iv.id}`, { method: "DELETE" }).catch(() => {});
  await refreshSteering();
  noteBar("neurons", "Neuron steering removed");
}

// ---------------------------------------------------------------- the bar
function setupBar() {
  const scroll = (sel) => $(sel)?.scrollIntoView({ behavior: "smooth", block: "start" });
  defineBar("neurons", {
    groups: ["Find", "Steer", "Verify"],
    context: () => (N.card ? `<b>${esc(N.card.neuron)}</b> · dose ${N.card.dose}` : N.loading ? `Reading <b>${esc(N.loading)}</b>…` : "No neuron open")
      + (N.set.length ? ` · steer set: ${N.set.length}` : "") + (N.steering.length ? ` · <b>steering the chat</b>` : ""),
    state: () => ({ neuron: N.card?.neuron || null, steer_set: N.set.length, neuron_steering: N.steering.length }),
    actions: [
      { id: "search", group: "Find", label: "Search", hint: "Find neurons by text: ones that fire on it, push it, or have it in their top contexts",
        need: () => (N.index?.meta ? null : "the neuron index: build it first"), run: () => { scroll("#h-nsearch"); $("#nv-s").focus(); } },
      { id: "build", group: "Find", label: () => (N.index?.meta ? "Rebuild index" : "Build index"), hint: "Read the corpus once and keep every neuron's top contexts and lens",
        need: () => (N.index?.status === "building" ? "the build running now to finish" : null), run: buildIndex },
      { id: "go", group: "Find", label: "Go to neuron", hint: "Open a neuron by name, like L14 N8949", run: () => { $("#nv-q").focus(); $("#nv-q").select(); } },
      { id: "back", group: "Find", label: "Back", hint: "The neuron opened before this one",
        need: () => (N.history.length ? null : "a neuron opened before this one"), run: () => open(N.history.pop(), { push: false }) },
      { id: "add", group: "Steer", label: "Add to steer set", hint: "Steer this neuron, at the dose above, with the others in the set",
        need: () => (N.card ? null : "a neuron open"), run: () => addToSet() },
      { id: "chat", group: "Steer", label: "Steer the chat", hint: "Steer every new request with the set, prompt and reply alike",
        need: () => (N.set.length ? null : "a neuron in the steer set"), run: steerChat },
      { id: "unsteer", group: "Steer", label: "Remove neuron steering", hint: "Stop steering the chat with neurons",
        need: () => (N.steering.length ? null : "neurons steering the chat"), run: unsteer },
      { id: "sweep", group: "Verify", label: "Run sweep", hint: "Answer the prompt at each dose level, side by side",
        need: () => (N.busy ? "the sweep running now to finish" : N.set.length ? null : "a neuron in the steer set"), run: () => { scroll("#h-nsteer"); runSweep(); } },
    ],
  });
}

function boot() {
  if (!$("#view-neurons")) return;
  try { N.set = JSON.parse(localStorage.getItem(SET_KEY) || "[]"); } catch { N.set = []; }
  setupBar();
  $("#nv-go").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const q = $("#nv-q").value;
    if (/L\d+\s*N\d+.*L\d+\s*N\d+/i.test(q) || /[+−-]\s*\d/.test(q)) { if (addTyped(q)) { message("Added to the steer set.", "ok"); return; } }
    open(q);
  });
  $("#nv-dose").addEventListener("change", () => { if (N.name) open(N.name, { push: false }); });
  $("#nv-sform").addEventListener("submit", (ev) => { ev.preventDefault(); search($("#nv-s").value); });
  $("#nv-index").addEventListener("click", (ev) => { if (ev.target.closest("[data-build]")) buildIndex(); });
  $("#nv-run").addEventListener("click", runSweep);
  $("#nv-set").addEventListener("change", (ev) => {
    const li = ev.target.closest("li[data-i]");
    if (li && ev.target.matches("[data-dose]")) { N.set[Number(li.dataset.i)].dose = Number(ev.target.value) || 0; saveSet(); }
  });
  $("#nv-set").addEventListener("click", (ev) => {
    const li = ev.target.closest("li[data-i]");
    if (li && ev.target.matches("[data-remove]")) { N.set.splice(Number(li.dataset.i), 1); saveSet(); render(); }
  });
  $("#nv-body").addEventListener("click", (ev) => {
    const b = ev.target.closest("[data-trace-unit]");
    if (b) { document.dispatchEvent(new CustomEvent("brainscan:show", { detail: "weights" })); document.dispatchEvent(new CustomEvent("brainscan:trace", { detail: { unit: Number(b.dataset.traceUnit) } })); }
  });
  // A neuron named anywhere in the app opens here (capture phase: before a row's own click).
  document.addEventListener("click", (ev) => {
    const b = ev.target.closest("[data-open-neuron]");
    if (!b) return;
    ev.preventDefault(); ev.stopPropagation();
    document.dispatchEvent(new CustomEvent("brainscan:neuron", { detail: b.dataset.openNeuron }));
  }, true);
  document.addEventListener("brainscan:neuron", (ev) => {
    if ($("#view-neurons").hidden) document.dispatchEvent(new CustomEvent("brainscan:show", { detail: "neurons" }));
    open(ev.detail);
  });
  document.addEventListener("brainscan:view", (ev) => { if (ev.detail === "neurons") { refreshSteering(); loadIndex(); } });
  document.addEventListener("brainscan:model", () => { N.card = null; N.history = []; N.sweep = null; N.results = null; render(); renderSweep(); renderResults(); loadIndex(); });
  render();
  refreshSteering();
  loadIndex();
}
boot();
