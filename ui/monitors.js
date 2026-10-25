// Monitors (the Monitor action): flag a run when a probe fires. A monitor is a rule on one direction
// of the bank: its z-score past a threshold for some tokens in a row. The recorder checks every
// enabled monitor as tokens are recorded and adds a "monitor" event where the rule is met
// (brainscan/recorder/monitors.py). This panel in Watch adds and lists monitors, shows the open
// run's flags beside how often the random direction trips the same rule, and scans recent runs.
// Its tools join Watch's bar (extendBar).
import { extendBar, noteBar, refreshBar } from "/ui/actionbar.js";

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const visTok = (s) => (s ?? "").replace(/\n/g, "⏎").replace(/\t/g, "⇥");
const pct = (x) => (x == null ? "–" : `${Math.round(x * 100)}%`);
async function api(path, opts) {
  const res = await fetch(path, opts);
  const text = await res.text();
  let body = null;
  try { body = text ? JSON.parse(text) : null; } catch { /* not JSON */ }
  if (!res.ok) throw new Error(body?.detail || text || `${res.status}`);
  return body;
}
const send = (method, path, body) => api(path, { method, headers: { "Content-Type": "application/json" }, body: body ? JSON.stringify(body) : undefined });

const M = { list: [], dirs: [], flags: null, flagsFor: null, across: null, at: null, timer: 0 };
const runId = () => $("#run-select")?.value || null;
const enabled = () => M.list.filter((m) => m.enabled);
/** The open run's flags, every monitor's, in tape order. */
const flagList = () => (M.flags || []).filter((r) => r.monitor.enabled)
  .flatMap((r) => r.flags.map((f) => ({ ...f, name: r.monitor.name }))).sort((a, b) => a.t - b.t);
const scrollTo = (sel) => $(sel)?.scrollIntoView({ behavior: "smooth", block: "start" });

function message(text, tone = "") {
  const el = $("#mon-msg");
  if (el) { el.className = `wb-msg ${tone}`; el.textContent = text || ""; }
}

// ---------------------------------------------------------------- state
async function refresh() {
  try {
    const [list, status] = await Promise.all([api("/brainscan/monitors"), api("/brainscan/status")]);
    M.list = list;
    M.dirs = status.directions || [];
  } catch { /* the recorder is loading: the next refresh catches up */ }
  renderForm(); renderList();
  await loadFlags();
}

async function loadFlags() {
  const id = runId();
  M.flagsFor = id;
  if (!id || !M.list.length) { M.flags = null; renderFlags(); return; }
  try {
    const flags = await api(`/brainscan/runs/${encodeURIComponent(id)}/flags`);
    if (M.flagsFor === id) M.flags = flags;
  } catch { M.flags = null; }
  renderFlags();
}
const loadFlagsSoon = () => { clearTimeout(M.timer); M.timer = setTimeout(loadFlags, 400); };

// ---------------------------------------------------------------- drawing
function renderForm() {
  const sel = $("#mon-dir");
  if (!sel) return;
  const keep = sel.value;
  const dirs = M.dirs.filter((d) => d.kind !== "random_control");
  sel.innerHTML = dirs.map((d) => `<option value="${esc(d.name)}">${esc(d.name)} · L${d.layer}</option>`).join("")
    || `<option value="">no directions yet: find one in Tune</option>`;
  if (dirs.some((d) => d.name === keep)) sel.value = keep;
  $("#mon-form button[type=submit]").disabled = !dirs.length;
}

function renderList() {
  const ul = $("#mon-list");
  if (!ul) return;
  ul.innerHTML = M.list.map((m) => `<li data-id="${esc(m.id)}" class="${m.enabled ? "" : "off"}">
      <input type="checkbox" ${m.enabled ? "checked" : ""} data-toggle aria-label="Watch with ${esc(m.name)}">
      <span class="desc">${esc(m.rule)}</span>
      <button type="button" data-remove aria-label="Remove ${esc(m.name)}">✕</button></li>`).join("")
    || `<li class="muted" style="border:none;background:none;padding:0">No monitors yet. Every run is recorded either way; a monitor flags the tokens where a direction stays past a threshold.</li>`;
  refreshBar("scanner");
}

function renderFlags() {
  const box = $("#mon-run");
  if (!box) return;
  refreshBar("scanner");
  if (!runId()) { box.innerHTML = `<p class="muted small">No run open.</p>`; return; }
  if (!M.list.length) { box.innerHTML = `<p class="muted small">Add a monitor to see where it fires in this run and in runs recorded before it.</p>`; return; }
  if (!M.flags) { box.innerHTML = `<p class="muted small">Reading this run…</p>`; return; }
  box.innerHTML = M.flags.filter((r) => r.monitor.enabled).map(flagRowHtml).join("")
    || `<p class="muted small">Every monitor is off.</p>`;
}

function flagRowHtml(r) {
  const m = r.monitor, c = r.control;
  if (!r.tokens) return `<div class="mon-row"><b>${esc(m.name)}</b> <span class="muted small">no tokens of the kind it watches in this run</span></div>`;
  if (!r.recorded) {
    return `<div class="mon-row"><b>${esc(m.name)}</b> <span class="muted small">not recorded on this run: ${esc(m.direction)} was saved after it. Ask again to record it.</span></div>`;
  }
  const n = r.flags.length;
  const noise = c && c.recorded && n && c.flags >= n;
  const ctl = c && c.recorded ? `random direction, same rule: <b>${c.flags}</b> flag${c.flags === 1 ? "" : "s"} · past the threshold on ${pct(r.past)} of tokens (random ${pct(c.past)})`
    : `past the threshold on ${pct(r.past)} of tokens`;
  return `<div class="mon-row">
    <div class="mon-row-head"><b>${esc(m.name)}</b><span class="mon-count${n ? " hit" : ""}">${n ? `⚑ ${n} flag${n === 1 ? "" : "s"}` : "no flags"}</span></div>
    <p class="muted small">${ctl}</p>
    ${noise ? `<p class="warn-text small">The random direction trips this rule as often: read these flags as noise, or raise the threshold.</p>` : ""}
    ${n ? `<ul class="mon-flags">${r.flags.map((f) => `<li>
      <button type="button" class="link" data-t="${f.t}" title="Select the token where the rule was met">⚑ t${f.start}${f.end > f.start ? `–${f.end}` : ""}</button>
      <span class="muted">${f.tokens} token${f.tokens === 1 ? "" : "s"} · peak z ${f.peak_z.toFixed(2)} at t${f.peak_t}</span>
      <span class="mon-text">${esc(visTok(f.text || ""))}</span>
      <button type="button" class="link" data-trace="${f.peak_t}" title="Open the peak token in Trace">Trace →</button></li>`).join("")}</ul>` : ""}
  </div>`;
}

function renderAcross() {
  const box = $("#mon-across");
  if (!box) return;
  const a = M.across;
  if (!a) { box.innerHTML = ""; return; }
  if (!a.monitors.length) { box.innerHTML = `<p class="muted small">Every monitor is off.</p>`; return; }
  const when = (t) => (t ? new Date(t * 1000).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "");
  box.innerHTML = `<h3>Recent runs</h3>
    <table class="mon-table"><thead><tr><th>Run</th>${a.monitors.map((m) => `<th title="${esc(m.rule)}">${esc(m.name)}</th>`).join("")}</tr></thead>
    <tbody>${a.runs.map((r) => `<tr class="${r.flagged ? "flagged" : ""}">
      <td><button type="button" class="link" data-open-run="${esc(r.run)}">${esc(r.label || r.run)}</button> <span class="muted">${when(r.created)} · ${r.tokens} tok</span></td>
      ${a.monitors.map((m) => `<td>${r.flags[m.id] ? `<b>⚑ ${r.flags[m.id]}</b>` : "0"}${m.id in r.control ? ` <span class="muted" title="The random direction on the same rule">(random ${r.control[m.id]})</span>` : ""}</td>`).join("")}
    </tr>`).join("")}</tbody></table>
    <p class="muted small">${a.runs.filter((r) => r.flagged).length} of ${a.runs.length} runs flagged. Counts in brackets: the random direction on the same rule.</p>`;
}

// ---------------------------------------------------------------- doing
async function addMonitor(ev) {
  ev.preventDefault();
  const body = { direction: $("#mon-dir").value, sign: $("#mon-sign").value, threshold: Number($("#mon-z").value),
    min_tokens: Number($("#mon-n").value), phase: $("#mon-phase").value };
  try {
    const m = await send("POST", "/brainscan/monitors", body);
    message(`Watching: ${m.rule}. Runs recorded from now on are checked as they go.`, "ok");
    await refresh();
  } catch (err) { message(err.message, "error"); }
}

async function onListClick(ev) {
  const li = ev.target.closest("li[data-id]");
  if (!li) return;
  try {
    if (ev.target.matches("[data-remove]")) await send("DELETE", `/brainscan/monitors/${li.dataset.id}`);
    else if (ev.target.matches("[data-toggle]")) await send("PATCH", `/brainscan/monitors/${li.dataset.id}`, { enabled: ev.target.checked });
    else return;
    M.across = null; renderAcross();
    await refresh();
  } catch (err) { message(err.message, "error"); }
}

function selectToken(t) {
  M.at = t;
  document.dispatchEvent(new CustomEvent("brainscan:select", { detail: { t } }));
}
function traceToken(t) {
  document.dispatchEvent(new CustomEvent("brainscan:show", { detail: "weights" }));
  document.dispatchEvent(new CustomEvent("brainscan:trace", { detail: { word: t } }));
}
function nextFlag() {
  const fl = flagList();
  if (!fl.length) return;
  const f = fl.find((x) => M.at == null || x.t > M.at) || fl[0];
  selectToken(f.t);
  noteBar("scanner", `⚑ ${f.name} at t${f.t} (z ${f.z.toFixed(2)}), ${fl.indexOf(f) + 1} of ${fl.length}`);
}

async function scanRuns() {
  const btn = $("#mon-scan");
  if (btn) { btn.disabled = true; btn.textContent = "Scanning…"; }
  try { M.across = await api("/brainscan/flags?limit=20"); renderAcross(); scrollTo("#mon-across"); }
  catch (err) { message(err.message, "error"); }
  finally { if (btn) { btn.disabled = false; btn.textContent = "Scan recent runs"; } }
}

function onPanelClick(ev) {
  const b = ev.target.closest("[data-t], [data-trace], [data-open-run]");
  if (!b) return;
  if (b.dataset.t != null) selectToken(Number(b.dataset.t));
  else if (b.dataset.trace != null) traceToken(Number(b.dataset.trace));
  else {
    const sel = $("#run-select");
    sel.value = b.dataset.openRun;
    sel.dispatchEvent(new Event("change"));
    scrollTo("#h-flags");
  }
}

// ---------------------------------------------------------------- Watch's bar
function setupBar() {
  const noMonitor = () => (enabled().length ? null : "a monitor: add one first");
  extendBar("scanner", {
    groups: ["Monitor"], before: "Hand off",
    state: () => ({ monitors: enabled().length, flags: flagList().length }),
    actions: [
      { id: "monitor", group: "Monitor", label: "Add a monitor", hint: "A rule on a direction: its z-score past a threshold for some tokens in a row",
        run: () => { scrollTo("#h-monitors"); setTimeout(() => $("#mon-dir")?.focus(), 350); } },
      { id: "flags", group: "Monitor", label: "Flags in this run", hint: "Where each monitor fired in the open run, beside the random direction on the same rule",
        need: () => (runId() ? noMonitor() : "a run: ask the model or pick one"), run: () => scrollTo("#h-flags") },
      { id: "nextflag", group: "Monitor", label: "Next flag", hint: "Select the token where the next flag in this run was raised",
        need: () => noMonitor() || (flagList().length ? null : "a flag in this run"), run: nextFlag },
      { id: "scanruns", group: "Monitor", label: "Scan recent runs", hint: "Flag counts for the 20 most recent runs, recorded before or after the monitor",
        need: noMonitor, run: scanRuns },
    ],
  });
}

function boot() {
  if (!$("#mon-form")) return;
  setupBar();
  $("#mon-form").addEventListener("submit", addMonitor);
  $("#mon-list").addEventListener("click", onListClick);
  $("#mon-scan").addEventListener("click", scanRuns);
  for (const id of ["#mon-run", "#mon-across"]) $(id).addEventListener("click", onPanelClick);
  $("#run-select").addEventListener("change", () => { M.at = null; M.flags = null; renderFlags(); loadFlagsSoon(); });
  document.addEventListener("brainscan:run", () => { M.at = null; loadFlagsSoon(); });
  // A flag raised live (ui/app.js relays the recorder's "monitor" events).
  document.addEventListener("brainscan:flag", (ev) => {
    const { run, ev: e } = ev.detail || {};
    if (!e) return;
    noteBar("scanner", `⚑ ${e.data?.name || "monitor"} fired at t${e.t} (z ${Number(e.data?.z).toFixed(2)})${run === runId() ? "" : ` in ${run}`}`);
    if (run === runId()) loadFlagsSoon();
  });
  document.addEventListener("brainscan:done", (ev) => { if (ev.detail?.run === runId()) loadFlagsSoon(); });
  document.addEventListener("brainscan:view", (ev) => { if (ev.detail === "scanner") refresh(); });  // directions saved in Tune
  document.addEventListener("brainscan:model", () => { M.across = null; renderAcross(); refresh(); });
  refresh();
}
boot();
