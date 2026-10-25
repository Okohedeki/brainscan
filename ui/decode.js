// Train a probe on recorded runs (Tune): a linear probe on the residual stream just before each
// agent step, trained to forecast how the run ends, with the controls that say whether it reads more
// than the transcript (brainscan/recorder/decoding.py). Training runs as a job on the CPU; the panel
// polls it, then shows the probe beside its shuffled-label nulls and baselines, by layer and by steps
// before the end, and saves it to the direction bank so Probe, Steer and Monitor can use it.

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
async function api(path, opts) {
  const res = await fetch(path, opts);
  const text = await res.text();
  let body = null;
  try { body = text ? JSON.parse(text) : null; } catch { /* not JSON */ }
  if (!res.ok) throw new Error(body?.detail || text || `${res.status}`);
  return body;
}
const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body ?? {}) });
const n2 = (x) => (x == null ? "–" : x.toFixed(2));  // null: undefined (e.g. no task with both outcomes)
const ci = (x) => (x ? `${n2(x.auc)} <span class="muted">[${n2(x.lo)}, ${n2(x.hi)}]</span>` : "–");

const P = { summary: null, job: null, timer: 0 };

function message(text, tone = "") {
  const el = $("#dc-msg");
  if (el) { el.className = `wb-msg ${tone}`; el.textContent = text || ""; }
}

async function refresh() {
  const study = $("#dc-study")?.value || "";
  try { P.summary = await api(`/brainscan/decoding${study ? `?study=${encodeURIComponent(study)}` : ""}`); }
  catch { return; }
  const sel = $("#dc-study");
  const keep = sel.value;
  sel.innerHTML = `<option value="">every agent run</option>` + P.summary.studies.map((s) => `<option value="${esc(s)}">runs tagged ${esc(s)}</option>`).join("");
  sel.value = P.summary.studies.includes(keep) ? keep : "";
  const tg = $("#dc-target");
  if (!tg.options.length) tg.innerHTML = Object.entries(P.summary.targets).map(([k, v]) => `<option value="${k}" title="${esc(v)}">${esc(v)}</option>`).join("");
  $("#dc-data").innerHTML = P.summary.runs
    ? `${P.summary.runs} agent runs with an outcome: ${P.summary.tasks.map((t) => `<b>${esc(t.task)}</b> ${t.runs} (${t.fails} failed)`).join(" · ")}`
    : "No agent runs of this model with an outcome yet. Run the mini agent (or a study) through the endpoint first.";
  if (!P.job && P.summary.jobs.length) { P.job = P.summary.jobs[P.summary.jobs.length - 1]; render(); if (P.job.status === "running") poll(); }
}

async function train() {
  const body = { target: $("#dc-target").value, study: $("#dc-study").value || null, perms: Number($("#dc-perms").value) || 0 };
  try {
    P.job = await post("/brainscan/decoding/train", body);
    message("");
    render();
    poll();
  } catch (err) { message(err.message, "error"); }
}

function poll() {
  clearTimeout(P.timer);
  P.timer = setTimeout(async () => {
    if (!P.job) return;
    try { P.job = await api(`/brainscan/decoding/jobs/${P.job.id}`); } catch (err) { message(err.message, "error"); return; }
    render();
    if (P.job.status === "running") poll();
  }, 1000);
}

async function save() {
  const name = $("#dc-name")?.value.trim();
  if (!name) { message("Give the probe a name.", "error"); return; }
  try {
    await post(`/brainscan/decoding/jobs/${P.job.id}/save`, { name });
    message(`Saved “${name}” to the direction bank. Watch reads it on every new token, and Monitor can flag runs with it.`, "ok");
    document.dispatchEvent(new CustomEvent("brainscan:workbench"));
  } catch (err) { message(err.message, "error"); }
}

// ---------------------------------------------------------------- drawing
function render() {
  const box = $("#dc-result");
  const j = P.job;
  $("#dc-train").disabled = j?.status === "running";
  if (!j) { box.innerHTML = ""; return; }
  if (j.status === "running") {
    box.innerHTML = `<div class="dc-progress"><div style="width:${Math.round(j.progress * 100)}%"></div></div>
      <p class="muted small">${esc(j.step)} · ${Math.round(j.progress * 100)}% · runs on the CPU; the model keeps working</p>`;
    return;
  }
  if (j.status === "error") { box.innerHTML = `<p class="warn-text small">${esc(j.error)}</p>`; return; }
  const r = j.report, p = r.probe, b = r.baselines, c = r.controls, w = c.shuffled_within_task || {}, a = c.shuffled || {};
  const tile = (label, value, sub) => `<div class="tile"><div class="label">${label}</div><div class="value">${value}</div><div class="sub">${sub}</div></div>`;
  box.innerHTML = `
    <p class="small">${esc(r.describes)}: ${r.runs} runs (${r.positive_runs} positive), ${r.steps} steps. AUC out of fold, 95% intervals over runs.</p>
    <div class="tiles">
      ${tile("Probe", ci(p), `within tasks ${ci(p.within_task)} · ${p.within_task.tasks} task${p.within_task.tasks === 1 ? "" : "s"} with both outcomes`)}
      ${tile("Transcript only", ci(b.transcript), `within tasks ${ci(b.transcript.within_task)}`)}
      ${tile("Position only", ci(b.position), "step number and tokens so far")}
      ${tile("Shuffled labels", a.n ? `${n2(a.mean)} · p ${a.p}` : "–", w.n ? `within tasks: ${n2(w.mean)} · p ${w.p}` : "not run")}
    </div>
    <ul class="dc-notes">${r.notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul>
    <h3>By layer</h3>${layerChart(r)}
    <h3>By steps before the end of the run</h3>${leadTable(r)}
    <div class="row dc-save">
      <label>Name<input id="dc-name" type="text" maxlength="64" placeholder="e.g. ${esc(r.target)}_probe" value="${esc(r.target)}_probe"></label>
      <button type="button" id="dc-save">Save as direction</button>
      <span class="muted small">Layer ${Object.entries(p.layers_chosen).sort((x, y) => y[1] - x[1])[0]?.[0] ?? "?"}, refit on every run; points toward ${esc(r.target.replace("_", " "))}.</span>
    </div>`;
  $("#dc-save").onclick = save;
}

function layerChart(r) {
  const L = r.layers, W = 640, H = 140, pad = { l: 34, r: 8, t: 8, b: 20 };
  const x = (i) => pad.l + (i / Math.max(1, L.length - 1)) * (W - pad.l - pad.r);
  const y = (v) => pad.t + (1 - (Math.min(1, Math.max(0.3, v)) - 0.3) / 0.7) * (H - pad.t - pad.b);
  const ok = L.filter((d) => d.lo != null && d.hi != null);
  const band = ok.length ? `M${ok.map((d) => `${x(d.layer)},${y(d.hi)}`).join("L")}L${[...ok].reverse().map((d) => `${x(d.layer)},${y(d.lo)}`).join("L")}Z` : "";
  const line = `M${L.map((d) => `${x(d.layer)},${y(d.auc != null ? d.auc : 0.5)}`).join("L")}`;
  const nul = r.controls.shuffled_within_task?.n ? r.controls.shuffled_within_task.mean : null;
  const tr = r.baselines.transcript.auc;
  const ticks = [0.5, 0.75, 1].map((v) => `<text x="${pad.l - 6}" y="${y(v) + 3}" text-anchor="end">${v}</text><line x1="${pad.l}" x2="${W - pad.r}" y1="${y(v)}" y2="${y(v)}" class="grid"/>`).join("");
  const xt = L.filter((d) => d.layer % 4 === 0).map((d) => `<text x="${x(d.layer)}" y="${H - 5}" text-anchor="middle">L${d.layer}</text>`).join("");
  return `<svg class="dc-chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Probe AUC by layer">
    ${ticks}${xt}
    <path d="${band}" class="band"/><path d="${line}" class="line"/>
    ${tr != null ? `<line x1="${pad.l}" x2="${W - pad.r}" y1="${y(tr)}" y2="${y(tr)}" class="tr"/><text x="${W - pad.r}" y="${y(tr) - 4}" text-anchor="end" class="tr-l">transcript only</text>` : ""}
    ${nul != null ? `<line x1="${pad.l}" x2="${W - pad.r}" y1="${y(nul)}" y2="${y(nul)}" class="nul"/><text x="${pad.l + 4}" y="${y(nul) - 4}" class="nul-l">shuffled within tasks</text>` : ""}
  </svg>`;
}

function leadTable(r) {
  if (!r.lead.length) return `<p class="muted small">Too few steps at any single distance from the end to score.</p>`;
  const lt = r.lead_time;
  return `<table class="mon-table dc-lead"><thead><tr><th>Steps before end</th><th>Steps</th><th>Probe</th><th>Transcript</th><th>Probe − transcript</th></tr></thead>
    <tbody>${r.lead.map((d) => `<tr class="${d.probe.lo > 0.5 ? "above" : ""}"><td>${d.lead}</td><td>${d.steps} <span class="muted">(${d.positive}+)</span></td>
      <td>${ci(d.probe)}</td><td>${ci(d.transcript)}</td>
      <td>${d.diff.diff >= 0 ? "+" : ""}${n2(d.diff.diff)} <span class="muted">[${n2(d.diff.lo)}, ${n2(d.diff.hi)}]</span></td></tr>`).join("")}</tbody></table>
    <p class="muted small">Above chance up to ${lt.above_chance ?? "–"} steps before the end; ahead of the transcript up to ${lt.above_transcript ?? "–"}.</p>`;
}

function boot() {
  if (!$("#dc-train")) return;
  $("#dc-train").onclick = train;
  $("#dc-study").onchange = refresh;
  document.addEventListener("brainscan:view", (ev) => { if (ev.detail === "workbench") refresh(); });
  document.addEventListener("brainscan:model", () => { P.job = null; render(); refresh(); });
  refresh();
}
boot();
