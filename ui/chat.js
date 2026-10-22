// Ask the model from the UI: a composer in the Watch and Trace tabs that sends a chat request to the
// recorder's own /v1 endpoint. Every request is recorded like any other run, so its tokens stream into
// the transcript, the timeline and the unit graph live. A conversation started here continues in the
// same run until "New chat"; a run recorded by an agent harness is never appended to.

const CHAT_PREFIX = "chat-";
const K = { busy: false, fresh: false, roots: [], settings: { think: "default", max: 512, temperature: 0 } };

try { Object.assign(K.settings, JSON.parse(localStorage.getItem("brainscan-chat") || "{}")); } catch { /* storage unavailable */ }
const save = () => { try { localStorage.setItem("brainscan-chat", JSON.stringify(K.settings)); } catch { /* storage unavailable */ } };

async function api(path, opts) {
  const res = await fetch(path, opts);
  const text = await res.text();
  let body = null;
  try { body = text ? JSON.parse(text) : null; } catch { /* not JSON */ }
  if (!res.ok) { const err = new Error(body?.detail || body?.error?.message || text || `${res.status}`); err.status = res.status; throw err; }
  return body;
}

const currentRun = () => document.querySelector("#run-select")?.value || null;
const isChatRun = (id) => !!id && id.startsWith(CHAT_PREFIX);
const newRunId = () => {
  const d = new Date(), p = (n) => String(n).padStart(2, "0");
  return `${CHAT_PREFIX}${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
};

/** The conversation so far, from the run's recorded messages (user, system and assistant text). */
async function history(runId) {
  const run = await api(`/brainscan/runs/${encodeURIComponent(runId)}`);
  return [...(run.messages || [])]
    .sort((a, b) => a.t - b.t || a.i - b.i)
    .filter((m) => ["system", "user", "assistant"].includes(m.role) && (m.content || "").length)
    .map((m) => ({ role: m.role, content: m.content }));
}

function setStatus(text, tone = "") {
  for (const root of K.roots) {
    const el = root.querySelector(".composer-status");
    el.textContent = text; el.className = `composer-status ${tone}`;
  }
}

function sync() {
  const where = K.fresh || !isChatRun(currentRun()) ? "starts a new chat" : "continues this chat";
  for (const root of K.roots) {
    root.querySelector(".composer-send").disabled = K.busy;
    root.querySelector("textarea").placeholder = K.busy ? "The model is answering…" : `Ask the model (${where}) · Enter to send, Shift+Enter for a new line`;
    root.querySelector(".composer-think").value = K.settings.think;
    root.querySelector(".composer-max").value = K.settings.max;
    root.querySelector(".composer-temp").value = K.settings.temperature;
  }
}

async function send(root) {
  const box = root.querySelector("textarea");
  const text = box.value.trim();
  if (!text || K.busy) return;
  K.busy = true; sync();
  const continuing = !K.fresh && isChatRun(currentRun());
  const runId = continuing ? currentRun() : newRunId();
  setStatus(continuing ? "Sending…" : "Starting a new chat…");
  try {
    const messages = [...(continuing ? await history(runId) : []), { role: "user", content: text }];
    // Open the run in every view as soon as it appears, and follow its tokens live.
    document.dispatchEvent(new CustomEvent("brainscan:expect-run", { detail: runId }));
    box.value = "";
    for (const r of K.roots) if (r !== root) r.querySelector("textarea").value = "";
    K.fresh = false;
    setStatus("Generating: the tokens stream into the transcript and the graph…");
    const kwargs = K.settings.think === "on" ? { enable_thinking: true } : K.settings.think === "off" ? { enable_thinking: false } : undefined;
    const res = await api("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Brainscan-Run": runId, "X-Brainscan-Label": "chat" },
      body: JSON.stringify({ messages, max_tokens: Number(K.settings.max) || 512, temperature: Number(K.settings.temperature) || 0,
        ...(kwargs ? { chat_template_kwargs: kwargs } : {}) }),
    });
    const u = res.usage || {};
    setStatus(`Done: ${u.completion_tokens ?? "?"} tokens (${res.choices?.[0]?.finish_reason || "stop"}). Click any word to see which units produced it.`, "ok");
  } catch (err) {
    if (!box.value) box.value = text;  // keep what was typed
    setStatus(err.status === 503 ? "The model is loading or switching; try again in a moment." : err.message, "error");
  } finally {
    K.busy = false; sync();
  }
}

/** Mount a composer into `root` (an element with the data-composer attribute). */
function mount(root) {
  root.innerHTML = `
    <textarea rows="2" spellcheck="true" aria-label="Message to the model"></textarea>
    <div class="composer-row">
      <label title="Reasoning models: think before answering">Think
        <select class="composer-think"><option value="default">model default</option><option value="on">on</option><option value="off">off</option></select></label>
      <label title="Most tokens to generate">Max<input class="composer-max" type="number" min="1" max="4096" step="1"></label>
      <label title="0 is greedy (the same answer every time)">Temp<input class="composer-temp" type="number" min="0" max="2" step="0.1"></label>
      <button type="button" class="btn sm composer-new" title="Start a new conversation on the next message">New chat</button>
      <button type="button" class="primary composer-send">Send</button>
    </div>
    <div class="composer-status" role="status"></div>`;
  const box = root.querySelector("textarea");
  box.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && !ev.shiftKey && !ev.isComposing) { ev.preventDefault(); send(root); }
    ev.stopPropagation();  // typing here never triggers the graph's shortcuts
  });
  root.querySelector(".composer-send").onclick = () => send(root);
  root.querySelector(".composer-new").onclick = () => { K.fresh = true; setStatus("The next message starts a new chat."); sync(); box.focus(); };
  root.querySelector(".composer-think").onchange = (ev) => { K.settings.think = ev.target.value; save(); sync(); };
  root.querySelector(".composer-max").onchange = (ev) => { K.settings.max = Number(ev.target.value) || 512; save(); sync(); };
  root.querySelector(".composer-temp").onchange = (ev) => { K.settings.temperature = Math.max(0, Number(ev.target.value) || 0); save(); sync(); };
  K.roots.push(root);
}

function boot() {
  document.querySelectorAll("[data-composer]").forEach(mount);
  document.querySelector("#run-select")?.addEventListener("change", sync);
  document.addEventListener("brainscan:run", sync);
  sync();
}
boot();
