// Demo scenes for recorder.mjs. Every function here is serialized and evaluated in the page, so it
// can't close over anything in this module: HELPER installs window.demo (a visible cursor, captions,
// title cards, real mouse input, app helpers), and each scene's setup and script use it.

export function HELPER() {
  if (window.demo) return true;
  const style = document.createElement("style");
  style.textContent = `
    #demo-cursor { position: fixed; left: 0; top: 0; width: 26px; height: 26px; margin: -13px 0 0 -13px; border-radius: 50%;
      background: rgba(255,255,255,0.22); border: 2.5px solid #fff; z-index: 2147483647; pointer-events: none;
      box-shadow: 0 0 0 4px rgba(139,124,255,0.55), 0 6px 18px rgba(0,0,0,0.55); transform: translate(960px, 540px); }
    #demo-cursor.press { animation: demo-press 0.45s ease-out; }
    @keyframes demo-press { 0% { box-shadow: 0 0 0 4px rgba(139,124,255,0.6), 0 6px 18px rgba(0,0,0,0.55); }
      45% { box-shadow: 0 0 0 18px rgba(139,124,255,0.25), 0 6px 18px rgba(0,0,0,0.55); }
      100% { box-shadow: 0 0 0 4px rgba(139,124,255,0.6), 0 6px 18px rgba(0,0,0,0.55); } }
    #demo-cap { position: fixed; left: 50%; bottom: 150px; transform: translate(-50%, 12px); max-width: 1120px; width: max-content;
      padding: 16px 26px; border-radius: 16px; background: rgba(9, 10, 16, 0.93); color: #fff; text-align: center;
      font: 600 24px/1.38 Inter, system-ui, sans-serif; border: 1px solid rgba(255,255,255,0.16);
      box-shadow: 0 16px 48px rgba(0,0,0,0.6); z-index: 2147483646; pointer-events: none; opacity: 0;
      transition: opacity 0.35s ease, transform 0.35s ease; }
    #demo-cap.top { bottom: auto; top: 74px; }
    #demo-cap.on { opacity: 1; transform: translate(-50%, 0); }
    #demo-cap small { display: block; margin-top: 6px; font: 500 18px/1.42 Inter, system-ui, sans-serif; color: #c9c9da; }
    #demo-title { position: fixed; inset: 0; z-index: 2147483645; display: flex; flex-direction: column; align-items: center;
      justify-content: center; gap: 16px; background: radial-gradient(900px 520px at 50% 45%, #1c1745, #07080b 70%);
      color: #fff; opacity: 0; transition: opacity 0.6s ease; pointer-events: none; text-align: center; }
    #demo-title.on { opacity: 1; }
    #demo-title .k { font: 700 16px Inter, system-ui, sans-serif; letter-spacing: 0.24em; text-transform: uppercase; color: #a99cff; }
    #demo-title .t { font: 800 68px/1.1 Inter, system-ui, sans-serif; letter-spacing: -0.02em; max-width: 1500px; }
    #demo-title .s { font: 500 28px/1.4 Inter, system-ui, sans-serif; color: #c9c9da; max-width: 1300px; }
  `;
  document.head.appendChild(style);
  const cursor = Object.assign(document.createElement("div"), { id: "demo-cursor" });
  const cap = Object.assign(document.createElement("div"), { id: "demo-cap" });
  const title = Object.assign(document.createElement("div"), { id: "demo-title" });
  document.body.append(cursor, cap, title);

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const ease = (k) => (k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2);
  const q = (sel) => document.querySelector(sel);
  const pos = { x: 960, y: 540 };
  const place = (x, y) => { pos.x = x; pos.y = y; cursor.style.transform = `translate(${x}px, ${y}px)`; };
  const center = (el) => { const r = el.getBoundingClientRect(); return { x: r.left + Math.min(r.width / 2, 150), y: r.top + Math.min(r.height / 2, 60) }; };
  const words = (s) => String(s).replace(/<[^>]+>/g, " ").split(/\s+/).filter(Boolean).length;

  // Real mouse input goes through the recorder (Input.dispatchMouseEvent), which acks when done.
  let seq = 0;
  const acks = new Map();
  window.__demoAck = (id) => { acks.get(id)?.(); acks.delete(id); };
  const input = (events) => new Promise((res) => {
    const id = ++seq;
    acks.set(id, res);
    window.__demoInput(JSON.stringify({ id, events }));
    setTimeout(() => { if (acks.has(id)) { acks.delete(id); res(); } }, 3000);
  });

  const d = window.demo = {
    sleep, q,
    async moveXY(x, y, ms = 750) {
      cursor.style.transition = `transform ${ms}ms cubic-bezier(0.22, 0.8, 0.24, 1)`;
      place(x, y);
      await sleep(ms + 60);
    },
    /** Scroll el into view in its own scroll container if it isn't visible. */
    async reveal(el) {
      let p = el.parentElement;
      while (p && p !== document.body) {
        const st = getComputedStyle(p);
        if (/(auto|scroll)/.test(st.overflowY) && p.scrollHeight > p.clientHeight) break;
        p = p.parentElement;
      }
      const r = el.getBoundingClientRect();
      const box = p && p !== document.body ? p.getBoundingClientRect() : { top: 0, bottom: innerHeight };
      if (r.top < box.top + 8 || r.bottom > box.bottom - 8) {
        el.scrollIntoView({ behavior: "smooth", block: r.height > (box.bottom - box.top) * 0.6 ? "start" : "center" });
        await sleep(750);
      }
    },
    async move(el, ms = 750) {
      if (!el) throw new Error("demo.move: missing element");
      await d.reveal(el);
      const p = center(el);
      await d.moveXY(p.x, p.y, ms);
    },
    press() { cursor.classList.remove("press"); void cursor.offsetWidth; cursor.classList.add("press"); },
    async click(el, after = 450) {
      if (!el) throw new Error("demo.click: missing element");
      await d.move(el);
      d.press();
      if (el.focus) el.focus({ preventScroll: true });
      if (typeof el.click === "function") el.click(); else el.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await sleep(after);
    },
    async hover(el, ms = 650) {
      await d.move(el, ms);
      el.dispatchEvent(new MouseEvent("mouseenter"));
      el.dispatchEvent(new MouseEvent("mouseover", { bubbles: true }));
    },
    unhover(el) { el.dispatchEvent(new MouseEvent("mouseleave")); el.dispatchEvent(new MouseEvent("mouseout", { bubbles: true })); },
    async type(el, text, perChar = 34) {
      el.focus({ preventScroll: true });
      for (const ch of text) {
        el.value += ch;
        el.dispatchEvent(new Event("input", { bubbles: true }));
        await sleep(perChar + Math.random() * 28);
      }
    },
    set(el, value) { el.value = value; el.dispatchEvent(new Event("input", { bubbles: true })); el.dispatchEvent(new Event("change", { bubbles: true })); },
    key(key, opts = {}, target = document) { target.dispatchEvent(new KeyboardEvent("keydown", { key, bubbles: true, cancelable: true, ...opts })); },

    // real mouse input (canvas hover, drag, wheel)
    async realMove(x, y, ms = 700, extra = {}) {
      cursor.style.transition = "none";
      const from = { ...pos }, steps = Math.max(6, Math.round(ms / 22));
      for (let i = 1; i <= steps; i++) {
        const k = ease(i / steps), px = from.x + (x - from.x) * k, py = from.y + (y - from.y) * k;
        place(px, py);
        await input([{ type: "mouseMoved", x: px, y: py, ...extra }]);
        await sleep(10);
      }
    },
    async realClick(x, y, modifiers = 0) {
      await d.realMove(x, y);
      d.press();
      await input([{ type: "mousePressed", x, y, button: "left", buttons: 1, clickCount: 1, modifiers },
        { type: "mouseReleased", x, y, button: "left", buttons: 0, clickCount: 1, modifiers }]);
      await sleep(400);
    },
    async realDrag(x0, y0, x1, y1, modifiers = 0, ms = 1100) {
      await d.realMove(x0, y0);
      await input([{ type: "mousePressed", x: x0, y: y0, button: "left", buttons: 1, clickCount: 1, modifiers }]);
      await d.realMove(x1, y1, ms, { button: "left", buttons: 1, modifiers });
      await sleep(150);
      await input([{ type: "mouseReleased", x: x1, y: y1, button: "left", buttons: 0, clickCount: 1, modifiers }]);
      await sleep(300);
    },
    async wheel(x, y, dy, times = 1, gap = 120) {
      await d.realMove(x, y, 450);
      for (let i = 0; i < times; i++) { await input([{ type: "mouseWheel", x, y, deltaX: 0, deltaY: dy }]); await sleep(gap); }
    },

    // captions and title cards
    async cap(text, sub = "", where = "bottom") {
      if (cap.classList.contains("on")) { cap.classList.remove("on"); await sleep(300); }
      cap.classList.toggle("top", where === "top");
      cap.innerHTML = text + (sub ? `<small>${sub}</small>` : "");
      cap.classList.add("on");
      await sleep(400);
    },
    /** A caption held long enough to read. */
    async say(text, sub = "", where = "bottom") {
      await d.cap(text, sub, where);
      await sleep(1100 + (words(text) + words(sub)) * 260);
    },
    async hide() { cap.classList.remove("on"); await sleep(450); },
    async title(text, sub = "", ms = 2800) {
      await d.hide();
      title.innerHTML = `<div class="k">Brainscan</div><div class="t">${text}</div>${sub ? `<div class="s">${sub}</div>` : ""}`;
      title.classList.add("on");
      await sleep(ms);
      title.classList.remove("on");
      await sleep(700);
    },
    async until(fn, timeout = 60000, what = "") {
      const t0 = performance.now();
      while (performance.now() - t0 < timeout) {
        try { if (fn()) return true; } catch { /* not yet */ }
        await sleep(200);
      }
      throw new Error(`demo.until timed out: ${what || fn.toString().slice(0, 120)}`);
    },

    // app helpers
    show(view) { document.dispatchEvent(new CustomEvent("brainscan:show", { detail: view })); },
    /** The last token in the Trace conversation whose text is `text`. */
    word(text) { return [...document.querySelectorAll("#tv-convo-body .ct")].reverse().find((e) => e.textContent.trim() === text) || null; },
    async latestChat() {
      const runs = await (await fetch("/brainscan/runs")).json();
      return runs.find((r) => r.id.startsWith("chat-") && !r.id.endsWith("-edited") && (r.tokens || 0) > 0)?.id || null;
    },
    async openRun(id) {
      const sel = q("#run-select");
      if (![...sel.options].some((o) => o.value === (id || ""))) sel.insertAdjacentHTML("beforeend", `<option value="${id || ""}">${id || "(none)"}</option>`);
      sel.value = id || "";
      sel.dispatchEvent(new Event("change"));
    },
    async api(method, path, body) {
      const res = await fetch(path, { method, headers: { "Content-Type": "application/json" }, body: body ? JSON.stringify(body) : undefined });
      const text = await res.text();
      return text ? JSON.parse(text) : null;
    },
    /** Put the app in a known state before a scene is recorded. */
    async reset(o = {}) {
      document.activeElement?.blur?.();
      const status = await d.api("GET", "/brainscan/status");
      for (const iv of status.interventions || []) await d.api("DELETE", `/brainscan/interventions/${iv.id}`);
      if (o.revert) await d.api("POST", "/brainscan/workbench/revert", {});
      if (o.clearEdits) for (const e of (await d.api("GET", "/brainscan/workbench")).edits) await d.api("DELETE", `/brainscan/workbench/edits/${e.id}`);
      if (o.clearChecks) for (const c of await d.api("GET", "/brainscan/checks")) await d.api("DELETE", `/brainscan/checks/${c.id}`);
      if (o.apply) {
        const wb = await d.api("GET", "/brainscan/workbench");
        if (wb.edits.length && !wb.applied.length) await d.api("POST", "/brainscan/workbench/apply", {});
      }
      document.dispatchEvent(new CustomEvent("brainscan:workbench"));
      if (!q("#guide").hidden) q("#guide-close").click();
      if (!q("#model-picker").hidden) q("#mp-close").click();
      if (!q("#gv-palette").hidden) d.key("Escape", {}, q("#gv-q"));
      if (q("#wb-kind").value !== "scale") d.set(q("#wb-kind"), "scale");  // the lesion form, the one the videos use
      if (o.view) d.show(o.view);
      await sleep(1500);
      if (o.run !== undefined) {
        await d.openRun(o.run === "latest-chat" ? await d.latestChat() : o.run);
        await sleep(2500);
        if (!o.run) {  // the picker re-renders with the newest run showing; say none is open
          const sel = q("#run-select");
          sel.insertAdjacentHTML("afterbegin", `<option value="">No run open: ask the model below</option>`);
          sel.value = "";
        }
      }
      if (o.view === "weights") {
        for (let i = 0; i < 6; i++) { d.key("Escape"); await sleep(60); }
        q("#gv-fit").click();
      }
      window.scrollTo(0, 0);
      for (const el of document.querySelectorAll("#view-workbench, .workbench, #messages, #an-detail, #tv-convo-body")) el.scrollTop = 0;
      d.hide();
      await sleep(1500);
      return true;
    },
    /** For setups: trace a word in the conversation and wait for its routes. */
    async traceQuick(text) {
      await d.until(() => d.word(text), 30000, `the word ${text}`);
      d.word(text).click();
      await d.until(() => q("#an-detail .routes li[data-i]"), 120000, "routes");
      await sleep(1200);
    },
  };
  return true;
}

const FIZZ = "20261022-030423-fix_fizzbuzz-93e6";

export const SCENES = [
  // ------------------------------------------------------------------ 1. overview
  {
    name: "01-overview",
    setup: async () => { await window.demo.reset({ view: "scanner", run: "20261022-030423-fix_fizzbuzz-93e6", revert: true }); return true; },
    script: async () => {
      const d = window.demo, q = d.q;
      await d.title("Trace and tune open-weights models", "See what happens inside a model while it works, on your own GPU", 3400);
      await d.say("Brainscan runs an open-weights model on your machine and records its internals, token by token.",
        "Here: Qwen3.5-9B on a single RTX 4090.");
      await d.move(q(".tabs"));
      await d.sleep(600);
      await d.click(q('.tabs [data-view="scanner"]'));
      await d.say("1 · Watch: the transcript, a per-token timeline, the logit lens and steering.",
        "Ask from the app, or point any OpenAI-compatible agent at it. Every request becomes a recorded run.");
      await d.click(q('.tabs [data-view="weights"]'));
      await d.sleep(1200);
      await d.say("2 · Trace: the model as a graph of attention heads and neuron clusters.",
        "Click any word to see which units and layers produced it.");
      await d.click(q('.tabs [data-view="workbench"]'));
      await d.say("3 · Tune: change the weights, measure what changed, and export the variant.");
      await d.click(q("#help-btn"));
      await d.sleep(600);
      await d.cap("The guide lists every tool in that order, each with a Show me button.", "Press ? to open it from anywhere.", "top");
      for (const s of document.querySelectorAll("#guide .guide-step")) { s.scrollIntoView({ behavior: "smooth", block: "start" }); await d.sleep(2600); }
      await d.click(q("#guide-close"));
      await d.click(q("#model-badge"));
      await d.until(() => q("#model-picker .mp-row"), 30000, "model list");
      await d.sleep(500);
      await d.move(q("#model-picker .mp-list"));
      await d.say("Switch models from the top bar: any Hugging Face checkpoint or Ollama tag already on this machine.",
        "Each one is checked against free GPU memory first. Nothing is downloaded.");
      await d.click(q("#mp-close"));
      await d.title("Next: ask the model and watch it work", "", 2400);
      return true;
    },
  },

  // ------------------------------------------------------------------ 2. ask and watch live
  {
    name: "02-ask-and-watch-live",
    setup: async () => { await window.demo.reset({ view: "weights", run: "", revert: true }); return true; },
    script: async () => {
      const d = window.demo, q = d.q;
      await d.title("Ask the model. Watch it work.", "Every token is recorded as it is generated", 3000);
      await d.say("The Trace tab: the conversation on the left, the model in the middle.",
        "Each dot is an attention head or a cluster of neurons. Layers run left to right, from input to output.");
      await d.moveXY(1150, 470, 900);
      await d.sleep(800);
      const c = q("#tv-convo .composer");
      await d.cap("Ask a question right here. No harness, no setup.");
      await d.click(c.querySelector(".composer-new"));
      d.set(c.querySelector(".composer-think"), "off");
      d.set(c.querySelector(".composer-max"), "60");
      d.set(c.querySelector(".composer-temp"), "0");
      const ta = c.querySelector("textarea");
      await d.click(ta);
      await d.type(ta, "What is the capital of Australia? Answer in one sentence.");
      await d.sleep(500);
      d.key("Enter", {}, ta);
      await d.cap("The answer streams in, and the graph lights up with the units that fired at each token.");
      await d.moveXY(1150, 500, 900);
      await d.until(() => /^Done/.test(c.querySelector(".composer-status").innerText), 120000, "the answer");
      await d.sleep(1500);
      const last = [...document.querySelectorAll("#tv-convo-body .cb")].pop();
      await d.move(last);
      await d.say("Done. Words the model found surprising are underlined.", "Every word is clickable: the trace video starts there.");
      await d.move(q("#an-trace-wrap"));
      await d.cap("The timeline under the graph has one column per token. Press play to replay the run.");
      await d.click(q("#an-play"));
      await d.sleep(7000);
      if (q("#an-play").getAttribute("aria-label") === "Pause") await d.click(q("#an-play"));
      await d.title("Next: why did it write “Canberra”?", "", 2400);
      return true;
    },
  },

  // ------------------------------------------------------------------ 3. the Watch tab with an agent run
  {
    name: "03-watch-agent-runs",
    setup: async () => { await window.demo.reset({ view: "scanner", run: "20261022-030423-fix_fizzbuzz-93e6" }); return true; },
    script: async () => {
      const d = window.demo, q = d.q;
      await d.title("Watch: every run, recorded", "Transcripts, tool calls and a per-token timeline", 3000);
      await d.move(q("#run-select"));
      await d.say("Point any OpenAI-compatible agent at Brainscan's /v1 endpoint and each request is recorded.",
        "This run is a coding agent fixing a FizzBuzz bug with tools.", "top");
      const box = q("#messages");
      await d.move(box);
      await d.cap("The transcript: messages, tool calls and their results.", "", "top");
      box.scrollTo({ top: box.scrollHeight * 0.4, behavior: "smooth" }); await d.sleep(2400);
      box.scrollTo({ top: box.scrollHeight, behavior: "smooth" }); await d.sleep(2400);
      const r = q("#scope").getBoundingClientRect();
      await d.cap("The timeline: one column per token, with events, the logit lens by layer, surprisal and projections.",
        "Hover for details. Scroll to pan, ctrl-scroll to zoom.", "top");
      await d.realMove(r.left + r.width * 0.3, r.top + r.height * 0.5, 900);
      for (const f of [0.4, 0.5, 0.6, 0.7]) await d.realMove(r.left + r.width * f, r.top + r.height * 0.5, 900);
      await d.sleep(1200);
      await d.realClick(r.left + r.width * 0.7, r.top + r.height * 0.5);
      await d.until(() => q("#inspector .tok-hero"), 30000, "inspector");
      await d.sleep(800);
      await d.move(q("#inspector .tok-hero"));
      await d.say("Click a token: what the model predicted there, and each layer's guess for it (the logit lens).", "", "top");
      const lens = q("#inspector .lens-wrap");
      if (lens) { await d.move(lens); await d.sleep(2500); }
      const tr = q("#insp-trace");
      if (tr) {
        await d.move(tr);
        await d.cap("Trace this token opens it in the graph: the layers, units and knockouts behind it.", "", "top");
        await d.sleep(1500);
        await d.click(tr);
        await d.until(() => q("#an-detail .d-hero"), 90000, "trace panel");
        await d.sleep(4500);
      }
      await d.title("Next: trace a word", "", 2200);
      return true;
    },
  },

  // ------------------------------------------------------------------ 4. trace a word
  {
    name: "04-trace-a-word",
    setup: async () => { await window.demo.reset({ view: "weights", run: "latest-chat", revert: true }); return true; },
    script: async () => {
      const d = window.demo, q = d.q;
      await d.title("Why did it write that?", "Click a word to trace it back through the model", 3000);
      await d.until(() => d.word("Canberra"), 20000, "Canberra");
      await d.cap("Click any word in the conversation. Here, “Canberra”.");
      await d.click(d.word("Canberra"));
      await d.cap("Brainscan reads the context again and splits the word's logit into what each unit wrote.");
      await d.until(() => q("#an-detail .d-hero"), 90000, "panel");
      await d.until(() => q("#an-detail .routes li[data-i]"), 90000, "routes");
      await d.sleep(1500);
      const stats = [...document.querySelectorAll("#an-detail .d-stats b")].map((b) => b.textContent);
      await d.move(q("#an-detail .tp-target"));
      await d.say(`“Canberra”: probability ${stats[0]}, rank ${stats[1]}.`, "Every unit on the graph is colored by its push on this word: up or down.");
      const alts = q("#an-detail .tp-alts");
      if (alts) { await d.move(alts); await d.say("The runners-up it also considered. Click one to trace it instead."); }
      const routes = q("#an-detail .routes");
      await d.move(routes);
      await d.say("The trace, start to end: the strongest routes from the input to the word it wrote.",
        "START is the token the model read. Each hop is a head (h) or a neuron cluster (c). END is the output word.");
      await d.cap("Hover a route to light it up on the graph, one at a time.");
      for (const li of [...routes.querySelectorAll("li[data-i]")].slice(0, 3)) { await d.hover(li, 600); await d.sleep(1700); d.unhover(li); }
      await d.click(q("#tp-follow"));
      await d.cap("▶ Follow route 1: the view flies from START, through each unit, to END.");
      await d.sleep(9000);
      const chart = q("#an-detail .layer-chart");
      await d.move(chart);
      const layers = [...chart.querySelectorAll("g.lc-layer")].map((g) => {
        const m = g.querySelector("title").textContent.match(/Layer (\d+): heads ([-+\d.]+), MLP ([-+\d.]+)/);
        return m ? { l: Number(m[1]), v: Number(m[2]) + Number(m[3]), g } : null;
      }).filter(Boolean).sort((a, b) => b.v - a.v);
      await d.say("Where the logit came from: each layer's heads and MLP clusters, and the running total.",
        layers.length ? `The biggest push comes from layer ${layers[0].l}. Click a layer to fly to it.` : "");
      if (layers.length) { await d.click(layers[0].g); await d.sleep(2200); }
      const lens = q("#an-detail .lens-strip");
      if (lens) {
        await d.move(lens);
        const first = lens.parentElement.querySelector(".hint b")?.textContent;
        await d.say("When the model knew: the logit lens reads each layer's best guess.", first ? `“Canberra” is the top guess from layer ${first} on.` : "");
      }
      await d.move(q("#an-detail .tp-units"));
      await d.say("The units that pushed it up most, and the ones that pulled it down.",
        "Direct effects add up to the logit. What a unit does through later layers takes a knockout.");
      await d.title("Next: knockout tests", "", 2200);
      return true;
    },
  },

  // ------------------------------------------------------------------ 5. knockouts
  {
    name: "05-knockout-tests",
    setup: async () => {
      const d = window.demo;
      await d.reset({ view: "weights", run: "latest-chat", revert: true });
      await d.traceQuick("Canberra");
      d.q("#tp-ko-top").scrollIntoView({ block: "center" });
      await d.sleep(800);
      return true;
    },
    script: async () => {
      const d = window.demo, q = d.q;
      await d.title("Knockout tests", "Remove units, run again, compare", 2800);
      const btn = q("#tp-ko-top");
      await d.move(btn);
      await d.say("Direct effects show what each unit wrote to the output. A knockout shows what the model actually needs.",
        "Knock out the 5 units that pushed “Canberra” up the most.");
      await d.click(btn);
      await d.cap("The whole context is read again with those units' writes set to zero.");
      await d.until(() => q("#tp-ko-list li .ko-nums"), 120000, "knockout");
      await d.sleep(600);
      let li = q("#tp-ko-list li");
      await d.move(li);
      const nums = li.querySelector(".ko-nums").innerText.replace(/\s+/g, " ").split(" logit")[0];
      const now = ((li.querySelector(".hint").innerText.match(/Most likely now: (.*)\./) || [])[1] || "").replace(/·(?=\S)/g, "");  // · marks a leading space
      await d.say(`Without them: ${nums}.`, now ? `Most likely now: ${now}.` : "");
      await d.move(li.querySelector(".ko-bars"));
      await d.say("Direct vs total effect: what the units wrote to the output, against everything they did through later layers.",
        li.querySelector(".hint").innerText.split(" Most likely")[0]);
      const r = q("#gv-canvas").getBoundingClientRect();
      await d.cap("Test any set of units: shift-drag a box on the graph to select them…");
      // The framed trace sits in a band across the middle; heads above the stream line, clusters below.
      for (const [fx, fy] of [[0.5, 0.47], [0.4, 0.56], [0.6, 0.46], [0.3, 0.5]]) {
        const cx = r.left + r.width * fx, cy = r.top + r.height * fy;
        await d.realDrag(cx - 38, cy - 24, cx + 38, cy + 24, 8, 1100);  // small: one knockout removes at most 64 units
        await d.sleep(400);
        if (!q("#gv-selchip").hidden) break;
      }
      await d.until(() => !q("#gv-selchip").hidden, 3000, "selection");
      await d.move(q("#gv-selchip"));
      await d.sleep(1500);
      await d.cap("…then knock them out.");
      await d.click(q("#gv-selko"));
      await d.until(() => document.querySelectorAll("#tp-ko-list li .ko-nums").length >= 2, 120000, "selection knockout");
      await d.sleep(600);
      li = q("#tp-ko-list li");
      await d.move(li);
      const nums2 = li.querySelector(".ko-nums").innerText.replace(/\s+/g, " ").split(" logit")[0];
      await d.say(`${li.querySelector(".ko-head b").innerText}: ${nums2}.`, li.querySelector(".hint").innerText.split(" Most likely")[0]);
      d.key("Escape");
      await d.title("Next: make a knockout permanent", "", 2200);
      return true;
    },
  },

  // ------------------------------------------------------------------ 6. explore the graph
  {
    name: "06-explore-the-graph",
    setup: async () => {
      const d = window.demo;
      await d.reset({ view: "weights", run: "latest-chat", revert: true });
      await d.traceQuick("Canberra");
      return true;
    },
    script: async () => {
      const d = window.demo, q = d.q;
      await d.title("Explore the graph", "Open any unit, jump by name, follow paths", 2800);
      const r = q("#gv-canvas").getBoundingClientRect();
      await d.cap("Scroll to zoom, drag to pan.", "Attention heads sit above the residual stream, neuron clusters below it.");
      await d.wheel(r.left + r.width * 0.62, r.top + r.height * 0.42, -260, 6, 140);
      await d.sleep(900);
      await d.realDrag(r.left + r.width * 0.62, r.top + r.height * 0.5, r.left + r.width * 0.42, r.top + r.height * 0.55, 0, 1000);
      await d.sleep(1200);
      const row = q("#an-detail .tp-units li[data-g]");
      const name = row.querySelector(".tp-name").textContent;
      await d.cap(`Open a unit: ${name}, the one that pushed “Canberra” up the most.`);
      await d.click(row);
      await d.until(() => !q("#an-detail .tp-units"), 20000, "unit panel");
      await d.sleep(2500);
      await d.realMove(r.left + r.width / 2, r.top + r.height / 2, 900);
      await d.say("Its weights, the tokens it writes toward, its connections…", "…and at this token, its attention pattern or its most active neurons.");
      const secs = [...q("#an-detail").querySelectorAll(".d-sec")];
      for (const s of secs.slice(1, 6)) { await d.move(s, 600); await d.sleep(1800); }
      await d.click(q("#gv-search"));
      const inp = q("#gv-q");
      await d.cap("Jump to any unit or parameter by name: ⌘K or /.");
      await d.type(inp, "L24 h30", 70);
      await d.sleep(1400);
      d.key("Enter", {}, inp);
      await d.sleep(2600);
      await d.click(q("#gv-paths-btn"));
      await d.click(q('#gv-path-mode [data-pm="explain"]'));
      await d.say("The path explorer: routes out of or into any units, between two of them, or the circuit behind the next token.");
      const hops = q("#gv-path-hops");
      await d.move(hops);
      d.set(hops, "4");
      await d.cap("Tune the hops and branches; hover a path to light it up.");
      await d.sleep(2500);
      for (const li of [...document.querySelectorAll("#gv-path-list li")].slice(0, 3)) { await d.hover(li, 600); await d.sleep(1500); d.unhover(li); }
      await d.click(q("#gv-paths-close"));
      for (let i = 0; i < 3; i++) d.key("Escape");
      await d.title("Next: from experiment to edit", "", 2200);
      return true;
    },
  },

  // ------------------------------------------------------------------ 7. make it permanent, check, re-run
  {
    name: "07-edit-and-check",
    setup: async () => {
      const d = window.demo, q = d.q;
      await d.reset({ view: "weights", run: "latest-chat", revert: true, clearEdits: true, clearChecks: true });
      await d.traceQuick("Canberra");
      q("#tp-ko-top").click();
      await d.until(() => q("#tp-ko-list li .ko-nums"), 120000, "knockout");
      q("#tp-ko-list li").scrollIntoView({ block: "center" });
      await d.sleep(1000);
      return true;
    },
    script: async () => {
      const d = window.demo, q = d.q;
      await d.title("From experiment to edit", "Make a knockout permanent, then check what it changed", 3000);
      const li = q("#tp-ko-list li");
      await d.move(li);
      await d.say("A knockout is a test. Make permanent turns it into a real weight edit.",
        "The 5 units' output weights are zeroed, as one edit you can undo.");
      await d.click(li.querySelector("[data-ko-perm]"));
      await d.until(() => q("#tp-ko-list .pill.ok"), 120000, "permanent");
      await d.until(() => !q("#gv-edits").hidden, 5000, "edits panel");
      await d.sleep(1200);
      await d.move(q("#gv-edits .ed-list li"));
      await d.say("The Edits panel opens beside the graph, with the edit applied.", "Restore the original weights any time. It's exact.");
      await d.until(() => q("#an-detail .cmp-line"), 90000, "before/after line");
      const cmp = q("#an-detail .cmp-line");
      await d.move(cmp);
      await d.say(`The trace now compares with the original weights: ${cmp.innerText.replace(/\s+/g, " ").trim()}.`,
        "Each unit is colored by how the edit changed its effect on the word.");
      await d.click(q("#tp-add-check"));
      await d.say("Add as a check: save this word as something to measure after every edit.");
      const form = q("#gv-edits form[data-add-check]");
      await d.cap("Or type one, to catch side effects.");
      await d.click(form.querySelector('[name="prompt"]'));
      await d.type(form.querySelector('[name="prompt"]'), "The capital of France is");
      await d.click(form.querySelector('[name="expect"]'));
      await d.type(form.querySelector('[name="expect"]'), "Paris");
      await d.click(form.querySelector('button[type="submit"]'));
      await d.until(() => document.querySelectorAll("#gv-edits .ck-list .ck-val").length >= 2, 120000, "checks");
      await d.sleep(800);
      await d.move(q("#gv-edits .ck-list"));
      await d.say("Each check shows its probability on the original weights → now.", "What the edit broke, and what it left alone.");
      await d.click(q("#gv-edits [data-replay]"));
      await d.cap("Re-run this chat: the same question, greedy, on the original and the edited weights.");
      await d.until(() => q("#gv-edits .rp-card"), 180000, "replay");
      await d.sleep(800);
      await d.move(q("#gv-edits .rp-card"));
      const cols = [...document.querySelectorAll("#gv-edits .rp-cols p")].map((p) => p.innerText.trim());
      await d.say(`Original: “${cols[0]}”`, `Edited: “${cols[1]}”`);
      const diff = q("#gv-edits [data-trace-diff]");
      if (diff) {
        await d.click(diff);
        await d.cap("Trace the difference: the first word that changed, traced on the edited model.");
        await d.sleep(2500);
        await d.until(() => q("#an-detail .cmp-line") && q("#an-detail .routes li[data-i]"), 180000, "diff trace");
        await d.sleep(1500);
        const c2 = q("#an-detail .cmp-line");
        const word = q("#an-detail .tp-target")?.innerText.trim().replace(/^·/, "");  // · marks a leading space
        await d.move(c2);
        await d.say(`“${word}”: ${c2.innerText.replace(/\s+/g, " ").trim()}.`, "The routes now show the circuit behind the change.");
      }
      await d.title("Next: the Tune tab", "", 2200);
      return true;
    },
  },

  // ------------------------------------------------------------------ 8. the Tune tab
  {
    name: "08-tune-tab",
    setup: async () => {
      const d = window.demo, q = d.q;
      await d.reset({ view: "workbench", apply: true });
      d.set(q("#wb-kind"), "scale");
      d.set(q("#wb-comp"), "head");
      q("#view-workbench").scrollTop = 0;
      await d.sleep(1500);
      return true;
    },
    script: async () => {
      const d = window.demo, q = d.q;
      await d.title("The Tune tab", "Edit, compare, check, export", 2800);
      await d.say("Tune holds every edit tool in one place, in order: find a direction, edit, compare, check, export.", "", "top");
      await d.move(q("#wb-list"));
      await d.say("2 · Edit weights. The stack holds the knockout made permanent in Trace.",
        "Scale or lesion an attention block, an MLP block, one head or one neuron here as well.", "top");
      await d.move(q("#wb-form"));
      await d.sleep(2000);
      const rp = q(".wb-replay");
      await d.move(rp);
      await d.say("3 · Compare: the open chat on the original and the edited weights, with the first word where they part.", "", "top");
      const ck = q("#h-checks").closest("section");
      await d.move(ck);
      await d.click(ck.querySelector("[data-run-checks]"));
      await d.until(() => ck.querySelectorAll(".ck-val").length >= 1 && !ck.querySelector("[data-run-checks]").disabled, 120000, "checks");
      await d.sleep(600);
      await d.say("4 · Behavior checks: each one's probability on the original weights → now.", "", "top");
      await d.move(q("#wb-path"));
      await d.say("5 · Export the edited model as a Hugging Face checkpoint.", "brainscan_edits.json records every edit and the base model.", "top");
      await d.move(q("#wb-export"));
      await d.sleep(1200);
      await d.click(q("#wb-revert"));
      await d.cap("Restore original weights: back to the model you started with.", "", "top");
      await d.until(() => /original/.test(q("#wb-state").innerText), 60000, "restored");
      await d.move(q("#wb-state"));
      await d.sleep(2500);
      await d.title("Next: directions and steering", "", 2200);
      return true;
    },
  },

  // ------------------------------------------------------------------ 9. directions and steering
  {
    name: "09-directions-and-steering",
    setup: async () => {
      const d = window.demo, q = d.q;
      await d.reset({ view: "workbench", revert: true });
      q("#wb-a").value = ""; q("#wb-b").value = "";
      q("#wb-sweep").innerHTML = ""; q("#wb-save-row").hidden = true; q("#wb-find-msg").textContent = "";
      q("#view-workbench").scrollTop = 0;
      await d.sleep(800);
      return true;
    },
    script: async () => {
      const d = window.demo, q = d.q;
      await d.title("Directions and steering", "Find a concept in the residual stream, then push on it", 3000);
      await d.click(q("#wb-example"));
      await d.say("1 · Find a direction from two sets of examples, one per line.", "Here: French sentences vs their English translations.", "top");
      await d.click(q("#wb-analyze"));
      await d.cap("Brainscan reads the residual stream for every example, at every layer, and scores how well the difference separates held-out examples.", "", "top");
      await d.until(() => q("#wb-sweep svg"), 180000, "sweep");
      await d.sleep(800);
      await d.move(q("#wb-sweep"));
      await d.say(q("#wb-find-msg").innerText.split(". ")[0] + ".", "Click a bar to pick a layer, name it, and save it to the direction bank.", "top");
      await d.move(q("#wb-dirs"));
      await d.say("Saved directions show up in Watch: as readouts on the timeline, and for steering.", "", "top");
      await d.click(q('.tabs [data-view="scanner"]'));
      await d.sleep(1500);
      const form = q("#iv-form");
      await d.move(form);
      d.set(q("#iv-direction"), "language_fr");
      await d.sleep(400);
      d.set(q("#iv-layer"), "16");
      d.set(q("#iv-mode"), "add");
      d.set(q("#iv-strength"), "0.8");
      d.set(q("#iv-phase"), "gen");
      await d.say("Steering: add the French direction at layer 16 while the model writes.", "Strength 0.8 × the residual's size, on generated tokens only.", "top");
      await d.click(form.querySelector('button[type="submit"]'));
      await d.until(() => !q("#steer-banner").hidden, 10000, "steering on");
      await d.move(q("#iv-list"));
      await d.sleep(1800);
      const c = q("#view-scanner .composer");
      await d.click(c.querySelector(".composer-new"));
      d.set(c.querySelector(".composer-think"), "off");
      d.set(c.querySelector(".composer-max"), "60");
      d.set(c.querySelector(".composer-temp"), "0");
      const ta = c.querySelector("textarea");
      await d.click(ta);
      await d.cap("Ask in English…", "", "top");
      await d.type(ta, "Describe a sunny morning in two sentences.");
      await d.sleep(400);
      d.key("Enter", {}, ta);
      await d.until(() => /^Done/.test(c.querySelector(".composer-status").innerText), 120000, "the answer");
      await d.sleep(1200);
      const msg = [...document.querySelectorAll("#messages .msg")].pop();
      await d.move(msg);
      await d.say("…and French starts creeping into the answer.", "Steered runs are marked ▲steered in the run list, so they're never mixed up with normal ones.", "top");
      await d.click(q("#iv-list li[data-id] button"));
      await d.cap("Remove the intervention and the model is back to normal.", "", "top");
      await d.sleep(2600);
      await d.title("Brainscan", "Watch · Trace · Tune: open-weights models, on your own GPU", 3400);
      return true;
    },
  },
];
