// Action bars: every tab shows, always, what can be done right now. Actions that need something
// first stay visible, greyed, and say what they need when hovered or clicked. Each tab's module
// registers its actions here (defineBar) and calls refreshBar when its state changes; the same
// list feeds the Do… palette (⌘K or Ctrl+K), which searches every tab's actions and runs one,
// switching tabs if needed.

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const TAB = { scanner: "Watch", weights: "Trace", workbench: "Tune" };
const TONE = { Ablate: "ex", Steer: "ex", Change: "ch", Verify: "vf", Ship: "vf" };

const BARS = new Map();   // tab -> { groups, context, actions, el, sig, note }
let view = null;          // the tab on screen

/** An action's state: null when it can run now, or what it needs first. */
const needs = (a) => { try { return a.need ? a.need() : null; } catch { return "Not available yet"; } };

/** spec: { groups: [names in order], context: () => html, actions: [{ id, group, label, hint, need, run, choices }] }
 *  `choices` (optional) returns [{ label, run }]: the action opens a menu of them. */
export function defineBar(tab, spec) {
  const bar = { ...spec, el: document.querySelector(`[data-bar="${tab}"]`), sig: "", note: null };
  BARS.set(tab, bar);
  if (bar.el) bar.el.addEventListener("click", (ev) => onBarClick(tab, ev));
  refreshBar(tab);
}

/** Re-render a tab's bar if what it shows changed (cheap to call often). */
export function refreshBar(tab) {
  const bar = BARS.get(tab);
  if (!bar?.el) return;
  let ctx = "";
  try { ctx = bar.context(); } catch { /* not ready */ }
  const state = bar.actions.map((a) => [a.id, needs(a), typeof a.label === "function" ? a.label() : a.label]);
  const sig = JSON.stringify([ctx, state, bar.note]);
  if (sig === bar.sig) return;
  bar.sig = sig;
  const label = (a) => (typeof a.label === "function" ? a.label() : a.label);
  bar.el.innerHTML = `<p class="ab-ctx">${bar.note ? `<span class="ab-note">${esc(bar.note)}</span>` : ctx}</p>
    <div class="ab-groups">${bar.groups.map((g) => {
      const acts = bar.actions.filter((a) => a.group === g);
      return acts.length ? `<div class="ab-group"><span class="ab-gl">${esc(g)}</span>${acts.map((a) => {
        const why = needs(a);
        return `<button type="button" class="ab-btn ${TONE[g] || ""}${why ? " off" : ""}" data-id="${a.id}" ${why ? 'aria-disabled="true"' : ""}
          title="${esc(why ? `Needs: ${why}` : a.hint || "")}">${esc(label(a))}${a.choices ? " ▾" : ""}</button>`;
      }).join("")}</div>` : "";
    }).join("")}</div>`;
}

/** A short message in a tab's bar, in place of its context line for a few seconds. */
export function noteBar(tab, text) { if (BARS.has(tab)) flashNote(tab, text); }

function flashNote(tab, text) {
  const bar = BARS.get(tab);
  bar.note = text;
  refreshBar(tab);
  clearTimeout(bar.noteTimer);
  bar.noteTimer = setTimeout(() => { bar.note = null; refreshBar(tab); }, 3500);
}

function onBarClick(tab, ev) {
  const btn = ev.target.closest(".ab-btn[data-id]");
  if (!btn) return;
  const a = BARS.get(tab).actions.find((x) => x.id === btn.dataset.id);
  if (!a) return;
  const why = needs(a);
  if (why) { flashNote(tab, `Needs: ${why}`); return; }
  if (a.choices) { openMenu(btn, a.choices()); return; }
  a.run();
}

// ---------------------------------------------------------------- a small menu for actions with choices
function openMenu(anchor, items) {
  closeMenu();
  if (!items.length) return;
  const r = anchor.getBoundingClientRect();
  const menu = document.createElement("div");
  menu.className = "ab-menu glass";
  menu.id = "ab-menu";
  menu.style.left = `${Math.min(r.left, innerWidth - 240)}px`;
  menu.style.top = `${r.bottom + 4}px`;
  menu.innerHTML = items.map((it, i) => `<button type="button" data-i="${i}">${esc(it.label)}</button>`).join("");
  menu.onclick = (ev) => { const b = ev.target.closest("[data-i]"); if (b) { closeMenu(); items[Number(b.dataset.i)].run(); } };
  document.body.appendChild(menu);
  setTimeout(() => document.addEventListener("pointerdown", outside, { once: true }), 0);
}
const outside = (ev) => { if (!ev.target.closest("#ab-menu")) closeMenu(); else document.addEventListener("pointerdown", outside, { once: true }); };
function closeMenu() { $("#ab-menu")?.remove(); }

// ---------------------------------------------------------------- the Do… palette
const D = { items: [], active: 0 };

function allItems() {
  const out = [];
  for (const [tab, bar] of BARS) {
    for (const a of bar.actions) {
      const why = needs(a), label = typeof a.label === "function" ? a.label() : a.label;
      if (a.choices && !why) {
        for (const c of a.choices()) out.push({ tab, group: a.group, label: `${label}: ${c.label}`, why: null, run: c.run });
      } else {
        out.push({ tab, group: a.group, label, why, run: a.run, hint: a.hint });
      }
    }
  }
  return out;
}

function openDo() {
  if (!$("#do-palette")) return;
  closeMenu();
  $("#do-palette").hidden = false;
  const q = $("#do-q");
  q.value = "";
  q.focus();
  searchDo();
}
function closeDo() { const p = $("#do-palette"); if (p) p.hidden = true; }

function searchDo() {
  const q = $("#do-q").value.trim().toLowerCase();
  const words = q.split(/\s+/).filter(Boolean);
  const items = allItems().filter((it) => {
    const hay = `${it.label} ${it.group} ${TAB[it.tab]}`.toLowerCase();
    return words.every((w) => hay.includes(w));
  });
  // The tab on screen first, then what can run now.
  items.sort((a, b) => (b.tab === view) - (a.tab === view) || (!!a.why) - (!!b.why));
  // An action two tabs share (Restore original, Re-run the chat) is listed once, from the nearest tab.
  const seen = new Set();
  D.items = items.filter((it) => !seen.has(it.label) && seen.add(it.label));
  D.active = Math.min(D.active, Math.max(0, D.items.length - 1));
  if (!q) D.active = 0;
  renderDo();
}

function renderDo() {
  const list = $("#do-list");
  list.innerHTML = D.items.map((it, i) => `<li role="option" data-i="${i}" class="${i === D.active ? "active" : ""}${it.why ? " off" : ""}" aria-selected="${i === D.active}">
      <span class="do-label">${esc(it.label)}</span>
      <span class="do-meta">${it.why ? `<span class="do-why">needs: ${esc(it.why)}</span>` : ""}<span class="do-where">${TAB[it.tab]} · ${esc(it.group.toLowerCase())}</span></span></li>`).join("")
    || `<li class="empty">No action matches. Try “knock”, “check” or “export”.</li>`;
  list.querySelector("li.active")?.scrollIntoView({ block: "nearest" });
}

function runDo(i) {
  const it = D.items[i];
  if (!it) return;
  if (it.why) { $("#do-note").textContent = `${it.label} needs: ${it.why}`; return; }
  closeDo();
  if (it.tab !== view) {
    document.dispatchEvent(new CustomEvent("brainscan:show", { detail: it.tab }));
    setTimeout(it.run, 120);
  } else {
    it.run();
  }
}

function boot() {
  document.addEventListener("brainscan:view", (ev) => { view = ev.detail; if (BARS.has(view)) refreshBar(view); });
  $("#do-btn")?.addEventListener("click", openDo);
  const q = $("#do-q");
  if (q) {
    q.addEventListener("input", () => { $("#do-note").textContent = ""; searchDo(); });
    q.addEventListener("keydown", (ev) => {
      if (ev.key === "ArrowDown") { D.active = Math.min(D.items.length - 1, D.active + 1); renderDo(); ev.preventDefault(); }
      else if (ev.key === "ArrowUp") { D.active = Math.max(0, D.active - 1); renderDo(); ev.preventDefault(); }
      else if (ev.key === "Enter") { runDo(D.active); ev.preventDefault(); }
      else if (ev.key === "Escape") { closeDo(); ev.preventDefault(); }
      ev.stopPropagation();
    });
    $("#do-list").addEventListener("click", (ev) => { const li = ev.target.closest("li[data-i]"); if (li) runDo(Number(li.dataset.i)); });
    $("#do-palette").addEventListener("pointerdown", (ev) => { if (ev.target.id === "do-palette") closeDo(); });
  }
  // ⌘K / Ctrl+K anywhere opens Do…
  document.addEventListener("keydown", (ev) => {
    if ((ev.metaKey || ev.ctrlKey) && ev.key.toLowerCase() === "k") {
      ev.preventDefault(); ev.stopImmediatePropagation();
      if ($("#do-palette")?.hidden) openDo(); else closeDo();
    }
  }, true);
}
boot();
