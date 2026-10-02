// EzeAi Comfy Agent - sidebar UI.
//
// Design (from Alusi / Ikenga / Content Studio Operator):
//   * the conversation is the primary surface; tool activity folds into one
//     collapsible block per turn
//   * every change is a CARD with an exact diff; nothing touches the canvas
//     until Apply (Auto mode applies simple edits, never removals)
//   * a card is re-validated against the canvas as it is NOW before applying
//   * diagnostics (models, runtime, settings, memory) live in the Inspector
// All model text is escaped before any formatting is applied.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const API = "/ezeai/agent";
const TAB = "ezeai-comfy-agent";

// ------------------------------------------------------------------ helpers

async function call(path, body) {
  const opts = body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  };
  const res = await api.fetchApi(API + path, opts);
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return res.json();
}

function el(tag, props = {}, kids = []) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") n.className = v;
    else if (k === "text") n.textContent = v;
    else if (k === "html") n.innerHTML = v;            // only ever given escaped markup
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v === true ? "" : v);
  }
  for (const c of [].concat(kids)) if (c) n.append(c);
  return n;
}

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// Small safe markdown: escape first, then format.
function md(text) {
  let s = esc(text);
  s = s.replace(/```[a-z]*\n?([\s\S]*?)```/g, (_, c) => `<pre class="ezag-code">${c}</pre>`);
  s = s.replace(/`([^`\n]+)`/g, "<code>$1</code>");
  s = s.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/^#{1,4} (.+)$/gm, "<strong>$1</strong>");
  s = s.replace(/^(\s*)[-*] (.+)$/gm, "$1• $2");
  return s.replace(/\n/g, "<br>");
}

const nodesOf = (g) => g._nodes || g.nodes || [];

// A downscaled screenshot of the canvas, so the agent can SEE the layout.
function canvasImage() {
  try {
    const src = app.canvas?.canvas;
    if (!src || !src.width) return null;
    const scale = Math.min(1, 1280 / Math.max(src.width, src.height));
    const c = document.createElement("canvas");
    c.width = Math.round(src.width * scale);
    c.height = Math.round(src.height * scale);
    c.getContext("2d").drawImage(src, 0, 0, c.width, c.height);
    return c.toDataURL("image/jpeg", 0.72).split(",")[1];
  } catch { return null; }
}
const linkOf = (g, id) => (g.links?.get ? g.links.get(id) : g.links?.[id]);
const byId = (id) => app.graph.getNodeById(Number(id)) ?? app.graph.getNodeById(id);

// The canvas as compact JSON for the agent.
function snapshot() {
  const g = app.graph;
  const nodes = [];
  for (const n of nodesOf(g)) {
    const widgets = {};
    for (const w of n.widgets || []) {
      if (!w.name || w.type === "button" || w.name === "control_after_generate") continue;
      let v = w.value;
      if (typeof v === "string" && v.length > 600) v = v.slice(0, 600) + "…";
      if (v !== undefined && typeof v !== "object") widgets[w.name] = v;
    }
    const inputs = (n.inputs || []).map((inp) => {
      const l = inp.link != null ? linkOf(g, inp.link) : null;
      return { name: inp.name, type: inp.type, from: l ? [l.origin_id, l.origin_slot] : null };
    });
    nodes.push({ id: n.id, type: n.type, title: n.title !== n.type ? n.title : undefined,
                 mode: n.mode, widgets, inputs });
  }
  return { nodes };
}

function tracker() {
  return app.extensionManager?.workflow?.activeWorkflow?.changeTracker;
}
function captureUndo() {
  const ct = tracker();
  try { (ct?.captureCanvasState || ct?.checkState)?.call(ct); } catch { /* optional */ }
}

function setWidget(node, name, value) {
  const w = (node.widgets || []).find((x) => x.name === name);
  if (!w) throw new Error(`${node.type} has no widget ${name}`);
  w.value = value;
  try { w.callback?.(value, app.canvas, node); } catch { /* widget callbacks are optional */ }
}

// Apply validated ops. Returns the nodes that were added.
function applyOps(ops) {
  const g = app.graph;
  const vmap = {};
  const resolve = (id) => vmap[id] || byId(id);
  const added = [];
  for (const op of ops) {
    if (op.op === "add_node") {
      const n = LiteGraph.createNode(op.type);
      if (!n) throw new Error(`cannot create ${op.type}`);
      if (op.title) n.title = op.title;
      g.add(n);
      for (const [k, v] of Object.entries(op.widgets || {})) setWidget(n, k, v);
      vmap[op.vid] = n;
      added.push(n);
    } else if (op.op === "connect") {
      const s = resolve(op.from.node), d = resolve(op.to.node);
      if (!s || !d) throw new Error("a node in this change no longer exists");
      const slot = d.findInputSlot(op.to.input);
      if (slot < 0) throw new Error(`${d.type} has no input ${op.to.input}`);
      s.connect(op.from.output, d, slot);
    } else if (op.op === "set_widget") {
      setWidget(resolve(op.node), op.name, op.value);
    } else if (op.op === "disconnect") {
      const d = resolve(op.node), slot = d?.findInputSlot(op.input);
      if (slot >= 0) d.disconnectInput(slot);
    } else if (op.op === "remove_node") {
      const n = resolve(op.node);
      if (n) g.remove(n);
    } else if (op.op === "set_title") {
      resolve(op.node).title = op.title;
    } else if (op.op === "bypass") {
      resolve(op.node).mode = op.on ? 4 : 0;
    }
  }
  layout(added, ops, vmap);
  g.setDirtyCanvas(true, true);
  return added;
}

// Place new nodes: dependency columns to the right of existing work, or right
// beside the node they were inserted after.
function layout(added, ops, vmap) {
  if (!added.length) return;
  const others = nodesOf(app.graph).filter((n) => !added.includes(n));
  const depth = new Map(added.map((n) => [n, 0]));
  for (let pass = 0; pass < added.length; pass++) {
    for (const op of ops) {
      if (op.op !== "connect") continue;
      const s = vmap[op.from.node], d = vmap[op.to.node];
      if (s && d) depth.set(d, Math.max(depth.get(d), depth.get(s) + 1));
    }
  }
  if (added.length <= 2 && others.length) {
    for (const n of added) {
      const src = ops.find((o) => o.op === "connect" && vmap[o.to.node] === n && !vmap[o.from.node]);
      const s = src && byId(src.from.node);
      if (s) { n.pos = [s.pos[0] + 40, s.pos[1] + (s.size?.[1] || 100) + 60]; continue; }
    }
    return;
  }
  const x0 = others.length ? Math.max(...others.map((n) => n.pos[0] + (n.size?.[0] || 200))) + 120 : 80;
  const y0 = others.length ? Math.min(...others.map((n) => n.pos[1])) : 80;
  const rows = new Map();
  for (const n of added) {
    const d = depth.get(n), r = rows.get(d) || 0;
    n.pos = [x0 + d * 330, y0 + r];
    rows.set(d, r + (n.size?.[1] || 120) + 40);
  }
}

function focusNodes(nodes) {
  try {
    app.canvas.selectNodes?.(nodes);
    const n = nodes[0];
    if (n && app.canvas.ds) {
      app.canvas.ds.offset = [-n.pos[0] + 120, -n.pos[1] + 120];
      app.canvas.setDirty(true, true);
    }
  } catch { /* cosmetic */ }
}

// -------------------------------------------------------------------- state

const S = {
  root: null, status: null, catalogue: null, chatId: null, runId: null,
  turnEl: null, textEl: null, thinkEl: null, activity: null, buffer: "", images: [],
  view: "chat", afterTurn: [],
};

// ---------------------------------------------------------------- rendering

function render(root) {
  S.root = root;
  root.replaceChildren();
  root.classList.add("ezag");
  const head = el("div", { class: "ezag-head" }, [
    el("div", { class: "ezag-brand" }, [el("span", { class: "ezag-mark", text: "◆" }),
                                        el("span", { text: "Comfy Agent" })]),
    el("div", { class: "ezag-spacer" }),
    S.modelChip = el("button", { class: "ezag-chip", title: "Model (open Inspector)",
                                 onclick: () => show("inspector") }, "…"),
    S.modeBtn = el("button", { class: "ezag-chip ezag-mode", onclick: toggleMode,
      title: "Inspect: read-only. Ask: every change waits for your click. " +
             "Autonomous: builds, runs and reviews by itself; removals still ask." }, "Ask"),
    el("button", { class: "ezag-icon", title: "New chat", onclick: newChat, text: "+" }),
    el("button", { class: "ezag-icon", title: "History", onclick: () => show("history"), text: "☰" }),
    el("button", { class: "ezag-icon", title: "Inspector", onclick: () => show("inspector"), text: "⚙" }),
  ]);
  S.body = el("div", { class: "ezag-body" });
  S.log = el("div", { class: "ezag-log", role: "log", "aria-live": "polite" });
  S.body.append(S.log);
  S.input = el("textarea", { class: "ezag-input", rows: 2,
    placeholder: "Ask anything, or describe what to build…  (Enter to send, Shift+Enter for a new line)" });
  S.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  });
  S.input.addEventListener("paste", (e) => {
    for (const item of e.clipboardData?.items || []) {
      if (item.type.startsWith("image/")) { attach(item.getAsFile()); e.preventDefault(); }
    }
  });
  S.attachRow = el("div", { class: "ezag-attach" });
  const file = el("input", { type: "file", accept: "image/*", style: "display:none",
    onchange: (e) => { for (const f of e.target.files) attach(f); e.target.value = ""; } });
  S.sendBtn = el("button", { class: "ezag-send", onclick: () => (S.runId ? stop() : send()), text: "Send" });
  const composer = el("div", { class: "ezag-composer" }, [
    S.attachRow, S.input,
    el("div", { class: "ezag-row" }, [
      el("button", { class: "ezag-icon", title: "Attach an image", onclick: () => file.click(), text: "📎" }),
      el("button", { class: "ezag-icon", title: "Free VRAM (unload the language model)", onclick: unload, text: "⏏" }),
      file, el("div", { class: "ezag-spacer" }), S.sendBtn]),
  ]);
  composer.addEventListener("dragover", (e) => e.preventDefault());
  composer.addEventListener("drop", (e) => {
    e.preventDefault();
    for (const f of e.dataTransfer.files) if (f.type.startsWith("image/")) attach(f);
  });
  S.mission = el("div", { class: "ezag-mission", hidden: true });
  root.append(head, S.mission, S.body, composer);
  refreshStatus().then(() => (S.chatId ? openChat(S.chatId) : welcome()));
}

function welcome() {
  S.log.replaceChildren();
  const s = S.status;
  const chips = ["Explain this workflow", "Look at my canvas - what's wrong?", "Why did my last run fail?",
                 "Make a photoreal portrait of a lighthouse keeper at dusk",
                 "Add a LoRA to this workflow", "Set 30 steps and cfg 6 on my sampler"];
  S.log.append(el("div", { class: "ezag-welcome" }, [
    el("div", { class: "ezag-hero", text: "A little more possible." }),
    el("div", { class: "ezag-sub", text: s?.recommended?.agent
      ? `Running locally on ${s.recommended.agent.split(":").slice(1).join(":")}. Nothing leaves this PC.`
      : "No local model found yet. Open the Inspector to point me at Ollama or a GGUF folder." }),
    el("div", { class: "ezag-sub", text: MODE_HELP[s?.settings?.permission_mode || "ask"] }),
    el("div", { class: "ezag-chips" }, chips.map((c) => el("button", { class: "ezag-sugg", text: c,
      onclick: () => { S.input.value = c; send(); } }))),
  ]));
}

async function refreshStatus() {
  try {
    S.status = await call("/status");
    const st = S.status.settings;
    const model = st.agent_model || S.status.recommended?.agent || "no model";
    S.modelChip.textContent = model.replace(/^(ollama|gguf|server):/, "").split(/[\\/]/).pop().slice(0, 26);
    S.modeBtn.textContent = MODE_LABEL[st.permission_mode] || "Ask";
    S.modeBtn.dataset.mode = st.permission_mode;
  } catch (e) {
    S.modelChip.textContent = "offline";
    S.status = null;
    toast(`Agent API unreachable: ${e.message}. Restart ComfyUI after installing.`, true);
  }
}

// ------------------------------------------------------- autonomous mission
// A pinned panel showing what the autonomous agent is doing right now:
// goal, phase, cycle, live render progress, elapsed time and a step log.

const PHASES = [["plan", "Plan"], ["build", "Build"], ["run", "Run"], ["review", "Review"]];
const TOOL_PHASE = {
  read_graph: "plan", search_nodes: "plan", node_info: "plan", list_models: "plan", list_workflows: "plan",
  use_skill: "plan", system_status: "plan", check_last_run: "review", look_at_output: "review",
  look_at_canvas: "review", queue_run: "run",
};
const M = { on: false, goal: "", phase: "plan", note: "", pct: 0, t0: 0, steps: [], timer: null, watching: false, end: "", finishedAt: 0, tick: 0, mem: "" };

function missionStart(goal) {
  Object.assign(M, { on: true, goal, phase: "plan", note: "Reading your canvas and request", pct: 0,
                     t0: Date.now(), steps: [], watching: false, end: "" });
  clearInterval(M.timer);
  M.timer = setInterval(() => { if (!(++M.tick % 3)) missionMem(); missionDraw(); }, 1000);
  M.tick = 0;
  missionMem();
  missionDraw();
}

function missionStep(phase, note) {
  if (!M.on) return;
  if (phase) M.phase = phase;
  if (note) {
    M.note = note;
    if (M.steps[M.steps.length - 1]?.text !== note) M.steps.push({ t: Date.now() - M.t0, text: note });
  }
  missionDraw();
}

function missionEnd(reason) {
  if (!M.on) return;
  M.end = reason;
  M.on = false;
  M.watching = false;
  clearInterval(M.timer);
  M.finishedAt = Date.now();
  missionDraw();
}

async function missionMem() {
  try {
    const d = await (await api.fetchApi("/system_stats")).json();
    const v = d.devices?.[0], gb = (n) => (n / 1e9).toFixed(1);
    M.mem = `VRAM ${gb(v.vram_total - v.vram_free)}/${gb(v.vram_total)} GB · RAM ${gb(d.system.ram_free)} GB free`;
  } catch { M.mem = ""; }
}

function missionDraw() {
  const box = S.mission;
  if (!box) return;
  const live = M.on;
  box.hidden = !(live || M.end);
  if (box.hidden) return;
  box.dataset.state = live ? "live" : /limit|stopped|failed|interrupted/i.test(M.end) ? "warn" : "done";
  const secs = Math.round(((live ? Date.now() : M.finishedAt) - M.t0) / 1000);
  const idx = PHASES.findIndex((p) => p[0] === M.phase);
  const stepper = el("div", { class: "ezag-stepper" }, PHASES.map(([k, label], i) => {
    const past = (live || box.dataset.state === "warn") ? i < idx : true;
    return el("div", { class: "ezag-step" + (live && i === idx ? " now" : past ? " past" : "") }, [
      el("span", { class: "ezag-dot", text: past ? "✓" : String(i + 1) }),
      el("span", { text: label })]);
  }));
  const kids = [
    el("div", { class: "ezag-m-top" }, [
      el("span", { class: "ezag-m-badge", text: live ? "● AUTONOMOUS" : "AUTONOMOUS" }),
      el("span", { class: "ezag-m-meta", text: `Cycle ${S.cycle || 0}/${maxCycles()} · ${secs}s` }),
      el("div", { class: "ezag-spacer" }),
      live ? el("button", { class: "ezag-btn ezag-m-stop", text: "Stop", onclick: stop })
           : el("button", { class: "ezag-icon", title: "Dismiss", text: "×", onclick: () => { M.end = ""; missionDraw(); } })]),
    el("div", { class: "ezag-m-goal", title: M.goal, text: M.goal }),
    stepper,
    el("div", { class: "ezag-m-note", text: live ? M.note : M.end }),
    live && M.mem ? el("div", { class: "ezag-m-mem", text: M.mem }) : null,
  ].filter(Boolean);
  if (live && M.phase === "run") {
    kids.push(el("div", { class: "ezag-bar" }, [el("div", { class: "ezag-bar-fill", style: `width:${M.pct}%` })]));
  }
  if (M.steps.length) {
    const log = el("details", { class: "ezag-m-log" }, [
      el("summary", { text: `Step log (${M.steps.length})` }),
      ...M.steps.slice(-30).map((st) => el("div", { class: "ezag-sub", text: `${String(Math.floor(st.t / 1000)).padStart(3, " ")}s  ${st.text}` }))]);
    if (box.querySelector(".ezag-m-log")?.open) log.open = true;
    kids.push(log);
  }
  box.replaceChildren(...kids);
}

const MODE_LABEL = { inspect: "Inspect", ask: "Ask", auto: "Autonomous" };
const MODE_NEXT = { inspect: "ask", ask: "auto", auto: "inspect" };
const MODE_HELP = {
  inspect: "Inspect: read-only. The agent looks, explains and diagnoses, and cannot change anything.",
  ask: "Ask: every change is a card that waits for your click.",
  auto: "Autonomous: the agent builds, runs and reviews by itself (with Undo). Removing or replacing your work still asks.",
};

async function toggleMode() {
  const next = MODE_NEXT[S.status?.settings?.permission_mode || "ask"];
  await call("/settings", { permission_mode: next });
  await refreshStatus();
  toast(MODE_HELP[next]);
}

function show(view) {
  S.view = view;
  if (view === "history") return renderHistory();
  if (view === "inspector") return renderInspector();
  S.body.replaceChildren(S.log);
}

function toast(text, bad = false) {
  const t = el("div", { class: "ezag-toast" + (bad ? " bad" : ""), text });
  S.root?.append(t);
  setTimeout(() => t.remove(), 4500);
}

// -------------------------------------------------------------- attachments

function attach(file) {
  if (!file || S.images.length >= 4) return;
  const reader = new FileReader();
  reader.onload = () => {
    const b64 = String(reader.result).split(",")[1];
    S.images.push(b64);
    const chip = el("div", { class: "ezag-thumb" }, [
      el("img", { src: reader.result, alt: "attachment" }),
      el("button", { text: "×", title: "Remove", onclick: () => {
        S.images.splice(S.images.indexOf(b64), 1); chip.remove(); } })]);
    S.attachRow.append(chip);
  };
  reader.readAsDataURL(file);
}

// -------------------------------------------------------------- chat flow

function newChat() {
  S.chatId = null;
  show("chat");
  welcome();
}

// opts.auto marks a message the autonomous loop wrote, not the user.
async function send(textArg, opts = {}) {
  const text = (textArg ?? S.input.value).trim();
  // Synchronous guard: a double Enter must not fire two requests. runId is
  // only known after /send returns, so it cannot be the guard by itself.
  if (!text || S.runId || S.sending) return;
  S.sending = true;
  if (S.view !== "chat") show("chat");
  if (S.log.querySelector(".ezag-welcome")) S.log.replaceChildren();
  if (!opts.auto) S.input.value = "";
  if (!opts.auto) { S.goal = text; S.cycle = 0; S.autoStopped = false; if (isAuto()) missionStart(text); }
  else missionStep("review", `Cycle ${S.cycle}: reviewing the result against the goal`);
  const images = opts.auto ? [] : S.images.splice(0);
  S.attachRow.replaceChildren();
  S.log.append(el("div", { class: "ezag-user" + (opts.auto ? " auto" : "") }, [
    opts.auto ? el("div", { class: "ezag-sub", text: `Autonomous cycle ${S.cycle}/${maxCycles()}` }) : null,
    el("div", { html: md(text) }),
    images.length ? el("div", { class: "ezag-sub", text: `${images.length} image(s) attached` }) : null]));
  startTurn();
  try {
    const share = S.status?.settings?.share_canvas_view !== false;
    const r = await call("/send", { chat_id: S.chatId, text, snapshot: snapshot(), images,
                                    canvas_image: share ? canvasImage() : null,
                                    client_id: api.clientId ?? api.initialClientId });
    if (!r.ok) throw new Error(r.error);
    S.chatId = r.chat_id;
    S.runId = r.run_id;
    S.sendBtn.textContent = "Stop";
    S.sendBtn.classList.add("stop");
  } catch (e) {
    // Never clear another run's id here: that would orphan a live answer.
    S.turnEl.replaceChildren(el("div", { class: "ezag-error", text: e.message }));
  } finally {
    S.sending = false;
  }
}

async function stop() {
  S.autoStopped = true;                      // also ends an autonomous cycle
  if (S.runId) await call("/stop", { run_id: S.runId }).catch(() => {});
  if (M.watching) await api.interrupt(null).catch(() => {});   // the render the agent queued
  missionEnd("Stopped by you.");
}

const maxCycles = () => S.status?.settings?.max_auto_cycles || 3;
const isAuto = () => S.status?.settings?.permission_mode === "auto";

// The autonomous loop: after a run it hands the result back to the agent.
function autoContinue(succeeded, detail) {
  if (!isAuto() || S.autoStopped || !S.goal) { missionEnd(S.autoStopped ? "Stopped by you." : "Finished."); return; }
  if (S.cycle >= maxCycles()) {
    missionEnd(`Cycle limit reached (${maxCycles()}). Over to you.`);
    S.log.append(el("div", { class: "ezag-sub", text: `Autonomous limit reached (${maxCycles()} cycles). Over to you.` }));
    return;
  }
  S.cycle += 1;
  const text = succeeded
    ? `Autonomous review. The goal was: "${S.goal}". Look at the result of the last run with ` +
      `look_at_output and ask the vision model to compare it with the goal point by point. First name ` +
      `the single most visible way it falls short of the goal (there is almost always one). If that ` +
      `gap is minor, say the result is good enough and stop. If it clearly misses part of the goal, ` +
      `make ONE specific change (usually to the prompt text) and call queue_run again.`
    : `Autonomous fix. The goal was: "${S.goal}". The last run failed: ${detail}. Use check_last_run, ` +
      `fix the cause with the smallest change, then call queue_run again.`;
  setTimeout(() => send(text, { auto: true }), 400);
}

async function unload() {
  const r = await call("/unload", {});
  const freed = [r.freed?.llama_server && "llama-server", ...(r.freed?.ollama || [])].filter(Boolean);
  toast(freed.length ? `Unloaded ${freed.join(", ")}` : "No language model was loaded");
}

function startTurn() {
  S.turnEl = el("div", { class: "ezag-turn" });
  S.activity = null;
  S.thinkEl = null;
  S.textEl = el("div", { class: "ezag-text" }, [el("span", { class: "ezag-dots", text: "thinking" })]);
  S.buffer = "";
  S.turnEl.append(S.textEl);
  S.log.append(S.turnEl);
  scroll();
}

function endTurn() {
  S.runId = null;
  S.sendBtn.textContent = "Send";
  S.sendBtn.classList.remove("stop");
}

function scroll() {
  S.log.scrollTop = S.log.scrollHeight;
}

function activityBlock() {
  if (!S.activity) {
    const list = el("div", { class: "ezag-acts" });
    const sum = el("summary", { text: "Working…" });
    S.activity = { box: el("details", { class: "ezag-activity" }, [sum, list]), list, sum, n: 0, ms: 0 };
    S.turnEl.insertBefore(S.activity.box, S.textEl);
  }
  return S.activity;
}

const TOOL_LABEL = {
  read_graph: "Read the canvas", search_nodes: "Searched nodes", node_info: "Checked a node",
  list_models: "Listed models", list_workflows: "Listed workflows", check_last_run: "Checked the last run",
  look_at_output: "Looked at an image", system_status: "Checked the GPU", use_skill: "Loaded a skill",
  remember: "Saved a preference", set_widgets: "Drafted value changes", add_node: "Drafted a node",
  connect: "Drafted a link", disconnect: "Drafted an unlink", remove_node: "Drafted a removal",
  bypass_node: "Drafted a bypass", insert_after: "Drafted an insertion", open_workflow: "Prepared to open a workflow",
  queue_run: "Prepared a run", look_at_canvas: "Looked at the canvas", save_workflow: "Prepared a save",
};

function onEvent(ev) {
  if (!S.turnEl || ev.run_id !== S.runId) return;
  if (ev.type === "delta") {
    if (ev.kind === "thinking") {
      if (!S.thinkEl) {
        S.thinkEl = el("pre", { class: "ezag-think" });
        S.turnEl.insertBefore(el("details", { class: "ezag-activity" },
          [el("summary", { text: "Reasoning" }), S.thinkEl]), S.textEl);
      }
      S.thinkEl.textContent += ev.text;
    } else {
      S.buffer += ev.text;
      S.textEl.innerHTML = md(S.buffer);
    }
    scroll();
  } else if (ev.type === "turn_break") {
    S.buffer = "";                               // narration before tool calls is not the answer
    S.textEl.replaceChildren(el("span", { class: "ezag-dots", text: "working" }));
  } else if (ev.type === "tool") {
    const a = activityBlock();
    a.n += 1;
    a.ms += ev.ms || 0;
    const failed = /"ok":\s*false|"errors"/.test(ev.result || "");
    a.list.append(el("details", { class: "ezag-act" + (failed ? " bad" : "") }, [
      el("summary", {}, [el("span", { class: "ezag-k " + ev.kind, text: ev.kind === "look" ? "LOOK" : "DO" }),
                         el("span", { text: ` ${TOOL_LABEL[ev.name] || ev.name}` }),
                         el("span", { class: "ezag-ms", text: ` ${ev.ms} ms` })]),
      el("pre", { text: `${ev.name}(${JSON.stringify(ev.args)})\n→ ${ev.result}` })]));
    if (M.on) missionStep(TOOL_PHASE[ev.name] || "build", TOOL_LABEL[ev.name] || ev.name);
    a.sum.textContent = `${a.n} step${a.n > 1 ? "s" : ""} · ${(a.ms / 1000).toFixed(1)} s`;
  } else if (ev.type === "route" && ev.step === "claim_check") {
    activityBlock().list.append(el("div", { class: "ezag-sub", text: "Checked a claimed action: no tool had run, asked the model to act or correct itself" }));
  } else if (ev.type === "route" && (ev.step === "limit" || ev.step === "stuck")) {
    activityBlock().list.append(el("div", { class: "ezag-sub", text: `Stopped looking (${ev.detail}); summarising` }));
  } else if (ev.type === "route" && ev.step === "vision") {
    activityBlock().list.append(el("div", { class: "ezag-sub", text: `Image handed to ${ev.model} (vision)` }));
  } else if (ev.type === "proposal") {
    S.turnEl.append(card(ev.card));
    scroll();
  } else if (ev.type === "done") {
    S.textEl.innerHTML = md(ev.result.text);
    const r = ev.result;
    if (r.unverified_claim) {
      S.turnEl.append(el("div", { class: "ezag-error", text:
        "⚠ No action was taken: the reply describes a change, but the agent did not call any tool. " +
        "Nothing on your canvas or disk changed." }));
    }
    const pending = r.proposals.filter((p) => !p.auto_apply).length;
    S.turnEl.append(el("div", { class: "ezag-meta", text:
      `${r.model} · ${r.rounds} round${r.rounds > 1 ? "s" : ""} · ${r.elapsed} s` +
      (pending ? ` · ${pending} change${pending > 1 ? "s" : ""} waiting for you` : "") }));
    S.lastManifest = r.manifest;
    if (M.on && !M.watching) {
      missionEnd(r.proposals.some((p) => !p.auto_apply) ? "Waiting for you: a change needs your click." : "Finished.");
    }
    endTurn();
    S.afterTurn.splice(0).forEach((f) => f());
    scroll();
  } else if (ev.type === "stopped") {
    S.textEl.append(el("div", { class: "ezag-sub", text: "Stopped." }));
    S.afterTurn.length = 0;
    missionEnd("Stopped by you.");
    endTurn();
  } else if (ev.type === "error") {
    S.textEl.replaceChildren(el("div", { class: "ezag-error", text: ev.message }));
    S.afterTurn.length = 0;
    missionEnd("Failed: " + String(ev.message).slice(0, 120));
    endTurn();
  }
}

// ------------------------------------------------------------------ cards

const RISK_TEXT = { edit: "edit", queue: "run", destructive: "removes work", replace: "replaces canvas" };

function card(c) {
  const box = el("div", { class: `ezag-card risk-${c.risk}` });
  const status = el("div", { class: "ezag-sub", text: c.policy || "" });
  const actions = el("div", { class: "ezag-row" });
  const head = el("div", { class: "ezag-card-head" }, [
    el("span", { class: "ezag-card-kind", text: { graph_edit: "Change", queue: "Run", open_workflow: "Open", save_workflow: "Save" }[c.kind] || c.kind }),
    el("span", { class: "ezag-card-title", text: c.summary }),
    el("span", { class: `ezag-risk ${c.risk}`, text: RISK_TEXT[c.risk] || c.risk })]);
  box.append(head);
  if (c.diff?.length) {
    const diff = el("div", { class: "ezag-diff" }, c.diff.map((d) => el("div", {
      class: "d-" + ({ "+": "add", "~": "set", "→": "link", "−": "del", "✕": "del" }[d[0]] || "other"), text: d })));
    box.append(c.diff.length > 8 ? el("details", {}, [el("summary", { text: `${c.diff.length} changes` }), diff]) : diff);
  }
  if (c.warnings?.length) {
    box.append(el("div", { class: "ezag-warn" }, c.warnings.map((w) => el("div", { text: "⚠ " + w }))));
  }
  box.append(status, actions);

  let before = null;
  let applied = [];
  const doApply = async () => {
    try {
      if (c.kind === "graph_edit") {
        const check = await call("/revalidate", { ops: c.ops, snapshot: snapshot() });
        if (!check.ok) {
          status.textContent = "The canvas changed since this was proposed: " + check.errors.join("; ");
          status.className = "ezag-error";
          return;
        }
        before = JSON.parse(JSON.stringify(app.graph.serialize()));
        applied = applyOps(c.ops);
        captureUndo();
        focusNodes(applied.length ? applied : []);
        status.textContent = c.auto_apply ? "Applied automatically · Ctrl+Z or Undo reverts" : "Applied · Ctrl+Z or Undo reverts";
        status.className = "ezag-ok";
        actions.replaceChildren(el("button", { class: "ezag-btn", text: "Undo", onclick: undo }));
      } else if (c.kind === "open_workflow") {
        const res = await api.fetchApi(`/userdata/${encodeURIComponent("workflows/" + c.name)}`);
        if (!res.ok) throw new Error(`could not read ${c.name}`);
        await app.loadGraphData(await res.json(), true, true, c.name);
        status.textContent = `Opened ${c.name}`;
        status.className = "ezag-ok";
        actions.replaceChildren();
      } else if (c.kind === "queue") {
        status.textContent = "Freeing VRAM…";
        await call("/prepare_queue", {}).catch(() => {});
        status.textContent = "Queued";
        await app.queuePrompt(0, c.count || 1);
        watchRun(box, status, !!c.auto_apply);
        actions.replaceChildren();
      } else if (c.kind === "save_workflow") {
        const r = await call("/save_workflow", { name: c.name, overwrite: !!c.overwrite,
                                                 graph: app.graph.serialize() });
        if (!r.ok) throw new Error(r.error);
        status.textContent = `Saved to your workflows as ${r.name}`;
        status.className = "ezag-ok";
        actions.replaceChildren();
      }
    } catch (e) {
      status.textContent = e.message;
      status.className = "ezag-error";
    }
  };
  const undo = () => {
    try {
      app.graph.configure(before);
      app.graph.setDirtyCanvas(true, true);
      captureUndo();
      status.textContent = "Undone";
      status.className = "ezag-sub";
      actions.replaceChildren(el("button", { class: "ezag-btn primary", text: "Apply again", onclick: doApply }));
    } catch (e) {
      status.textContent = `Undo failed: ${e.message}. Use Ctrl+Z.`;
    }
  };
  if (c.auto_apply && c.kind === "queue") {
    // Wait for the agent's own reply to finish: otherwise the language model reloads
    // into VRAM while the render is starting and both crawl.
    S.afterTurn.push(doApply);
    M.watching = true;
    status.textContent = "Will start when the agent finishes";
  } else if (c.auto_apply) {
    setTimeout(doApply, 0);
  } else {
    actions.append(
      el("button", { class: "ezag-btn primary", text: { queue: "Run", open_workflow: "Open", save_workflow: c.overwrite ? "Overwrite" : "Save" }[c.kind] || "Apply",
                     onclick: doApply }),
      el("button", { class: "ezag-btn", text: "Dismiss", onclick: () => {
        status.textContent = "Dismissed"; actions.replaceChildren(); } }));
    if (c.kind === "graph_edit") {
      const touched = [...new Set(c.ops.flatMap((o) => [o.node, o.to?.node, o.from?.node]))]
        .map((id) => id && byId(id)).filter(Boolean);
      if (touched.length) actions.append(el("button", { class: "ezag-btn", text: "Show", onclick: () => focusNodes(touched) }));
    }
  }
  return box;
}

// Follow a queued run and show its images inline.
function watchRun(box, status, autonomous = false) {
  const gallery = el("div", { class: "ezag-gallery" });
  box.append(gallery);
  const off = [];
  const on = (name, fn) => { api.addEventListener(name, fn); off.push(() => api.removeEventListener(name, fn)); };
  const done = () => {
    if (autonomous) M.watching = false;
    off.forEach((f) => f());
    call("/free_image_models", {}).catch(() => {});   // hand VRAM and RAM back
  };
  if (autonomous) { M.watching = true; missionStep("run", "Rendering…"); }
  on("progress", (e) => {
    status.textContent = `Rendering ${e.detail.value}/${e.detail.max}`;
    if (autonomous) { M.pct = Math.round(100 * e.detail.value / e.detail.max); missionStep("run", `Rendering ${e.detail.value}/${e.detail.max} steps`); }
  });
  on("executed", (e) => {
    for (const img of e.detail?.output?.images || []) {
      const src = api.apiURL(`/view?filename=${encodeURIComponent(img.filename)}&subfolder=${encodeURIComponent(img.subfolder || "")}&type=${img.type}`);
      gallery.append(el("img", { src, alt: img.filename, title: img.filename,
                                 onclick: () => window.open(src, "_blank") }));
    }
  });
  on("execution_success", () => {
    status.textContent = "Finished";
    status.className = "ezag-ok";
    if (autonomous) { missionStep("review", "Render finished: image models offloaded, handing the image to the reviewer"); autoContinue(true); }
    else box.append(el("div", { class: "ezag-row" }, [el("button", { class: "ezag-btn", text: "Ask the agent to review it",
      onclick: () => send("Look at the result of my last run and tell me honestly how it turned out.") })]));
    done();
  });
  on("execution_error", (e) => {
    status.textContent = `Failed in ${e.detail?.node_type}: ${String(e.detail?.exception_message || "").slice(0, 160)}`;
    status.className = "ezag-error";
    if (autonomous) { missionStep("run", "Run failed, asking the agent to fix it"); autoContinue(false, `${e.detail?.node_type}: ${String(e.detail?.exception_message || "").slice(0, 200)}`); }
    else box.append(el("div", { class: "ezag-row" }, [el("button", { class: "ezag-btn primary", text: "Ask why it failed",
      onclick: () => send("My last run failed. Why, and what is the smallest fix?") })]));
    done();
  });
  on("execution_interrupted", () => { status.textContent = "Interrupted"; if (autonomous) missionEnd("Run interrupted."); done(); });
}

// ----------------------------------------------------------------- history

async function openChat(id) {
  const r = await call("/chat", { id });
  S.chatId = id;
  show("chat");
  S.log.replaceChildren();
  for (const t of r.chat.turns) {
    S.log.append(el("div", { class: "ezag-user", html: md(t.user) }));
    const turn = el("div", { class: "ezag-turn" }, [el("div", { class: "ezag-text", html: md(t.assistant) })]);
    for (const p of t.proposals || []) {
      turn.append(el("div", { class: "ezag-card past" }, [
        el("div", { class: "ezag-card-head" }, [el("span", { class: "ezag-card-title", text: p.summary }),
                                               el("span", { class: "ezag-sub", text: " (earlier card)" })])]));
    }
    turn.append(el("div", { class: "ezag-meta", text: `${t.model} · ${t.elapsed ?? "?"} s` }));
    S.log.append(turn);
  }
  if (!r.chat.turns.length) welcome();
  scroll();
}

async function renderHistory() {
  const r = await call("/chats");
  const list = el("div", { class: "ezag-panel" }, [el("div", { class: "ezag-h", text: "Chats" })]);
  if (!r.chats.length) list.append(el("div", { class: "ezag-sub", text: "No chats yet." }));
  for (const c of r.chats) {
    list.append(el("div", { class: "ezag-hist" }, [
      el("button", { class: "ezag-hist-open", onclick: () => openChat(c.id) }, [
        el("div", { text: c.title }),
        el("div", { class: "ezag-sub", text: `${c.turns} turn(s) · ${new Date(c.updated * 1000).toLocaleString()}` })]),
      el("button", { class: "ezag-icon", title: "Delete", text: "🗑", onclick: async () => {
        await call("/chat/delete", { id: c.id }); renderHistory(); } })]));
  }
  list.append(el("button", { class: "ezag-btn", text: "Back", onclick: () => show("chat") }));
  S.body.replaceChildren(list);
}

// --------------------------------------------------------------- inspector

async function renderInspector() {
  const panel = el("div", { class: "ezag-panel" }, [el("div", { class: "ezag-sub", text: "Loading models…" })]);
  S.body.replaceChildren(panel);
  const [cat, st, mem] = await Promise.all([call("/models"), call("/status"), call("/memory")]);
  const s = st.settings;
  const caps = (m) => ["tools", "vision", "thinking", "audio"].filter((k) => m.caps[k])
    .map((k) => el("span", { class: "ezag-cap", text: k }));
  const option = (m) => el("option", { value: m.ref,
    text: `${m.name}  ·  ${m.provider}${m.size_gb ? " · " + m.size_gb + " GB" : ""}` });
  const agentSel = el("select", {}, [el("option", { value: "", text: `Auto (${(cat.recommended.agent || "none").replace(/^\w+:/, "")})` }),
    ...cat.models.filter((m) => m.caps.tools).map(option)]);
  agentSel.value = s.agent_model;
  const visionSel = el("select", {}, [el("option", { value: "", text: `Auto (${(cat.recommended.vision || "none").replace(/^\w+:/, "")})` }),
    ...cat.models.filter((m) => m.caps.vision).map(option)]);
  visionSel.value = s.vision_model;
  const field = (label, input, hint) => el("label", { class: "ezag-field" }, [
    el("span", { text: label }), input, hint ? el("span", { class: "ezag-sub", text: hint }) : null]);
  const txt = (v) => el("input", { value: v ?? "" });
  const num = (v, min, max) => el("input", { type: "number", value: v, min, max });
  const chk = (v) => { const c = el("input", { type: "checkbox" }); c.checked = !!v; return c; };
  const f = {
    ollama_url: txt(s.ollama_url), llama_server_exe: txt(s.llama_server_exe),
    external_llama_url: txt(s.external_llama_url), extra_model_dirs: el("textarea", { rows: 2 }),
    max_rounds: num(s.max_rounds, 1, 20), context_tokens: num(s.context_tokens, 2048, 262144),
    think: chk(s.think), free_vram_before_queue: chk(s.free_vram_before_queue), offload_between_steps: chk(s.offload_between_steps),
    allow_remote_hosts: chk(s.allow_remote_hosts),
    max_auto_cycles: num(s.max_auto_cycles, 1, 8), share_canvas_view: chk(s.share_canvas_view),
  };
  f.extra_model_dirs.value = (s.extra_model_dirs || []).join("\n");
  const save = async () => {
    await call("/settings", {
      agent_model: agentSel.value, vision_model: visionSel.value, ollama_url: f.ollama_url.value,
      llama_server_exe: f.llama_server_exe.value, external_llama_url: f.external_llama_url.value,
      extra_model_dirs: f.extra_model_dirs.value.split("\n"), max_rounds: f.max_rounds.value,
      context_tokens: f.context_tokens.value, think: f.think.checked,
      free_vram_before_queue: f.free_vram_before_queue.checked, offload_between_steps: f.offload_between_steps.checked, allow_remote_hosts: f.allow_remote_hosts.checked,
      max_auto_cycles: f.max_auto_cycles.value, share_canvas_view: f.share_canvas_view.checked,
    });
    await refreshStatus();
    toast("Saved");
  };
  const models = el("div", { class: "ezag-models" }, cat.models.map((m) => el("div", { class: "ezag-model" }, [
    el("div", { class: "ezag-model-name", text: m.name, title: m.path || m.source }),
    el("div", { class: "ezag-row" }, [el("span", { class: "ezag-prov", text: m.provider }), ...caps(m),
      m.mmproj ? el("span", { class: "ezag-cap", text: "projector" }) : null,
      m.provider === "llama.cpp" && !m.runnable ? el("span", { class: "ezag-cap bad", text: "no llama-server" }) : null])])));
  const manifest = S.lastManifest ? el("pre", { class: "ezag-code", text: S.lastManifest.map((p) =>
    `${p.part.padEnd(9)} ${p.chars ?? ""}${p.dropped ? `  (trimmed ${p.dropped})` : ""}${p.turns != null ? `turns ${p.turns}` : ""}`).join("\n") }) : null;
  panel.replaceChildren(
    el("div", { class: "ezag-h", text: "Brains" }),
    field("Agent model (needs tools)", agentSel, "Spark X2.5 is the fastest tool-trained model here; Qwen 3.5 9B is smarter."),
    field("Vision model", visionSel, "Used to look at images when the agent model cannot see."),
    el("div", { class: "ezag-h", text: "Runtime" }),
    el("div", { class: "ezag-sub", text: st.runtime.resident ? `llama-server running ${st.runtime.resident.split(/[\\/]/).pop()}` : "No GGUF model resident." }),
    el("div", { class: "ezag-sub", text: cat.llama_server ? `llama-server: ${cat.llama_server}` : "llama-server not found (install Ollama, or set its path)." }),
    cat.ollama_error ? el("div", { class: "ezag-error", text: cat.ollama_error }) : null,
    field("Ollama URL", f.ollama_url), field("llama-server.exe (blank = Ollama's)", f.llama_server_exe),
    field("Running OpenAI-compatible server URL (optional)", f.external_llama_url),
    field("Extra GGUF folders (one per line)", f.extra_model_dirs, `Also scanned: ${cat.dirs.join(" · ")}`),
    el("div", { class: "ezag-h", text: "Behaviour" }),
    field("Autonomous cycles per request", f.max_auto_cycles, "Build → run → review → improve, at most this many times."),
    field("Let the agent see the canvas", f.share_canvas_view, "Sends a screenshot of the canvas with each message (stays on this PC)."),
    field("Max tool rounds", f.max_rounds), field("Context tokens", f.context_tokens),
    field("Show reasoning (slower)", f.think), field("Free VRAM before every run", f.free_vram_before_queue),
    field("Offload between steps", f.offload_between_steps, "After a run, unload image models and clear ComfyUI's cache; the vision model leaves memory as soon as it has answered."),
    field("Allow model servers on other machines", f.allow_remote_hosts, "Off = local-first: your data never leaves this PC."),
    el("div", { class: "ezag-row" }, [el("button", { class: "ezag-btn primary", text: "Save", onclick: save }),
      el("button", { class: "ezag-btn", text: "Back", onclick: () => show("chat") })]),
    el("div", { class: "ezag-h", text: `Models found (${cat.models.length})` }), models,
    el("div", { class: "ezag-h", text: "Remembered" }),
    mem.notes.length ? el("div", {}, mem.notes.map((n) => el("div", { class: "ezag-hist" }, [
      el("span", { text: n.note }), el("button", { class: "ezag-icon", text: "×", title: "Forget",
        onclick: async () => { await call("/memory", { note: n.note, forget: true }); renderInspector(); } })])))
      : el("div", { class: "ezag-sub", text: "Nothing yet. Tell the agent a preference and it will remember it." }),
    el("div", { class: "ezag-h", text: "Last request context" }),
    manifest || el("div", { class: "ezag-sub", text: "Send a message to see what went into the prompt." }),
  );
}

// ---------------------------------------------------------------- register

app.registerExtension({
  name: "EzeAi.ComfyAgent",
  async setup() {
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = new URL("./comfy_agent.css", import.meta.url).href;
    document.head.append(link);
    api.addEventListener("ezeai.agent", (e) => onEvent(e.detail));
    app.extensionManager.registerSidebarTab({
      id: TAB, icon: "pi pi-sparkles", title: "EzeAi Agent", tooltip: "EzeAi Comfy Agent (Ctrl+Shift+A)",
      type: "custom", render: (elm) => render(elm),
    });
  },
  commands: [{ id: "EzeAi.ComfyAgent.toggle", label: "Toggle EzeAi Comfy Agent", icon: "pi pi-sparkles",
               function: () => app.extensionManager.sidebarTab?.toggleSidebarTab?.(TAB) }],
  keybindings: [{ combo: { key: "a", ctrl: true, shift: true }, commandId: "EzeAi.ComfyAgent.toggle" }],
  getNodeMenuItems(node) {
    return [null, {
      content: "Ask EzeAi Agent about this node",
      callback: () => {
        app.extensionManager.sidebarTab?.toggleSidebarTab?.(TAB);
        setTimeout(() => {
          if (S.input) { S.input.value = `Explain node #${node.id} (${node.type}) and suggest improvements.`; S.input.focus(); }
        }, 200);
      },
    }];
  },
});
