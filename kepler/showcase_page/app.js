// Kepler Lab Replay. Static page: real clips + the real analysis, drawn and typeset live.
// Everything shown at video time t comes from data/<trial>.json; the science panel only
// uses snapshots of cycles completed before t (precomputed in Python, see kepler/showcase.py).
"use strict";

// Each law as the analysis uses it: the equation of motion it comes from, the period or envelope
// it predicts, and (for the exact law) how K is evaluated. The fitted forms are the ones in
// kepler/pendulum.py; nothing here is decoration.
const PERIOD_LAWS = [
  { id: "H1", name: "isochronous (small-angle)", label: "small-angle",
    tex: String.raw`\ddot{\theta} + \frac{g}{L}\,\theta = 0
      \;\;\Longrightarrow\;\; T = 2\pi\sqrt{\frac{L}{g}}\quad(\text{independent of }\theta_0)` },
  { id: "H2", name: "quadratic correction", label: "series",
    tex: String.raw`\begin{aligned}
      T &= 2\pi\sqrt{\frac{L}{g}}\;\sum_{n=0}^{\infty}\left[\frac{(2n)!}{2^{2n}\,(n!)^{2}}\right]^{2}\sin^{2n}\!\frac{\theta_0}{2}
         = 2\pi\sqrt{\frac{L}{g}}\left(1+\frac{\theta_0^{2}}{16}+\frac{11\,\theta_0^{4}}{3072}+\cdots\right)\\
      T &\approx 2\pi\sqrt{\frac{L}{g}}\left(1+\frac{\theta_0^{2}}{16}\right) \qquad \text{(fitted: first order in } \theta_0^{2}\text{)}
    \end{aligned}` },
  { id: "H3", name: "exact (elliptic integral)", label: "exact",
    tex: String.raw`\begin{aligned}
      \ddot{\theta} + \frac{g}{L}\sin\theta = 0 \;\;\Longrightarrow\;\; T &= 4\sqrt{\frac{L}{g}}\;K(m),
        \qquad m = \sin^{2}\frac{\theta_0}{2}\\
      K(m) &= \int_{0}^{\pi/2}\frac{d\varphi}{\sqrt{1-m\sin^{2}\varphi}}
            = \frac{\pi}{2\,\operatorname{AGM}\!\left(1,\sqrt{1-m}\right)}\\
      \operatorname{AGM}:\;\; a_{n+1} &= \frac{a_n+b_n}{2},\qquad b_{n+1} = \sqrt{a_n\,b_n},\qquad a_0 = 1,\; b_0 = \sqrt{1-m}
    \end{aligned}` },
  { id: "H4", name: "free quadratic", label: "free coefficient",
    tex: String.raw`T = 2\pi\sqrt{\frac{L}{g}}\left(1 + c\,\theta_0^{2}\right),
      \qquad c \text{ fitted freely}\quad\left(\text{exact law: } c = \tfrac{1}{16} = 0.0625\right)` },
];
const DAMP_LAWS = [
  { id: "D1", name: "linear (viscous) damping", label: "viscous",
    tex: String.raw`\frac{dA}{dt} = -a\,A \;\;\Longrightarrow\;\; A(t) = A_0\,e^{-a t}` },
  { id: "D2", name: "quadratic (air drag) damping", label: "air drag",
    tex: String.raw`\frac{dA}{dt} = -b\,A^{2} \;\;\Longrightarrow\;\; A(t) = \frac{A_0}{1 + b\,A_0\,t}` },
  { id: "D3", name: "dry (Coulomb) friction", label: "Coulomb",
    tex: String.raw`\frac{dA}{dt} = -c \;\;\Longrightarrow\;\; A(t) = A_0 - c\,t` },
  { id: "D4", name: "viscous + air drag", label: "combined",
    tex: String.raw`\frac{dA}{dt} = -a\,A - b\,A^{2} \;\;\Longrightarrow\;\;
      A(t) = \frac{a\,A_0\,e^{-a t}}{a + b\,A_0\left(1 - e^{-a t}\right)}` },
];
const BIC_TEX = String.raw`\mathrm{BIC}_i = n\,\ln\frac{\mathrm{RSS}_i}{n} + k_i\ln n,
  \qquad \Delta\mathrm{BIC}_i = \mathrm{BIC}_i - \min_j \mathrm{BIC}_j
  \qquad \bigl(\le 2\ \text{leading},\; > 10\ \text{ruled out}\bigr)`;
const LAW_BY_NAME = Object.fromEntries([...PERIOD_LAWS, ...DAMP_LAWS].map((l) => [l.name, l]));
const LAYERS = { mask: true, trail: true, pivot: true, string: true, angle: true };

let TRIALS = [], INDEX = null, G_LOCAL = 9.79, MIN_EV = 5;
let tMaster = 0, playing = false, rate = 1, lastNow = 0, duration = 1;
let hl = { law: null, trial: null, layer: null, until: 0 }, hoverLaw = null;

// ---------- helpers ----------
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const f = (x, d) => Number(x).toFixed(d);
function status(d) { return d === null || d === undefined ? "wait" : d <= 2 ? "leading" : d > 10 ? "ruled" : "alive"; }
const STATUS_TEXT = { leading: "leading", alive: "alive", ruled: "ruled out", wait: "waiting for cycles" };
function tex(el, s, display = true) {
  if (el._tex === s) return;
  el._tex = s;
  katex.render(s, el, { displayMode: display, throwOnError: false, strict: "ignore" });
}
function lastLE(arr, t, key = (x) => x) {   // index of the last item with key <= t, or -1
  let lo = 0, hi = arr.length - 1, ans = -1;
  while (lo <= hi) { const m = (lo + hi) >> 1; if (key(arr[m]) <= t) { ans = m; lo = m + 1; } else hi = m - 1; }
  return ans;
}
function ellipK(m) {  // complete elliptic integral of the first kind, parameter m = k^2 (AGM)
  let a = 1, b = Math.sqrt(1 - m);
  for (let i = 0; i < 20 && Math.abs(a - b) > 1e-15; i++) { const an = (a + b) / 2; b = Math.sqrt(a * b); a = an; }
  return Math.PI / (2 * a);
}
function periodCurve(law, p, L, A) {   // A in rad; parameters from the fit (same models as kepler/pendulum.py)
  const T0 = 2 * Math.PI * Math.sqrt(L / p.g);
  if (law === "H1") return T0;
  if (law === "H2") return T0 * (1 + A * A / 16);
  if (law === "H3") return 4 * Math.sqrt(L / p.g) * ellipK(Math.sin(A / 2) ** 2);
  if (law === "H4") return T0 * (1 + p.c * A * A);
}
function trialTime(T) { return T.video ? Math.min(T.video.currentTime, T.duration) : 0; }
function snapAt(T, t) { const i = lastLE(T.snapshots, t + 1e-6, (s) => s.t); return i < 0 ? null : T.snapshots[i]; }
function frameAt(T, t) {
  const i = lastLE(T.frames, t + 0.5 / T.fps, (fr) => fr.t);
  return i < 0 ? null : T.frames[i];
}

// ---------- build the page ----------
async function load() {
  INDEX = await (await fetch("index.json")).json();
  G_LOCAL = INDEX.g_local;
  MIN_EV = INDEX.min_evidence_cycles || 5;
  TRIALS = await Promise.all(INDEX.trials.map(async (id) => (await fetch(`${id}.json`)).json()));
  duration = Math.max(...TRIALS.map((T) => T.duration));
  $("scrub").max = duration.toFixed(2);
  TRIALS.forEach(buildVideo);
  if (TRIALS.length === 1) { $("videos").classList.add("single"); $("plots").classList.add("single"); }
  buildHyps("hyps", PERIOD_LAWS, "period_laws");
  buildHyps("damps", DAMP_LAWS, "damping_laws");
  buildPlots();
  if (TRIALS.length === 1) $("plots").classList.add("single");
  buildStatements();
  tex($("m-def"), String.raw`\theta = \operatorname{atan2}\!\left(x - x_p,\; y - y_p\right) - \theta_{\mathrm{eq}}`);
  tex($("bic-def"), BIC_TEX);
  tex($("g-def"), String.raw`g = \frac{4\pi^{2} L_{\mathrm{eff}}}{T_0^{2}}, \qquad
      L_{\mathrm{eff}} = L + \frac{2r^{2}}{5L}, \qquad
      \sigma_g = g\sqrt{\left(\frac{\sigma_{\mathrm{fit}}}{g}\right)^{2} + \left(\frac{\sigma_L}{L}\right)^{2}}`);
  bindControls();
  requestAnimationFrame(loop);
  const m = location.hash.match(/t=([\d.]+)/);
  if (m) await testSeek(parseFloat(m[1]));
  const sc = location.hash.match(/scrub=([\d.,]+)/);   // test: scrub through several positions quickly
  if (sc) {
    for (const t of sc[1].split(",").map(Number)) { seekAll(t); await new Promise((r) => setTimeout(r, 120)); }
    await new Promise((r) => setTimeout(r, 1500));
    document.body.dataset.scrubResult = `master=${tMaster.toFixed(3)} ` +
      TRIALS.map((T) => `${T.id}=${T.video.currentTime.toFixed(3)}`).join(" ");
  }
}

function buildVideo(T) {
  const card = document.createElement("div");
  card.className = "vcard";
  card.innerHTML = `
    <div class="vhead"><span class="dot" style="background:${T.colour}"></span><h3>${esc(T.title)}</h3>
      <span class="badge flag">APPARATUS FLAGGED</span></div>
    <div class="stage" style="aspect-ratio:${T.frame_size[0]} / ${T.frame_size[1]}"><video muted playsinline preload="auto" src="${T.clip}"></video><canvas></canvas></div>
    <div class="readout"><span class="L"></span><span class="g"></span></div>
    <div class="note">${esc(T.setup)}. Tracked by ${esc(T.tracker)} (“${esc(T.phrase)}”).
      ${T.camera_corrected ? `Track corrected for camera motion (the phone moved ${T.camera_motion_px[0]} × ${T.camera_motion_px[1]} px).` : "No camera-motion correction (it made the timing worse)."}</div>`;
  $("videos").appendChild(card);
  T.video = card.querySelector("video");
  T.canvas = card.querySelector("canvas");
  T.elL = card.querySelector(".L");
  T.elG = card.querySelector(".g");
  tex(T.elL, String.raw`L = ${f(T.L, 3)}\ \mathrm{m}`, false);
  T.video.addEventListener("ended", () => T.video.pause());
}

function buildHyps(container, laws, key) {
  const el = $(container);
  for (const law of laws) {
    const card = document.createElement("div");
    card.className = "hyp";
    card.dataset.id = law.id;
    card.innerHTML = `<div class="hname"><span class="id">${law.id}</span>${esc(law.label)}</div>
      <div class="badges"></div><div class="eq"></div>`;
    tex(card.querySelector(".eq"), law.tex);
    card.addEventListener("mouseenter", () => { hoverLaw = law.id; });
    card.addEventListener("mouseleave", () => { hoverLaw = null; });
    el.appendChild(card);
    law.card = card;
    law.badges = card.querySelector(".badges");
    law.key = key;
  }
}

function buildPlots() {
  for (const T of TRIALS) {
    const d = document.createElement("div");
    d.className = "plot";
    d.innerHTML = `<canvas></canvas><div class="cap"><span style="color:${T.colour}">●</span> ${esc(T.short)}: period (ms) against amplitude (°)</div>`;
    $("plots").appendChild(d);
    T.plot = d.querySelector("canvas");
  }
}

function buildStatements() {
  for (const s of INDEX.statements) {
    const T = TRIALS.find((x) => x.id === s.trial);
    const b = document.createElement("button");
    b.className = "stmt";
    b.style.borderLeftColor = T.colour;
    b.innerHTML = `${esc(s.text)}<span class="when">at ${f(s.t, 1)} s</span>`;
    b.addEventListener("click", () => {
      seekAll(s.t + 0.05);
      hl = { law: s.law, trial: s.trial, layer: s.overlay, diag: s.diagnosis, until: performance.now() + 4500 };
      if (s.law) { const law = LAW_BY_NAME[[...PERIOD_LAWS, ...DAMP_LAWS].find((l) => l.id === s.law).name];
        law.card.scrollIntoView({ block: "nearest", behavior: "smooth" }); }
    });
    $("statements").appendChild(b);
  }
}

// ---------- transport (one shared timeline) ----------
function bindControls() {
  $("play").addEventListener("click", () => (playing ? pause() : play()));
  $("scrub").addEventListener("input", (e) => seekAll(parseFloat(e.target.value)));
  document.querySelectorAll(".speed").forEach((b) => b.addEventListener("click", () => {
    rate = parseFloat(b.dataset.rate);
    document.querySelectorAll(".speed").forEach((x) => x.classList.toggle("on", x === b));
    TRIALS.forEach((T) => (T.video.playbackRate = rate));
  }));
  document.querySelectorAll("[data-layer]").forEach((c) => c.addEventListener("change", () => (LAYERS[c.dataset.layer] = c.checked)));
}
function play() {
  if (tMaster >= duration - 0.05) seekAll(0);
  playing = true; lastNow = performance.now();
  $("play").textContent = "❚❚ Pause";
  TRIALS.forEach((T) => { T.video.playbackRate = rate; if (tMaster < T.duration) T.video.play().catch(() => {}); });
}
function pause() {
  playing = false;
  $("play").textContent = "▶ Play";
  TRIALS.forEach((T) => T.video.pause());
}
function seekAll(t) {
  tMaster = Math.max(0, Math.min(duration, t));
  TRIALS.forEach((T) => { T.video.currentTime = Math.min(tMaster, T.duration); });
  resetFeed();
}
function syncVideos() {
  for (const T of TRIALS) {
    const v = T.video, target = Math.min(tMaster, T.duration);
    if (tMaster >= T.duration) { if (!v.paused) v.pause(); continue; }
    if (playing && v.paused && v.readyState >= 2) v.play().catch(() => {});
    if (Math.abs(v.currentTime - target) > 0.2) v.currentTime = target;   // keep both on the shared clock
  }
}

// ---------- the loop ----------
let lastPanel = 0;
function loop(now) {
  if (playing) {
    tMaster += ((now - lastNow) / 1000) * rate;
    if (tMaster >= duration) { tMaster = duration; pause(); }
  }
  lastNow = now;
  syncVideos();
  $("scrub").value = tMaster.toFixed(2);
  $("clock").textContent = `${f(tMaster, 2)} s / ${f(duration, 1)} s`;
  TRIALS.forEach(drawOverlay);
  if (now - lastPanel > 90) { updatePanel(now); lastPanel = now; }
  requestAnimationFrame(loop);
}

// ---------- overlay ----------
function drawOverlay(T) {
  const c = T.canvas, v = T.video, dpr = window.devicePixelRatio || 1;
  const W = c.clientWidth, H = c.clientHeight;
  if (c.width !== Math.round(W * dpr) || c.height !== Math.round(H * dpr)) { c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); }
  const ctx = c.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  if (!v.videoWidth) return;
  const [fw, fh] = T.frame_size;                       // analysis pixel coordinates
  const s = Math.min(W / fw, H / fh), ox = (W - fw * s) / 2, oy = (H - fh * s) / 2;
  const P = (x, y) => [ox + x * s, oy + y * s];
  const t = trialTime(T), fr = frameAt(T, t);
  if (!fr) return;
  const pulse = hl.trial === T.id && performance.now() < hl.until;
  const hiFor = (layer) => pulse && hl.layer === layer;
  ctx.lineCap = "round"; ctx.lineJoin = "round";
  ctx.font = "600 12px system-ui,sans-serif";

  // trail: the last ~1 s of tracked centroids, fading
  if (LAYERS.trail) {
    const i1 = lastLE(T.frames, t + 0.5 / T.fps, (q) => q.t);
    let prev = null;
    for (let i = Math.max(0, i1 - Math.ceil(T.fps * 1.0)); i <= i1; i++) {
      const q = T.frames[i];
      if (!q.b) { prev = null; continue; }
      const [x, y] = P(q.b[0], q.b[1]);
      if (prev) {
        const age = t - q.t;
        ctx.strokeStyle = hiFor("trail") ? `rgba(255,224,138,${1 - age})` : `rgba(90,169,255,${Math.max(0, 0.9 - 0.9 * age)})`;
        ctx.lineWidth = hiFor("trail") ? 4 : 2.5;
        ctx.beginPath(); ctx.moveTo(prev[0], prev[1]); ctx.lineTo(x, y); ctx.stroke();
      }
      prev = [x, y];
    }
  }
  const fit = P(fr.fit[0], fr.fit[1]);
  const mp = fr.mp ? P(fr.mp[0], fr.mp[1]) : null;
  // string (model): from the measured pivot (else the fit centre) to the bob
  if (LAYERS.string && fr.b) {
    const [bx, by] = P(fr.b[0], fr.b[1]), a = mp || fit;
    ctx.setLineDash([7, 5]);
    ctx.strokeStyle = hiFor("string") ? "#ffe08a" : "rgba(240,160,80,.95)";
    ctx.lineWidth = hiFor("string") ? 3.5 : 2;
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(bx, by); ctx.stroke();
    ctx.setLineDash([]);
    label(ctx, "string (model)", (a[0] + bx) / 2 + 8, (a[1] + by) / 2, "rgba(240,160,80,1)");
  }
  // mask outline and centroid
  if (LAYERS.mask && fr.mask) {
    ctx.beginPath();
    for (let i = 0; i < fr.mask.length; i += 2) {
      const [x, y] = P(fr.mask[i], fr.mask[i + 1]);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.closePath();
    ctx.fillStyle = "rgba(95,209,154,.18)"; ctx.fill();
    ctx.strokeStyle = "#5fd19a"; ctx.lineWidth = 1.6; ctx.stroke();
  }
  if (fr.b) {
    const [bx, by] = P(fr.b[0], fr.b[1]);
    ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.arc(bx, by, 3, 0, 7); ctx.fill();
  }
  // pivots: measured (ring) and circle-fit centre (cross, origin of θ)
  if (LAYERS.pivot) {
    const big = hiFor("pivot");
    if (mp) {
      ctx.strokeStyle = big ? "#ffe08a" : "#5fd19a"; ctx.lineWidth = big ? 3.5 : 2;
      ctx.beginPath(); ctx.arc(mp[0], mp[1], big ? 11 : 8, 0, 7); ctx.stroke();
      label(ctx, "measured pivot", mp[0] + 12, mp[1] - 8, "#5fd19a");
    }
    ctx.strokeStyle = big ? "#ffe08a" : "#c9ced6"; ctx.lineWidth = 2;
    const r = 6;
    ctx.beginPath(); ctx.moveTo(fit[0] - r, fit[1] - r); ctx.lineTo(fit[0] + r, fit[1] + r);
    ctx.moveTo(fit[0] - r, fit[1] + r); ctx.lineTo(fit[0] + r, fit[1] - r); ctx.stroke();
    label(ctx, "circle-fit centre (origin of θ)", fit[0] + 10, fit[1] + 16, "#c9ced6");
  }
  // angle: arc from the rest direction to the bob, about the circle-fit centre
  if (LAYERS.angle && fr.b && fr.th !== undefined) {
    const eq = T.theta_eq_deg * Math.PI / 180, th = fr.th * Math.PI / 180;
    const [bx, by] = P(fr.b[0], fr.b[1]);
    const R = Math.hypot(bx - fit[0], by - fit[1]), rArc = Math.max(28, 0.3 * R);
    const a0 = Math.PI / 2 - eq, a1 = Math.PI / 2 - (eq + th);   // canvas angles (y down)
    ctx.strokeStyle = "rgba(201,206,214,.6)"; ctx.lineWidth = 1; ctx.setLineDash([3, 4]);
    ctx.beginPath(); ctx.moveTo(fit[0], fit[1]); ctx.lineTo(fit[0] + R * Math.cos(a0), fit[1] + R * Math.sin(a0)); ctx.stroke();
    ctx.setLineDash([]);
    ctx.strokeStyle = hiFor("angle") ? "#ffe08a" : "#ffd166"; ctx.lineWidth = hiFor("angle") ? 4 : 2.5;
    ctx.beginPath(); ctx.arc(fit[0], fit[1], rArc, a0, a1, a1 < a0); ctx.stroke();
    const am = (a0 + a1) / 2;
    label(ctx, `θ = ${fr.th >= 0 ? "+" : ""}${f(fr.th, 1)}°`, fit[0] + (rArc + 10) * Math.cos(am), fit[1] + (rArc + 10) * Math.sin(am), "#ffd166", 14);
    // zero-crossing tick: flashes on the rest line for 0.25 s after each crossing
    const ci = lastLE(T.crossings, t);
    if (ci >= 0 && t - T.crossings[ci] < 0.25) {
      const tx = fit[0] + R * Math.cos(a0), ty = fit[1] + R * Math.sin(a0);
      ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(tx - 16 * Math.sin(a0), ty + 16 * Math.cos(a0)); ctx.lineTo(tx + 16 * Math.sin(a0), ty - 16 * Math.cos(a0)); ctx.stroke();
    }
  }
  // counters
  const n = lastLE(T.cycle_ends, t) + 1, nc = lastLE(T.crossings, t) + 1;
  label(ctx, `zero crossings ${nc} · cycles measured ${n}`, 10, 18, "#e9eaee", 13);
  if (T.camera_corrected) label(ctx, "camera-motion-corrected track", 10, 36, "#9aa0ab", 12);
}
function label(ctx, text, x, y, colour, size = 12) {
  ctx.font = `600 ${size}px system-ui,sans-serif`;
  const w = ctx.measureText(text).width;
  ctx.fillStyle = "rgba(15,17,21,.72)"; ctx.fillRect(x - 3, y - size, w + 6, size + 5);
  ctx.fillStyle = colour; ctx.fillText(text, x, y);
}

// ---------- science panel ----------
function updatePanel(now) {
  const state = TRIALS.map((T) => { const t = trialTime(T); return { T, t, s: snapAt(T, t), fr: frameAt(T, t) }; });
  const hlOn = now < hl.until;

  // measurement, live numbers
  $("m-live").innerHTML = "";
  for (const { T, fr } of state) {
    const d = document.createElement("div");
    d.className = "live";
    d.innerHTML = `<div class="lbl" style="color:${T.colour}">${esc(T.short)}</div><div class="math"></div>`;
    $("m-live").appendChild(d);
    const el = d.querySelector(".math");
    if (fr && fr.b && fr.th !== undefined) {
      tex(el, String.raw`\theta = \operatorname{atan2}\!\left(${f(fr.b[0], 1)} - ${f(fr.fit[0], 1)},\; ${f(fr.b[1], 1)} - ${f(fr.fit[1], 1)}\right) - (${f(T.theta_eq_deg, 2)}^{\circ}) = ${f(fr.th, 2)}^{\circ}`);
    } else el.innerHTML = `<div class="waiting">no tracked position at this frame</div>`;
  }

  // hypothesis cards: per-trial status and ΔBIC
  for (const law of [...PERIOD_LAWS, ...DAMP_LAWS]) {
    law.card.classList.toggle("hl", (hlOn && hl.law === law.id) || hoverLaw === law.id);
    law.badges.innerHTML = state.map(({ T, s }) => {
      const fit = s && s[law.key] ? s[law.key][law.name] : null;
      const d = fit ? fit.dbic : null, st = status(d);
      const extra = law.id === "H4" && fit && fit.params.c !== undefined ? ` · c = ${f(fit.params.c, 3)} ± ${f(fit.errors.c, 3)}` : "";
      return `<span class="chip ${st}"><span class="dot" style="background:${T.colour}"></span>${esc(T.short)}: <span class="st">${STATUS_TEXT[st]}</span>${d !== null ? ` · ΔBIC ${f(d, 1)}` : ""}${extra}</span>`;
    }).join(" ");
  }

  // g, solved with each trial's leading law, values substituted
  $("g-live").innerHTML = "";
  for (const { T, s } of state) {
    const d = document.createElement("div");
    d.className = "live";
    d.innerHTML = `<div class="lbl" style="color:${T.colour}">${esc(T.short)}</div><div class="math"></div>`;
    $("g-live").appendChild(d);
    const el = d.querySelector(".math");
    if (s && s.g !== undefined) {
      const law = LAW_BY_NAME[s.leading];
      tex(el, String.raw`\begin{aligned}
        T_0 &= ${f(s.T0, 4)}\ \mathrm{s} \qquad \text{(${law.id} ${law.label}, ${s.n} cycles)}\\[2pt]
        g &= \frac{4\pi^{2}\,(${f(T.L_eff, 4)}\ \mathrm{m})}{(${f(s.T0, 4)}\ \mathrm{s})^{2}}
           = ${f(s.g, 3)} \pm ${f(s.g_err_total, 3)}\ \mathrm{m\,s^{-2}}
      \end{aligned}`);
      tex(T.elG, String.raw`g = ${f(s.g, 3)} \pm ${f(s.g_err_total, 3)}\ \mathrm{m\,s^{-2}}`, false);
    } else {
      el.innerHTML = `<div class="waiting">waiting for cycles (${s ? s.n : 0} of 3 needed)</div>`;
      T.elG._tex = null; T.elG.innerHTML = `<span class="waiting">g: waiting for cycles</span>`;
    }
  }

  state.forEach(({ T, s }) => drawPlot(T, s, hlOn));
  updateDiagnoses(state, hlOn);
  updateFinal(state);
  updateFeed(state);
}

function drawPlot(T, s, hlOn) {
  const c = T.plot, dpr = window.devicePixelRatio || 1, W = c.clientWidth, H = c.clientHeight;
  if (c.width !== Math.round(W * dpr)) { c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); }
  const ctx = c.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const pts = T.snapshots.filter((x) => s && x.t <= s.t + 1e-9);
  // fixed axes from the whole trial, so points do not jump around as cycles arrive
  const allA = T.snapshots.map((x) => x.amp_deg), allP = T.snapshots.map((x) => x.period_s * 1000);
  const aMin = 0, aMax = Math.ceil((Math.max(...allA) + 3) / 5) * 5;
  const pLo = Math.floor((Math.min(...allP) - 10) / 10) * 10, pHi = Math.ceil((Math.max(...allP) + 10) / 10) * 10;
  const L = 34, R = 8, Tp = 8, B = 22;
  const X = (a) => L + (a - aMin) / (aMax - aMin) * (W - L - R), Y = (p) => H - B - (p - pLo) / (pHi - pLo) * (H - Tp - B);
  ctx.strokeStyle = "#2a2e36"; ctx.lineWidth = 1; ctx.fillStyle = "#9aa0ab"; ctx.font = "10px system-ui,sans-serif";
  for (let k = 0; k <= 4; k++) {
    const p = pLo + (pHi - pLo) * k / 4, y = Y(p);
    ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(W - R, y); ctx.stroke(); ctx.fillText(f(p, 0), 2, y + 3);
    const a = aMin + (aMax - aMin) * k / 4, x = X(a);
    ctx.fillText(f(a, 0) + "°", x - 6, H - 6);
  }
  if (!pts.length) { ctx.fillStyle = "#9aa0ab"; ctx.font = "italic 12px system-ui"; ctx.fillText("waiting for cycles", L + 10, H / 2); return; }
  // law curves from the current fit: all fitted laws faint, the leading one bold, a hovered one highlighted
  if (s && s.period_laws) {
    for (const law of PERIOD_LAWS) {
      const fit = s.period_laws[law.name];
      if (!fit || fit.dbic === null) continue;
      const lead = law.name === s.leading, hov = hoverLaw === law.id || (hlOn && hl.law === law.id);
      ctx.strokeStyle = hov ? "#ffe08a" : lead ? T.colour : "rgba(201,206,214,.28)";
      ctx.lineWidth = hov ? 3 : lead ? 2.2 : 1;
      ctx.save(); ctx.beginPath(); ctx.rect(L, Tp, W - L - R, H - Tp - B); ctx.clip();   // clip, never clamp
      ctx.beginPath();
      for (let k = 0; k <= 80; k++) {
        const a = aMin + (aMax - aMin) * k / 80, p = 1000 * periodCurve(law.id, fit.params, T.L_eff, a * Math.PI / 180);
        k ? ctx.lineTo(X(a), Y(p)) : ctx.moveTo(X(a), Y(p));
      }
      ctx.stroke(); ctx.restore();
    }
  }
  const glow = hlOn && hl.trial === T.id;
  for (const p of pts) {
    ctx.fillStyle = glow ? "#ffe08a" : T.colour;
    ctx.beginPath(); ctx.arc(X(p.amp_deg), Y(p.period_s * 1000), glow ? 3.6 : 2.8, 0, 7); ctx.fill();
  }
}

function updateDiagnoses(state, hlOn) {
  const box = $("diagnoses");
  const cards = [];
  for (const { T, s } of state) {
    if (!s) continue;
    if (T.id === "shoelace" && s.n >= MIN_EV && s.g !== undefined && Math.abs(s.g - G_LOCAL) > 2 * s.g_err_total && s.implied_L !== undefined) {
      cards.push({ id: "shoelace", T, html: `<h3 style="color:${T.colour}">Shoelace: the pendulum is shorter than measured</h3>
        <div class="math" data-tex="${esc(String.raw`L_{\mathrm{eff}} = \frac{g\,T_0^{2}}{4\pi^{2}} = \frac{9.79\ \mathrm{m\,s^{-2}}\times(${f(s.T0_exact, 4)}\ \mathrm{s})^{2}}{4\pi^{2}} = ${f(s.implied_L, 3)}\ \mathrm{m} \quad\text{against the measured}\quad ${f(T.L, 3)}\ \mathrm{m}`)}"></div>
        <p>Taking Cairo's g and the exact law's \\(T_0\\), the swing behaves like a pendulum ${f(100 * (T.L - s.implied_L), 1)} cm shorter than the one measured.</p>
        ${s.pivot_side_frac !== undefined && s.pivot_side_frac > 0.02 ? `<p>Flag: the circle through the path is centred ${f(100 * s.pivot_side_frac, 1)}% of its radius to the side of the measured pivot (the phone was about 14° off square).</p>` : ""}
        <p class="muted">Likely cause: the lace's own mass, which shortens the effective length (not yet tested by weighing).</p>` });
    }
    if (T.id === "keys" && s.n >= MIN_EV && s.c !== undefined && Math.abs(s.c - 1 / 16) > 2 * s.c_err && s.leading === "free quadratic") {
      cards.push({ id: "keys", T, html: `<h3 style="color:${T.colour}">Keys: not a simple pendulum</h3>
        <div class="math" data-tex="${esc(String.raw`c = ${f(s.c, 3)} \pm ${f(s.c_err, 3)} \quad\text{against the exact law's}\quad \tfrac{1}{16} = 0.0625`)}"></div>
        <p>The period rises with amplitude faster than any rigid simple pendulum allows.</p>
        ${s.pivot_side_frac !== undefined && s.pivot_side_frac > 0.02 ? `<p>Flag: the path is centred ${f(100 * s.pivot_side_frac, 1)}% of its radius off the measured pivot.</p>` : ""}
        <p>Flag: ${s.mask_jumps} frames with mask jumps so far.</p>
        <p class="muted">Diagnosis: cord mass and keys swinging on the ring make this a compound pendulum.</p>` });
    }
  }
  const key = cards.map((c) => c.html).join("|");
  if (box._key !== key) {
    box._key = key;
    box.innerHTML = cards.length ? cards.map((c) => `<div class="diag" data-diag="${c.id}">${c.html}</div>`).join("")
      : `<p class="muted">Diagnosis cards appear when the evidence for them exists.</p>`;
    box.querySelectorAll("[data-tex]").forEach((el) => tex(el, el.dataset.tex));
    renderMathInElement(box, { delimiters: [{ left: "\\(", right: "\\)", display: false }], throwOnError: false });
  }
  box.querySelectorAll(".diag").forEach((d) => d.classList.toggle("hl", hlOn && hl.diag === d.dataset.diag));
  if (hlOn && hl.diag && !hl.scrolled) {
    const d = box.querySelector(`[data-diag="${hl.diag}"]`);
    if (d) { d.scrollIntoView({ block: "nearest", behavior: "smooth" }); hl.scrolled = true; }
  }
}

function updateFinal(state) {
  const done = state.every(({ T, t }) => t >= T.duration - 0.06);
  const card = $("final-card");
  if (!done) { card.hidden = true; return; }
  if (!card.hidden) return;
  card.hidden = false;
  card.innerHTML = `<h2>Final result <span class="badge real">REAL DATA</span></h2><div class="final-grid">${state.map(({ T, s }) => {
    const lead = LAW_BY_NAME[s.leading];
    const runner = Object.entries(s.period_laws).filter(([n, v]) => n !== s.leading && v.dbic !== null && v.dbic <= 2).map(([n]) => LAW_BY_NAME[n].id);
    const damp = Object.entries(s.damping_laws).sort((a, b) => a[1].dbic - b[1].dbic)[0][0];
    return `<div><div class="lbl" style="color:${T.colour};font-weight:700">${esc(T.title)}</div>
      <div class="math g" data-tex="${esc(String.raw`g = ${f(s.g, 3)} \pm ${f(s.g_err_total, 3)}\ \mathrm{m\,s^{-2}} \quad\text{vs. Cairo}\ \approx 9.79`)}"></div>
      <div class="small">Period law: ${lead.id} ${esc(lead.label)}${runner.length ? ` (tied with ${runner.join(", ")})` : ""} · Damping law: ${esc(LAW_BY_NAME[damp].label)} · ${s.n} cycles</div>
      <div class="small"><span class="badge flag">APPARATUS FLAGGED</span> <span class="muted">clip ${esc(T.clip_name)} · ${esc(T.result_dir)}</span></div></div>`;
  }).join("")}</div>`;
  card.querySelectorAll("[data-tex]").forEach((el) => tex(el, el.dataset.tex));
}

// ---------- Kepler's live feed ----------
let shown = new Set(), typingQueue = [];
function resetFeed() { shown = new Set(); typingQueue = []; $("feed").innerHTML = ""; }
function feedItems() {
  return TRIALS.flatMap((T) => (T.feed || []).map((m, i) => ({ ...m, T, key: `${T.id}:${i}` }))).sort((a, b) => a.t - b.t);
}
function updateFeed(state) {
  const times = Object.fromEntries(state.map(({ T, t }) => [T.id, t]));
  const items = feedItems();
  if (!items.length && !$("feed").childElementCount) {
    $("feed").innerHTML = `<p class="muted small">No feed messages in this build.</p>`;
    return;
  }
  for (const m of items) {
    if (shown.has(m.key) || m.t > times[m.T.id]) continue;
    shown.add(m.key);
    const el = document.createElement("div");
    el.className = "msg";
    el.innerHTML = `<div class="meta"><span class="chip"><span class="dot" style="background:${m.T.colour}"></span>${esc(m.T.short)}</span>
      <span>t = ${f(m.t, 1)} s</span><span>Kepler (Muse Wonder)</span></div><div class="body"></div>`;
    $("feed").appendChild(el);
    const live = playing && times[m.T.id] - m.t < 2;
    if (live) typeOut(el, m.text); else { el.querySelector(".body").innerHTML = renderInline(m.text); }
    $("feed").scrollTop = $("feed").scrollHeight;
  }
}
function renderInline(text) {
  return text.split(/(\$[^$]+\$)/).map((seg) => seg.startsWith("$") && seg.endsWith("$") && seg.length > 1
    ? katex.renderToString(seg.slice(1, -1), { throwOnError: false, strict: "ignore" }) : esc(seg)).join("");
}
function typeOut(el, text) {   // text types character by character; each $math$ appears whole, typeset
  const body = el.querySelector(".body"), segs = text.split(/(\$[^$]+\$)/);
  el.classList.add("typing");
  let i = 0, j = 0, html = "";
  const step = () => {
    if (i >= segs.length) { el.classList.remove("typing"); return; }
    const seg = segs[i];
    if (seg.startsWith("$") && seg.endsWith("$") && seg.length > 1) {
      html += katex.renderToString(seg.slice(1, -1), { throwOnError: false, strict: "ignore" }); i++; j = 0;
    } else if (j < seg.length) { html += esc(seg[j]); j++; }
    else { i++; j = 0; }
    body.innerHTML = html;
    setTimeout(step, 14);
  };
  step();
}

// ---------- test hook: #t=<seconds> seeks, pauses and reports both videos' time ----------
async function testSeek(t) {
  await Promise.all(TRIALS.map((T) => new Promise((res) => T.video.readyState >= 1 ? res() : T.video.addEventListener("loadedmetadata", res, { once: true }))));
  seekAll(t);
  await Promise.all(TRIALS.map((T) => new Promise((res) => {
    if (!T.video.seeking) res(); else T.video.addEventListener("seeked", res, { once: true });
  })));
  document.title = `sync ${TRIALS.map((T) => `${T.id}=${T.video.currentTime.toFixed(3)}`).join(" ")}`;
  document.body.dataset.ready = "1";
}

window.addEventListener("load", () => load().catch((e) => {
  document.body.insertAdjacentHTML("afterbegin", `<p style="color:#ff5d5d;padding:20px">Could not load the data: ${esc(e.message)}. Serve this folder over HTTP (GitHub Pages, or <code>python -m http.server</code>).</p>`);
}));
