"use strict";
const $ = (id) => document.getElementById(id);
const TOKEN = location.pathname.split("/")[1];        // URL: /<token>/...
const api = (p) => `/${TOKEN}${p}`;
let offline = false, pollTimer = null, heartbeatTimer = null;
let submitting = false;
const hiddenJobs = new Set();      // client-side "Hide" on finished cards
const cardEls = new Map();         // job_id -> { el, sig } (stable rendering)
const prevOutcomes = new Map();    // "job:index" -> outcome (aria-live transitions)

async function jfetch(url, opts) {
  try {
    const r = await fetch(url, opts);
    if (r.status === 404 && url.includes("/api/")) { showOffline(); throw new Error("gone"); }
    return r;
  } catch (e) { showOffline(); throw e; }
}
function showOffline(msg) {
  if (offline) return; offline = true;
  if (msg) $("offlineText").textContent = msg;
  $("offline").hidden = false; $("drop").classList.add("disabled");
  document.querySelectorAll("button").forEach(b => b.disabled = true);
  // Download is an <a>, not a button; the offline rule pins it disabled too.
  document.querySelectorAll("#jobs a").forEach(a => { a.style.pointerEvents = "none"; a.style.opacity = ".5"; });
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  if (heartbeatTimer) { clearInterval(heartbeatTimer); heartbeatTimer = null; }
}
function announce(text) { $("announcer").textContent = text; }

// Map a mark-mode radio value to the server options it sends. One mutually
// exclusive choice replaces the old legacy/force/preset trio, so the two
// combinations the server rejects (preset with legacy or force) cannot be
// expressed at all.
function modeOptions(group) {
  const el = document.querySelector(`input[name="${group}"]:checked`);
  const mode = el.value;
  const o = {};
  if (mode === "usual") o.forceRemove = true;
  if (mode === "older") { o.legacy = true; o.forceRemove = true; }  // --force --legacy
  if (el.dataset.preset && (mode === "small" || mode === "large")) o.geoPreset = el.dataset.preset;
  return o;
}

async function init() {
  try {
    const v = await (await jfetch(api("/api/version"))).json();
    $("version").textContent = "v" + v.version;
    const servedPresets = v.presets || [];
    // A preset the server does not know (renamed upstream) degrades to
    // disabled-with-a-reason instead of a 400 on every upload.
    document.querySelectorAll("input[data-preset]").forEach(r => {
      if (!servedPresets.includes(r.dataset.preset)) {
        r.disabled = true;
        r.closest("label").title = "Not available in this build of wmr";
        r.closest("label").classList.add("muted");
      }
    });
    if (v.features && v.features.denoise_ai)
      document.querySelector('#denoise option[value=ai]').hidden = false;
    checkUpdate(v.update);                       // fresh-cache result is already here
    setTimeout(async () => {                     // the background fetch resolves later
      try { checkUpdate((await (await jfetch(api("/api/version"))).json()).update); }
      catch (e) { /* offline banner already up */ }
    }, 4000);
  } catch (e) { /* offline banner already up */ }
  await refreshJobs();

  // Drop anywhere on the page feeds the same submit path; without this a
  // miss-drop lets the browser navigate away to display the image.
  document.addEventListener("dragover", e => e.preventDefault());
  document.addEventListener("drop", e => {
    e.preventDefault();
    if (e.dataTransfer && e.dataTransfer.files.length) submit([...e.dataTransfer.files]);
  });
  let dragDepth = 0;                             // enter/leave counter: no flicker
  $("drop").addEventListener("dragenter", () => { if (++dragDepth) $("drop").classList.add("dragging"); });
  $("drop").addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; $("drop").classList.remove("dragging"); } });
  $("drop").addEventListener("drop", () => { dragDepth = 0; $("drop").classList.remove("dragging"); });
  $("browse").addEventListener("click", () => $("file").click());
  // The focus affordance the tabindex promises: Enter/Space opens the picker.
  $("drop").addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("file").click(); }
  });
  // Reset after reading: browsers fire no change event when the picker
  // re-selects the SAME files, which would silently ignore a re-run of the
  // same image with different options (the classic file-input gotcha).
  $("file").addEventListener("change", (e) => { submit([...e.target.files]); e.target.value = ""; });
  // Hidden finished cards come back on demand (display-only; jobs stay on
  // the server).
  $("showHidden").addEventListener("click", () => { hiddenJobs.clear(); refreshJobs(); });

  // The closed Advanced summary reflects non-default choices, so a leftover
  // pick cannot silently apply to the next drop.
  const updateSummary = () => {
    const bits = [];
    const mode = document.querySelector('input[name="markmode"]:checked');
    if (mode && mode.value !== "auto")
      bits.push(mode.parentElement.querySelector("strong").textContent.toLowerCase());
    if ($("denoise").value !== "off") bits.push("cleanup: " + $("denoise").value);
    if ($("keepprov").checked) bits.push("keep provenance");
    $("advanced").querySelector("summary").textContent =
      bits.length ? `Advanced (${bits.join(" · ")})` : "Advanced";
    // The cleanup select's one-line hint follows the selection.
    $("denoiseHint").textContent = {
      off: "Undo the overlay exactly, with no other change (recommended).",
      soft: "Gentle blur over the spot, for when a faint trace remains.",
      ns: "Fill the spot in from the surrounding pixels.",
      telea: "Fill the spot in from the surrounding pixels (a second algorithm).",
      ai: "Neural cleanup of the spot; the slowest option.",
    }[$("denoise").value] || "";
  };
  document.querySelectorAll('#markmode input, #denoise, #keepprov')
    .forEach(el => el.addEventListener("change", updateSummary));
  updateSummary();

  // Quit button: the server's graceful path (same as Ctrl-C). After the POST
  // the page cannot reach it anymore; the banner explains.
  $("quit").addEventListener("click", async () => {
    if (!confirm("Stop wmr? Running jobs will be cancelled and session files " +
                 "deleted. The page will go offline.")) return;
    $("quit").disabled = true;
    try {
      await jfetch(api("/api/shutdown"), { method: "POST" });
      showOffline("wmr was stopped from this page. Restart wmr and open the new URL it prints.");
    } catch (e) { /* jfetch already showed the offline banner */ }
  });

  // Clicking OUTSIDE THE DIALOG BOX (the backdrop) closes it; clicks inside
  // the dialog box, including its padding, do not. Two guards:
  // the geometric test (padding is inside the rect) AND target === dialog
  // (keyboard-synthesized clicks carry clientX/Y = 0 and target the focused
  // inner element, so they are filtered by both). Esc and the Close/Cancel
  // buttons also close.
  [$("compare"), $("manual")].forEach(d =>
    d.addEventListener("click", (e) => {
      const r = d.getBoundingClientRect();
      if (e.target === d &&
          (e.clientX < r.left || e.clientX > r.right ||
           e.clientY < r.top || e.clientY > r.bottom)) d.close();
    }));
}

// Show the update notice when the server reports a known-newer release
// (absent on WMR_UPDATE_CHECK=OFF builds or when opted out).
function checkUpdate(u) {
  if (!(u && u.known && u.newer)) return;
  $("updateLatest").textContent = u.latest;
  $("updateLink").href = u.url;
  $("updateNotice").hidden = false;
}

function options() {
  return Object.assign(modeOptions("markmode"), {
    denoise: $("denoise").value,
    keepProvenance: $("keepprov").checked,
  });
}

const JOB_CAP_BYTES = 1073741824, JOB_CAP_FILES = 100;
async function submit(files) {
  if (offline || !files.length || submitting) return;
  // No client-side extension filter: the server's sniff classifies every file,
  // and unsupported ones come back as visible per-file "unsupported format"
  // rows (the spec's upload contract). The split below enforces the job caps.
  const chunks = []; let cur = [], curB = 0;           // client-side split per the caps
  for (const f of files) {
    if (cur.length && (cur.length >= JOB_CAP_FILES || curB + f.size > JOB_CAP_BYTES)) {
      chunks.push(cur); cur = []; curB = 0;            // never push an empty chunk
    }
    cur.push(f); curB += f.size;
  }
  if (cur.length) chunks.push(cur);

  submitting = true;                                    // block accidental double drops
  const failedChunks = [];
  for (let c = 0; c < chunks.length; c++) {
    $("dropStatus").hidden = false;
    $("dropStatus").textContent =
      `Adding ${files.length} file${files.length > 1 ? "s" : ""}…` +
      (chunks.length > 1 ? ` (part ${c + 1} of ${chunks.length})` : "");
    const fd = new FormData();
    chunks[c].forEach(f => fd.append("files", f, f.name));
    fd.append("options", JSON.stringify(options()));
    try {
      const r = await jfetch(api("/api/jobs"), { method: "POST", body: fd });
      if (!r.ok) { failedChunks.push(...chunks[c].map(f => f.name)); continue; }
    } catch (e) { failedChunks.push(...chunks[c].map(f => f.name)); break; }
  }
  submitting = false;
  $("dropStatus").hidden = true; $("dropStatus").textContent = "";
  if (failedChunks.length) {
    // One visible inline message; never a silent loss and never alert().
    $("dropMsg").hidden = false;
    $("dropMsg").textContent =
      `${failedChunks.length} file${failedChunks.length > 1 ? "s" : ""} could not be ` +
      `uploaded: ${failedChunks.slice(0, 5).join(", ")}` +
      (failedChunks.length > 5 ? "…" : "");
  }
  await refreshJobs(); startPolling();
}
function startPolling() {
  if (pollTimer || offline) return;
  pollTimer = setInterval(refreshJobs, 500);
  if (heartbeatTimer) { clearInterval(heartbeatTimer); heartbeatTimer = null; }
}
// Idle liveness: with no running jobs the fast poll stops, so a Ctrl-C'd or
// dead server would otherwise go unnoticed until the next Download click
// (which is a plain link, not a jfetch). A slow heartbeat notices for us.
function startHeartbeatIfIdle() {
  if (heartbeatTimer || pollTimer || offline) return;
  heartbeatTimer = setInterval(async () => {
    try { await jfetch(api("/api/version")); } catch (e) { /* banner is up */ }
  }, 12000);
}

const BADGE = { pending: "…", queued: "queued", removed: "removed", "no-watermark": "no watermark found",
                failed: "failed", "not-run": "skipped" };
let refreshInFlight = null;
function refreshJobs() {
  // Single-flight: callers (interval, submit, cancel, retry) must not
  // interleave clear/append cycles.
  return refreshInFlight ??= doRefresh().finally(() => { refreshInFlight = null; });
}
async function doRefresh() {
  let jobs;
  try { jobs = await (await jfetch(api("/api/jobs"))).json(); }
  catch (e) { return; }
  const visible = jobs.filter(j => !hiddenJobs.has(j.job_id));
  let anyRunning = false;
  const seen = new Set();
  for (const j of visible) {
    let full;
    try { full = await (await jfetch(api("/api/jobs/" + j.job_id))).json(); }
    catch (e) { return; }
    if (full.status === "queued" || full.status === "running") anyRunning = true;
    seen.add(j.job_id);
    const sig = signature(full);
    const existing = cardEls.get(j.job_id);
    if (existing && existing.sig === sig) continue;      // untouched: keep the node
    const el = renderJob(full);
    if (existing) existing.el.replaceWith(el);
    else $("jobs").prepend(el);
    cardEls.set(j.job_id, { el, sig });
    announceTransitions(full);
  }
  for (const [id, { el }] of cardEls) {
    if (!seen.has(id)) { el.remove(); cardEls.delete(id); }
  }
  const anyHidden = hiddenJobs.size > 0;
  $("showHidden").hidden = !anyHidden;
  if (anyRunning) startPolling();
  else { if (pollTimer) { clearInterval(pollTimer); pollTimer = null; } startHeartbeatIfIdle(); }
}
function signature(j) {
  return JSON.stringify([j.status, j.current_file_index, j.queue_position,
    j.files.map(f => [f.outcome, f.error, f.forced, f.geometry_source, f.bbox, f.score])]);
}
function announceTransitions(j) {
  j.files.forEach((f, i) => {
    const key = `${j.job_id}:${i}`;
    const was = prevOutcomes.get(key);
    if (was !== undefined && was !== f.outcome && f.outcome !== "pending") {
      const said = f.outcome === "removed" ? "watermark removed"
                 : f.outcome === "no-watermark" ? "no watermark found"
                 : f.outcome === "not-run" ? null : f.outcome;
      if (said) announce(`${f.name}: ${said}`);
      if (f.outcome === "removed" || f.outcome === "failed" || f.outcome === "not-run") {
        flashTitleIfHidden();
      }
    }
    if (f.outcome !== "pending") prevOutcomes.set(key, f.outcome);
  });
}
function flashTitleIfHidden() {
  if (!document.hidden || document.title.startsWith("Done")) return;
  document.title = "Done · wmr";
  const restore = () => { document.title = "wmr"; document.removeEventListener("visibilitychange", restore); };
  document.addEventListener("visibilitychange", restore);
}

const MODE_LABELS = { auto: "Find it automatically", usual: "The usual spot (skip the search)",
                      small: "Small diamond, close to the corner", large: "Large diamond, further from the corner",
                      older: "Older watermark (before Gemini 3.5)" };
function renderJob(j) {
  const card = document.createElement("section"); card.className = "job";
  const head = document.createElement("h2");
  // The snapshot's progress fields, rendered (they were served but unused).
  if (j.status === "running")
    head.textContent = `Running · file ${Math.min((j.current_file_index ?? 0) + 1, j.files.length)} of ${j.files.length}`;
  else if (j.status === "queued")
    head.textContent = `Queued · position ${Math.max(j.queue_position ?? 0, 1)} in line`;
  else head.textContent = `${j.status} (${j.files.length} file${j.files.length>1?"s":""})`;
  card.appendChild(head);
  const headWrap = document.createElement("div"); headWrap.className = "jobhead";
  headWrap.appendChild(head);
  if (j.status === "queued" || j.status === "running") {
    const c = document.createElement("button"); c.textContent = "Cancel";
    c.addEventListener("click", () => {
      c.disabled = true; c.textContent = "Cancelling…";   // acknowledged; no double click
      jfetch(api(`/api/jobs/${j.job_id}/cancel`), {method:"POST"})
        .then(refreshJobs).catch(() => {});
    });
    headWrap.appendChild(c);
  } else {
    const h = document.createElement("button"); h.className = "hide"; h.textContent = "Hide";
    h.addEventListener("click", () => { hiddenJobs.add(j.job_id); refreshJobs(); });
    headWrap.appendChild(h);
  }
  card.appendChild(headWrap);
  const list = document.createElement("ul");
  j.files.forEach((f, i) => {
    const li = document.createElement("li");
    const name = document.createElement("span"); name.textContent = f.name; li.appendChild(name);
    const badge = document.createElement("span"); badge.className = "badge " + (f.outcome || "pending");
    badge.textContent = BADGE[f.outcome || "pending"];   // the "forced" nuance lives in the detail line
    if ((f.outcome || "pending") === "pending") badge.setAttribute("aria-label", "working");
    li.appendChild(badge);
    if (f.outcome === "removed") {
      const a = document.createElement("a"); a.href = api(`/api/jobs/${j.job_id}/files/${i}/image?kind=cleaned`);
      a.download = ""; a.textContent = "Download"; li.appendChild(a);
      const cmp = document.createElement("button"); cmp.textContent = "Compare";
      cmp.addEventListener("click", () => openCompare(j, i, f)); li.appendChild(cmp);
      renderTypeDetail(li, f);
    }
    if (f.outcome === "no-watermark") {
      const m = document.createElement("button"); m.textContent = "Retry with a hint";
      m.addEventListener("click", () => openRetry(j.job_id, i, f.name)); li.appendChild(m);
    }
    if (f.error) { const e = document.createElement("span"); e.className = "err"; e.textContent = f.error; li.appendChild(e); }
    list.appendChild(li);
  });
  card.appendChild(list);
  return card;
}

// The per-file detail line: "Watermark removed · {which mark} ({W} × {H} px)
// · {how the spot was chosen}", plus the glance note on rows where no
// verifying search ran. The px figure is the footprint actually erased: the
// searched bbox, or (forced rows) the size the engine's rule erased, served
// with the row.
function renderTypeDetail(li, f) {
  const size = f.bbox ? f.bbox[2] : f.mark_size;
  const which = f.variant === "V1" ? "older watermark"
              : size >= 96 ? "large diamond" : "small diamond";
  const sizeSeg = size ? ` (${size} × ${size} px)` : "";
  const how = (f.forced || f.geometry_source === "model") ? "at its usual spot"
            : (f.geometry_source === "preset" || f.geometry_source === "rect") ? "at a known spot"
            : "found automatically";
  const line = document.createElement("span"); line.className = "muted";
  line.textContent = `Watermark removed · ${which}${sizeSeg} · ${how}`;
  li.appendChild(line);
  // The glance note fires ONLY where no verifying search ran (forced, or an
  // explicitly pinned spot). geometry_source "model" on a non-forced row
  // means the detector DID search and confirmed the mark there.
  if (f.forced || f.geometry_source === "preset" || f.geometry_source === "rect") {
    const note = document.createElement("span"); note.className = "muted glance";
    note.textContent = "Spot set without a search · worth a quick glance at the corner";
    li.appendChild(note);
  }
}

// The engine's own position model (core/types.hpp get_watermark_config),
// mirrored client-side so the retry preview shows exactly where each choice
// erases: {logo_size, margin} per mode for a WxH image.
function markConfigFor(mode, W, H) {
  const large = W > 1024 && H > 1024;
  if (mode === "older") return large ? { n: 96, m: 64 } : { n: 48, m: 32 };   // V1
  if (mode === "small") return { n: 48, m: 96 };                              // preset
  if (mode === "large") return { n: 96, m: 192 };                             // preset
  // usual (V2 model): large -> 96 @192; small -> 36 @ scaled margin
  // (v2_small_config_from_dims: 192 * long/side, side by short-side band)
  if (large) return { n: 96, m: 192 };
  const long_s = Math.max(W, H), short_s = Math.min(W, H);
  const src = short_s >= 566 ? 2752 : short_s >= 550 ? 2816 : 2848;
  return { n: 36, m: Math.round(192 * long_s / src) };
}
let cmpMarkBBox = null;   // the compare dialog's current mark box (zoom target)
function openCompare(j, i, f) {
  const a = $("cmpA"), box = $("cmpBox");
  $("cmpB").src = api(`/api/jobs/${j.job_id}/files/${i}/image?kind=cleaned`);
  a.onload = () => {
    // The mark region is drawn on the ORIGINAL layer only (a plain bordered
    // div positioned in percent: crisp and constant thickness at any size).
    // Forced rows carry no bbox; their spot is the position model itself.
    let b = f.bbox;
    if (!b && f.mark_size) {
      const { n, m } = markConfigFor(f.variant === "V1" ? "older" : "usual",
                                     a.naturalWidth, a.naturalHeight);
      b = [a.naturalWidth - m - n, a.naturalHeight - m - n, n, n];
    }
    if (!b) { box.hidden = true; $("cmpZoom").disabled = true; cmpMarkBBox = null; return; }
    $("cmpZoom").disabled = false;
    cmpMarkBBox = b;
    const [x, y, w, h] = b;
    box.style.left = (x / a.naturalWidth * 100) + "%";
    box.style.top = (y / a.naturalHeight * 100) + "%";
    box.style.width = (w / a.naturalWidth * 100) + "%";
    box.style.height = (h / a.naturalHeight * 100) + "%";
    box.hidden = !$("cmpShowBox").checked;
    applyZoom(b);
  };
  a.src = api(`/api/jobs/${j.job_id}/files/${i}/image?kind=original`);
  $("cmpShowBox").onchange = (e) => { box.hidden = !e.target.checked; };
  $("cmpZoom").onchange = () => {
    if (box.hidden && $("cmpZoom").checked) box.hidden = false;
    applyZoom(cmpMarkBBox);
  };
  // First frame must match the knob: clip the cleaned layer to the slider
  // position before the dialog opens (it was showing 100% cleaned before).
  $("cmpBwrap").style.clipPath = `inset(0 ${100 - $("cmpSlider").value}% 0 0)`;
  $("cmpSlider").oninput = (e) =>
    $("cmpBwrap").style.clipPath = `inset(0 ${100 - e.target.value}% 0 0)`;
  $("cmpClose").onclick = () => $("compare").close();
  $("compare").onclose = resetZoom;
  resetZoom();
  $("compare").showModal();
}
// Zoom-to-mark: scale the stack so the mark spans roughly 40% of the dialog
// width, then center it. Both layers, the clip, and the box are
// percent-positioned on the same geometry, so they follow the scale for free.
function applyZoom(b) {
  if (!$("cmpZoom").checked) { resetZoom(); return; }
  const bw = (b && b[2]) || 48;
  const target = Math.min(Math.max(bw * 2.5, 300), 4000);
  $("compare").classList.add("zoomed");
  const stack = document.querySelector(".cmp");
  stack.style.width = target + "px";
  $("cmpBox").scrollIntoView({ block: "center", inline: "center" });
}
function resetZoom() {
  document.querySelector(".cmp").style.width = "";
  $("compare").classList.remove("zoomed");
}

// Position the dashed preview diamond over the original.
function placeRetryOverlay() {
  const img = $("rImg"), ovl = $("rOvl");
  const mode = document.querySelector('input[name="rmarkmode"]:checked').value;
  if (!img.naturalWidth) { ovl.hidden = true; return; }
  const W = img.naturalWidth, H = img.naturalHeight;
  const { n, m } = markConfigFor(mode, W, H);
  const x = W - m - n, y = H - m - n;   // margin is from the right/bottom edges
  ovl.style.left = (x / W * 100) + "%";
  ovl.style.top = (y / H * 100) + "%";
  ovl.style.width = (n / W * 100) + "%";
  ovl.style.height = (n / H * 100) + "%";
  ovl.hidden = false;
}

// Retry for a no-watermark file: the original image with the chosen diamond
// previewed at its standard position, plus the same mark-mode question minus
// the automatic option (already tried), pre-selecting the usual spot (the
// most common cause is a visible mark the search keeps missing).
function openRetry(jobId, fileIndex, fileName) {
  $("manualTitle").textContent = fileName;
  document.querySelector('input[name="rmarkmode"][value="usual"]').checked = true;
  const img = $("rImg");
  img.onload = placeRetryOverlay;
  img.src = api(`/api/jobs/${jobId}/files/${fileIndex}/image?kind=original`);
  document.querySelectorAll('input[name="rmarkmode"]').forEach(r =>
    r.onchange = placeRetryOverlay);   // property rebinding: no listener pile-up across opens
  $("mGo").onclick = async () => {
    let blob;
    $("mGo").disabled = true; $("mGo").textContent = "Trying…";
    const reenable = () => { $("mGo").disabled = false; $("mGo").textContent = "Try again"; };
    try {
      const r = await jfetch(api(`/api/jobs/${jobId}/files/${fileIndex}/image?kind=original`));
      if (!r.ok) { reenable(); return; }        // e.g. the orig vanished; the row shows it
      blob = await r.blob();
    } catch (e) { reenable(); return; }
    // Same inheritances as the main form (an earlier round dropped
    // keepProvenance here, silently stripping metadata on retried files).
    const o = Object.assign(modeOptions("rmarkmode"), {
      denoise: $("denoise").value, keepProvenance: $("keepprov").checked });
    const fd = new FormData();
    fd.append("files", blob, fileName);
    fd.append("options", JSON.stringify(o));
    try {
      const r = await jfetch(api("/api/jobs"), { method: "POST", body: fd });
      if (!r.ok) { alert(((await r.json()).error || {}).message || "upload rejected"); reenable(); return; }
    } catch (e) { reenable(); return; }
    $("manual").close();
    await refreshJobs(); startPolling();
  };
  $("mCancel").onclick = () => $("manual").close();
  $("manual").showModal();
}
init();
