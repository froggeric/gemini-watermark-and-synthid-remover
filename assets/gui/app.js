"use strict";
const $ = (id) => document.getElementById(id);
const TOKEN = location.pathname.split("/")[1];        // URL: /<token>/...
const api = (p) => `/${TOKEN}${p}`;
let offline = false, pollTimer = null;

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
  if (pollTimer) clearInterval(pollTimer);
}

async function init() {
  try {
    const v = await (await jfetch(api("/api/version"))).json();
    $("version").textContent = "v" + v.version;
    v.presets.forEach(p => {
      for (const sel of [$("preset"), $("mPreset")]) {
        const o = document.createElement("option"); o.textContent = p; o.value = p;
        sel.appendChild(o);
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
  await refreshJobs(); startPolling();       // reload recovery: re-attach AND resume polling
                                             // (startPolling is idempotent; refreshJobs clears it when idle)
  $("drop").addEventListener("dragover", e => { e.preventDefault(); });
  $("drop").addEventListener("drop", e => { e.preventDefault(); submit([...e.dataTransfer.files]); });
  $("browse").addEventListener("click", () => $("file").click());
  // Reset after reading: browsers fire no change event when the picker
  // re-selects the SAME files, which would silently ignore a re-run of the
  // same image with different options (the classic file-input gotcha).
  $("file").addEventListener("change", (e) => { submit([...e.target.files]); e.target.value = ""; });
  const combo = () => {                       // legacy/force disable the preset (400 combos)
    $("preset").disabled = $("legacy").checked || $("force").checked;
  };
  $("legacy").addEventListener("change", combo);
  $("force").addEventListener("change", combo);

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
  // the dialog, including its padding around the image, do not. Two guards:
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
  const o = { denoise: $("denoise").value, legacy: $("legacy").checked,
              forceRemove: $("force").checked, keepProvenance: $("keepprov").checked };
  const p = $("preset").value; if (p) o.geoPreset = p;
  return o;
}
const JOB_CAP_BYTES = 1073741824, JOB_CAP_FILES = 100;
async function submit(files) {
  if (offline || !files.length) return;
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
  for (const chunk of chunks) {
    const fd = new FormData();
    chunk.forEach(f => fd.append("files", f, f.name));
    fd.append("options", JSON.stringify(options()));
    try {
      const r = await jfetch(api("/api/jobs"), { method: "POST", body: fd });
      if (!r.ok) { alert(((await r.json()).error || {}).message || "upload rejected"); return; }
    } catch (e) { return; }
  }
  await refreshJobs(); startPolling();
}
function startPolling() { if (!pollTimer) pollTimer = setInterval(refreshJobs, 500); }

const BADGE = { pending: "…", queued: "queued", removed: "removed", "no-watermark": "no watermark found",
                failed: "failed", "not-run": "skipped" };
async function refreshJobs() {
  let jobs;
  try { jobs = await (await jfetch(api("/api/jobs"))).json(); }
  catch (e) { return; }
  const root = $("jobs"); root.replaceChildren();
  let anyRunning = false;
  for (const j of jobs) {
    let full;
    try { full = await (await jfetch(api("/api/jobs/" + j.job_id))).json(); }
    catch (e) { return; }
    if (full.status === "queued" || full.status === "running") anyRunning = true;
    root.appendChild(renderJob(full));
  }
  if (!anyRunning && pollTimer) { clearInterval(pollTimer); pollTimer = null; }
}
function renderJob(j) {
  const card = document.createElement("section"); card.className = "job";
  const head = document.createElement("h2"); head.textContent = `${j.status} (${j.files.length} file${j.files.length>1?"s":""})`;
  card.appendChild(head);
  if (j.status === "queued" || j.status === "running") {
    const c = document.createElement("button"); c.textContent = "Cancel";
    c.addEventListener("click", () => jfetch(api(`/api/jobs/${j.job_id}/cancel`), {method:"POST"}).then(refreshJobs));
    card.appendChild(c);
  }
  const list = document.createElement("ul");
  j.files.forEach((f, i) => {
    const li = document.createElement("li");
    const name = document.createElement("span"); name.textContent = f.name; li.appendChild(name);
    const badge = document.createElement("span"); badge.className = "badge " + (f.outcome || "pending");
    badge.textContent = f.outcome === "removed" && f.forced ? "removed (forced)" : BADGE[f.outcome || "pending"];
    li.appendChild(badge);
    if (f.outcome === "removed") {
      const a = document.createElement("a"); a.href = api(`/api/jobs/${j.job_id}/files/${i}/image?kind=cleaned`);
      a.download = ""; a.textContent = "Download"; li.appendChild(a);
      const cmp = document.createElement("button"); cmp.textContent = "Compare";
      cmp.addEventListener("click", () => openCompare(j, i, f)); li.appendChild(cmp);
      // The detected type: variant profile + how the position was resolved.
      const type = [f.variant, f.geometry_source].filter(Boolean).join(" · ");
      if (type) { const t = document.createElement("span"); t.className = "muted";
                  t.textContent = type; li.appendChild(t); }
    }
    if (f.outcome === "no-watermark") {
      const m = document.createElement("button"); m.textContent = "Mark manually";
      m.addEventListener("click", () => openManual(j.job_id, i, f.name)); li.appendChild(m);
    }
    if (f.error) { const e = document.createElement("span"); e.className = "err"; e.textContent = f.error; li.appendChild(e); }
    list.appendChild(li);
  });
  card.appendChild(list);
  return card;
}
function openCompare(j, i, f) {
  const a = $("cmpA"), box = $("cmpBox");
  $("cmpB").src = api(`/api/jobs/${j.job_id}/files/${i}/image?kind=cleaned`);
  a.onload = () => {
    // The mark region is drawn on the ORIGINAL layer only (a plain bordered
    // div positioned in percent: crisp and constant thickness at any size,
    // unlike the canvas stroke it replaces). The cleaned layer stacks above,
    // so the box never covers cleaned pixels to begin with.
    if (!f.bbox) { box.hidden = true; return; }
    const [x, y, w, h] = f.bbox;
    box.style.left = (x / a.naturalWidth * 100) + "%";
    box.style.top = (y / a.naturalHeight * 100) + "%";
    box.style.width = (w / a.naturalWidth * 100) + "%";
    box.style.height = (h / a.naturalHeight * 100) + "%";
    box.hidden = !$("cmpShowBox").checked;
  };
  a.src = api(`/api/jobs/${j.job_id}/files/${i}/image?kind=original`);
  $("cmpShowBox").onchange = (e) => { box.hidden = !e.target.checked; };
  $("cmpSlider").oninput = (e) =>
    $("cmpBwrap").style.clipPath = `inset(0 ${100 - e.target.value}% 0 0)`;
  $("cmpClose").onclick = () => $("compare").close();
  $("compare").showModal();
}

// Manual retry for a no-watermark file: an explicit type or position instead
// of the failed auto-detection. The server rejects preset combined with
// legacy or force (its 400 rules), so the preset disables while either
// checkbox is checked; legacy+force is the documented --force --legacy
// escape. Remove re-submits just this file as a new job.
function openManual(jobId, fileIndex, fileName) {
  $("manualTitle").textContent = fileName;
  $("mPreset").value = "";
  $("mLegacy").checked = false;
  $("mForce").checked = false;
  const combo = () => { $("mPreset").disabled = $("mLegacy").checked || $("mForce").checked; };
  $("mLegacy").onchange = combo;
  $("mForce").onchange = combo;
  combo();

  $("mGo").onclick = async () => {
    let blob;
    try {
      const r = await jfetch(api(`/api/jobs/${jobId}/files/${fileIndex}/image?kind=original`));
      if (!r.ok) return;                       // e.g. the orig vanished; the row shows it
      blob = await r.blob();
    } catch (e) { return; }
    const o = { denoise: $("denoise").value };      // inherit the cleanup choice only
    if (!$("mLegacy").checked && !$("mForce").checked && $("mPreset").value)
      o.geoPreset = $("mPreset").value;
    o.legacy = $("mLegacy").checked;
    o.forceRemove = $("mForce").checked;
    const fd = new FormData();
    fd.append("files", blob, fileName);
    fd.append("options", JSON.stringify(o));
    try {
      const r = await jfetch(api("/api/jobs"), { method: "POST", body: fd });
      if (!r.ok) { alert(((await r.json()).error || {}).message || "upload rejected"); return; }
    } catch (e) { return; }
    $("manual").close();
    await refreshJobs(); startPolling();
  };
  $("mCancel").onclick = () => $("manual").close();
  $("manual").showModal();
}
init();
