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
function showOffline() {
  if (offline) return; offline = true;
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
      const o = document.createElement("option"); o.textContent = p; o.value = p;
      $("preset").appendChild(o);
    });
    if (v.features && v.features.denoise_ai)
      document.querySelector('#denoise option[value=ai]').hidden = false;
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
  const combo = () => {                       // legacy/force disable rect+preset (400 combos)
    const off = $("legacy").checked || $("force").checked;
    ["rx","ry","rw","rh"].forEach(i => $(i).disabled = off);
    $("preset").disabled = off;
  };
  $("legacy").addEventListener("change", combo);
  $("force").addEventListener("change", combo);
}
function options() {
  const o = { denoise: $("denoise").value, legacy: $("legacy").checked,
              forceRemove: $("force").checked, keepProvenance: $("keepprov").checked };
  const p = $("preset").value; if (p) o.geoPreset = p;
  // Rect: send only when all four fields are filled; mirror the server grammar
  // client-side (x,y >= 0, w,h >= 8) to avoid a pointless 400 round trip.
  const ids = ["rx","ry","rw","rh"];
  if (ids.every(i => $(i).value.trim() !== "")) {
    const r = ids.map(i => Number($(i).value));
    if (r[0] >= 0 && r[1] >= 0 && r[2] >= 8 && r[3] >= 8) o.rect = r;
  }
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
    }
    if (f.error) { const e = document.createElement("span"); e.className = "err"; e.textContent = f.error; li.appendChild(e); }
    list.appendChild(li);
  });
  card.appendChild(list);
  return card;
}
function openCompare(j, i, f) {
  $("cmpA").src = api(`/api/jobs/${j.job_id}/files/${i}/image?kind=original`);
  $("cmpB").src = api(`/api/jobs/${j.job_id}/files/${i}/image?kind=cleaned`);
  $("cmpB").onload = () => {                       // bbox overlay once the cleaned image has its size
    const img = $("cmpB"), c = $("cmpBox");
    if (!f.bbox) { c.width = 0; return; }
    c.width = img.clientWidth; c.height = img.clientHeight;
    const s = c.width / img.naturalWidth, g = c.getContext("2d");
    g.strokeStyle = "#ff4d4d"; g.lineWidth = 2;
    g.strokeRect(f.bbox[0]*s, f.bbox[1]*s, f.bbox[2]*s, f.bbox[3]*s);
  };
  $("cmpSlider").oninput = (e) =>
    $("cmpBwrap").style.clipPath = `inset(0 ${100 - e.target.value}% 0 0)`;
  $("cmpClose").onclick = () => $("compare").close();
  $("compare").showModal();
}
init();
