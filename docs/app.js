// Renders progress.json. Prefers the raw GitHub copy so the page is fresh before Pages redeploys.
const RAW = "https://raw.githubusercontent.com/SubToFeralsloth11/realistic-human/main/docs/";
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const isVideo = (src) => /\.(mp4|webm)$/i.test(src);

function ago(iso) {
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (!isFinite(s)) return iso;
  if (s < 90) return "just now";
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return new Date(iso).toLocaleString();
}
const when = (iso) => `<time datetime="${esc(iso)}" title="${esc(new Date(iso).toLocaleString())}">${esc(ago(iso))}</time>`;

// Media tag with a fallback to the raw copy when Pages has not deployed the file yet.
function media(src, { poster, controls = false, thumb = false } = {}) {
  const onerr = `if(!this.dataset.r){this.dataset.r=1;this.src='${RAW}${esc(src)}'}`;
  if (isVideo(src)) {
    const p = poster ? ` poster="${esc(poster)}"` : "";
    return `<video src="${esc(src)}"${p} muted loop playsinline ${thumb ? 'preload="metadata"' : "autoplay"} ${controls ? "controls" : ""} onerror="${onerr}"></video>`;
  }
  return `<img src="${esc(src)}" loading="lazy" alt="" onerror="${onerr}">`;
}

let DATA = null;

function render(d) {
  DATA = d;
  $("now-text").textContent = d.now || "";
  $("updated").innerHTML = when(d.updated);

  const items = d.phases.flatMap((p) => p.items);
  const done = items.filter((i) => i.status === "done").length;
  const active = items.filter((i) => i.status === "active").length;
  const pct = Math.round(((done + active * 0.4) / items.length) * 100);
  $("pct").textContent = `${pct}%`;
  $("bar").style.width = `${pct}%`;
  $("count").textContent = `${done} of ${items.length} done, ${active} in progress`;

  $("stats").innerHTML = Object.entries(d.stats || {}).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");

  const hero = (d.gallery || []).find((g) => g.hero) || (d.gallery || [])[0];
  if (hero) {
    $("hero-media").innerHTML = media(hero.src, { poster: hero.poster });
    $("hero-media").onclick = () => open(hero);
  }

  $("phases").innerHTML = d.phases.map((p) => {
    const n = p.items.filter((i) => i.status === "done").length;
    return `<div class="phase"><h3>${esc(p.name)} <small>${n}/${p.items.length}</small></h3>${p.items.map(itemHtml).join("")}</div>`;
  }).join("");

  const animPhase = d.phases.find((p) => p.name === "Animations");
  $("anims").innerHTML = animPhase ? animPhase.items.map((i) => {
    const vid = (i.media || []).find(isVideo) || (i.media || [])[0];
    const body = vid ? media(vid) : `<span>${i.status === "active" ? "in progress" : "not started"}</span>`;
    return `<div class="anim"><div class="frame" data-src="${esc(vid || "")}">${body}</div><div class="lbl"><span>${esc(i.name)}</span><span class="pill ${i.status}">${esc(i.status)}</span></div></div>`;
  }).join("") : "";

  const gal = d.gallery || [];
  $("gallery").innerHTML = gal.length ? gal.map((g, k) =>
    `<figure class="card" data-k="${k}"><div class="media">${media(g.src, { poster: g.poster, thumb: true })}</div><figcaption>${esc(g.caption)}${when(g.time)}</figcaption></figure>`
  ).join("") : `<p class="empty">No previews yet.</p>`;

  $("log").innerHTML = (d.log || []).map((l) =>
    `<li>${when(l.time)}${esc(l.text)}${l.commit ? ` <a href="https://github.com/SubToFeralsloth11/realistic-human/commit/${esc(l.commit)}">${esc(l.commit.slice(0, 7))}</a>` : ""}</li>`
  ).join("");

  document.querySelectorAll(".card").forEach((el) => (el.onclick = () => open(gal[+el.dataset.k])));
  document.querySelectorAll(".thumbs [data-src], .anim .frame[data-src]").forEach((el) => {
    if (!el.dataset.src) return;
    el.onclick = () => open(gal.find((g) => g.src === el.dataset.src) || { src: el.dataset.src, caption: "" });
  });
}

function itemHtml(i) {
  const thumbs = (i.media || []).slice(0, 4).map((m) =>
    isVideo(m) ? `<video data-src="${esc(m)}" src="${esc(m)}" muted preload="metadata"></video>` : `<img data-src="${esc(m)}" src="${esc(m)}" loading="lazy" alt="">`
  ).join("");
  return `<div class="item ${i.status}"><span class="st ${i.status}"></span><div><div class="name">${esc(i.name)}</div>${i.note ? `<div class="note">${esc(i.note)}</div>` : ""}${thumbs ? `<div class="thumbs">${thumbs}</div>` : ""}</div></div>`;
}

function open(g) {
  $("lb-body").innerHTML = media(g.src, { poster: g.poster, controls: true });
  $("lb-cap").textContent = g.caption || "";
  $("lightbox").hidden = false;
}
$("lb-close").onclick = () => { $("lightbox").hidden = true; $("lb-body").innerHTML = ""; };
$("lightbox").onclick = (e) => { if (e.target.id === "lightbox") $("lb-close").onclick(); };
document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("lb-close").onclick(); });

async function load() {
  for (const url of [`${RAW}progress.json?t=${Date.now()}`, `progress.json?t=${Date.now()}`]) {
    try {
      const r = await fetch(url, { cache: "no-store" });
      if (!r.ok) continue;
      const d = await r.json();
      if (!DATA || d.updated !== DATA.updated) render(d);
      return;
    } catch (e) { /* try next source */ }
  }
}
load();
setInterval(load, 60000);
