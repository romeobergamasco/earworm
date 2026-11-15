// Earworm PWA: capture ~10 s of microphone, resample to 11 025 Hz in the browser,
// send 16-bit PCM to /api/identify, show the result. No framework.

const API = (window.EARWORM_API || "").replace(/[/]+$/, ""); // "" = same origin
const MODEL_RATE = 11025;
const SECONDS = 10;
const DEBUG = new URLSearchParams(location.search).has("debug");

const $ = (id) => document.getElementById(id);
const ui = {
  listen: $("listen"), label: $("listen-label"), progress: $("progress"),
  meter: $("meter"), meterFill: $("meter-fill"), status: $("status"),
  result: $("result"), cover: $("cover"), title: $("song-title"), artist: $("song-artist"),
  year: $("song-year"), link: $("song-link"), notfound: $("notfound"), retry: $("retry"),
  debug: $("debug"), file: $("file"), debugOut: $("debug-out"),
  kind: $("result-kind"), back: $("back"), backLabel: $("back-label"), coverPh: $("cover-ph"),
  genre: $("genre"), genreName: $("genre-name"), genreProba: $("genre-proba"),
  genreBar: $("genre-bar"), genreOthers: $("genre-others"),
  similar: $("similar"), similarList: $("similar-list"),
  propose: $("propose"), proposeForm: $("propose-form"), pUrl: $("p-url"), pTitle: $("p-title"),
  pArtist: $("p-artist"), proposeMsg: $("propose-msg"), proposeSend: $("propose-send"),
};
const RING = 2 * Math.PI * 54;
let busy = false;
// a cover that fails to load leaves the gradient placeholder, never a broken image
ui.cover.addEventListener("error", () => showCover(false));

// ------------------------------------------------------------------ UI states
function setStatus(text) { ui.status.textContent = text; }
function setProgress(fraction) { ui.progress.style.strokeDashoffset = String(RING * (1 - fraction)); }
function setLevel(rms) {
  const db = 20 * Math.log10(Math.max(rms, 1e-6)); // -120 .. 0
  ui.meterFill.style.width = `${Math.max(0, Math.min(100, (db + 60) * (100 / 60)))}%`;
}
function setBusy(on, label) {
  busy = on;
  ui.listen.disabled = on;
  ui.listen.classList.toggle("busy", on);
  ui.label.textContent = label ?? (on ? "…" : "Écouter");
  ui.meter.classList.toggle("on", on);
  if (!on) setProgress(0);
}
function hideResults() {
  ui.result.hidden = true;
  ui.notfound.hidden = true;
  if (!ui.propose.hidden) history.back(); // closes the proposal form (popstate)
}
function debugLog(obj) { if (DEBUG) ui.debugOut.textContent = JSON.stringify(obj, null, 2); }
const pct = (p) => `${Math.round(p * 100)} %`;

// ------------------------------------------------------------------ song page
// views: the recognised song first, then the similar songs opened from it
let views = [];

function showSong(song) {
  ui.title.textContent = song.title;
  ui.artist.textContent = song.artist;
  ui.year.textContent = song.year ? String(song.year) : "";
  ui.year.hidden = !song.year;
  ui.link.href = `https://www.youtube.com/watch?v=${encodeURIComponent(song.youtube_id)}`;
  if (song.cover_url) {
    ui.cover.src = song.cover_url;
    ui.cover.alt = `Pochette : ${song.title}`;
  } else {
    ui.cover.removeAttribute("src");
    ui.cover.alt = "";
  }
  showCover(Boolean(song.cover_url));
}
// no cover, or one that fails to load: a placeholder, never a broken-image icon
function showCover(ok) {
  ui.cover.hidden = !ok;
  ui.coverPh.hidden = ok;
}

function showGenre(genre) {
  ui.genre.hidden = !genre;
  if (!genre) return;
  ui.genreName.textContent = genre.name || genre.label;
  ui.genreProba.textContent = pct(genre.proba);
  ui.genreBar.style.width = `${Math.round(genre.proba * 100)}%`;
  const others = (genre.top3 || []).slice(1).filter((g) => g.proba >= 0.05);
  ui.genreOthers.textContent = others.length
    ? "sinon : " + others.map((g) => `${g.name || g.label} ${pct(g.proba)}`).join(", ")
    : "";
  ui.genreOthers.hidden = !others.length;
}

function thumb(song) {
  if (song.cover_url) {
    const img = document.createElement("img");
    img.className = "thumb";
    img.src = song.cover_url;
    img.alt = "";
    img.loading = "lazy";
    img.addEventListener("error", () => img.replaceWith(placeholder()));
    return img;
  }
  return placeholder();
}
function placeholder() {
  const span = document.createElement("span");
  span.className = "thumb";
  span.textContent = "♪";
  span.setAttribute("aria-hidden", "true");
  return span;
}

function showSimilar(similar) {
  ui.similarList.replaceChildren();
  ui.similar.hidden = !similar || similar.length === 0;
  for (const s of similar || []) {
    const btn = document.createElement("button");
    btn.type = "button";
    const text = document.createElement("span");
    text.className = "sim-text";
    const title = document.createElement("span");
    title.className = "sim-title";
    title.textContent = s.title;
    const sub = document.createElement("span");
    sub.className = "sim-sub";
    sub.textContent = s.year ? `${s.artist} · ${s.year}` : s.artist;
    text.append(title, sub);
    btn.append(thumb(s), text);
    btn.setAttribute("aria-label", `${s.title}, ${s.artist} : ouvrir sa fiche`);
    btn.addEventListener("click", () => openSong(s.id));
    const li = document.createElement("li");
    li.append(btn);
    ui.similarList.append(li);
  }
}

/** Render a page: {song, genre, similar}. kind: "found" | "similar". */
function render(page, kind) {
  showSong(page.song);
  showGenre(page.genre);
  showSimilar(page.similar);
  ui.kind.textContent = kind === "found" ? "Trouvé" : "Morceau similaire";
  const prev = views.length > 1 ? views[views.length - 2].song : null;
  ui.back.hidden = !prev;
  if (prev) ui.backLabel.textContent = prev.title;
  ui.notfound.hidden = true;
  ui.result.hidden = false;
}

function showFound(answer) {
  views = [answer];
  render(answer, "found");
}

async function openSong(id) {
  if (busy) return;
  setStatus("Chargement…");
  try {
    const res = await fetch(`${API}/api/songs/${encodeURIComponent(id)}`);
    if (!res.ok) throw new UserError(`Erreur du serveur (${res.status}).`);
    const page = await res.json();
    views.push(page);
    history.pushState({ depth: views.length }, "");
    render(page, "similar");
    setStatus("");
    ui.result.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) {
    setStatus(e instanceof UserError ? e.message : "Pas de connexion au serveur. Vérifiez le réseau.");
  }
}

function goBack() {
  if (views.length < 2) return;
  views.pop();
  render(views[views.length - 1], views.length === 1 ? "found" : "similar");
}
ui.back.addEventListener("click", () => history.back());
// the phone's back button (or swipe) closes the proposal form, then a similar song's page
window.addEventListener("popstate", () => (ui.propose.hidden ? goBack() : closePropose()));

// ------------------------------------------------------------------ proposals
// same shape as the server's checks (earworm/youtube.py); the server decides
const YOUTUBE = /^(https?:\/\/)?((www\.|m\.|music\.)?youtube\.com\/(watch\?|shorts\/)|(www\.)?youtu\.be\/)/i;

function openPropose() {
  if (!ui.propose.hidden) { ui.pUrl.focus(); return; }
  ui.notfound.hidden = true;
  ui.propose.hidden = false;
  history.pushState({ propose: true }, "");
  ui.propose.scrollIntoView({ behavior: "smooth", block: "start" });
  ui.pUrl.focus({ preventScroll: true });
}
function closePropose() {
  ui.propose.hidden = true;
  setProposeMsg("");
}
function setProposeMsg(text, kind) {
  ui.proposeMsg.textContent = text;
  ui.proposeMsg.className = `form-msg${kind ? " " + kind : ""}`;
}

const PROPOSE_ERRORS = {
  409: (detail) => detail.includes("catalogue")
    ? "Ce morceau est déjà dans le catalogue : essayez de le reconnaître !"
    : "Ce morceau a déjà été proposé, il sera bientôt examiné.",
  422: () => "Ce lien n'est pas une vidéo YouTube. Copiez l'adresse de la vidéo depuis YouTube.",
  429: () => "Vous avez fait beaucoup de propositions. Réessayez dans une heure.",
  503: () => "Les propositions sont fermées pour le moment.",
};

async function sendProposal(event) {
  event.preventDefault();
  const url = ui.pUrl.value.trim();
  const bad = !YOUTUBE.test(url);
  ui.pUrl.setAttribute("aria-invalid", String(bad));
  if (bad) {
    setProposeMsg(url ? PROPOSE_ERRORS[422]() : "Collez d'abord le lien YouTube du morceau.", "err");
    ui.pUrl.focus();
    return;
  }
  ui.proposeSend.disabled = true;
  setProposeMsg("Envoi…");
  try {
    const res = await fetch(`${API}/api/proposals`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, title: ui.pTitle.value.trim() || null, artist: ui.pArtist.value.trim() || null }),
    });
    if (res.status === 201) {
      ui.proposeForm.reset();
      setProposeMsg("Merci, votre proposition sera examinée.", "ok");
      return;
    }
    const detail = String((await res.json().catch(() => ({}))).detail || "");
    const known = PROPOSE_ERRORS[res.status];
    setProposeMsg(known ? known(detail) : `Erreur du serveur (${res.status}).`, "err");
    if (res.status === 422) ui.pUrl.setAttribute("aria-invalid", "true");
  } catch {
    setProposeMsg("Pas de connexion au serveur. Vérifiez le réseau.", "err");
  } finally {
    ui.proposeSend.disabled = false;
  }
}

$("open-propose").addEventListener("click", openPropose);
$("notfound-propose").addEventListener("click", openPropose);
$("propose-close").addEventListener("click", () => history.back());
ui.proposeForm.addEventListener("submit", sendProposal);
ui.pUrl.addEventListener("input", () => ui.pUrl.removeAttribute("aria-invalid"));

// ------------------------------------------------------------------ capture
const COLLECTOR = `
class Collector extends AudioWorkletProcessor {
  constructor() { super(); this.buf = []; this.n = 0; }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) {
      this.buf.push(ch.slice(0));
      this.n += ch.length;
      if (this.n >= 2048) {
        const out = new Float32Array(this.n);
        let o = 0;
        for (const b of this.buf) { out.set(b, o); o += b.length; }
        this.port.postMessage(out, [out.buffer]);
        this.buf = []; this.n = 0;
      }
    }
    return true;
  }
}
registerProcessor("collector", Collector);`;

async function captureMicrophone(seconds, onProgress, onLevel) {
  if (!navigator.mediaDevices?.getUserMedia) throw new UserError("Ce navigateur ne donne pas accès au micro.");
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
    });
  } catch (e) {
    if (e.name === "NotAllowedError") throw new UserError("Accès au micro refusé. Autorisez-le dans les réglages du navigateur.");
    if (e.name === "NotFoundError") throw new UserError("Aucun micro détecté.");
    throw e;
  }
  const ctx = new AudioContext();
  try {
    const url = URL.createObjectURL(new Blob([COLLECTOR], { type: "application/javascript" }));
    await ctx.audioWorklet.addModule(url);
    URL.revokeObjectURL(url);
    const source = ctx.createMediaStreamSource(stream);
    const node = new AudioWorkletNode(ctx, "collector");
    const mute = ctx.createGain();
    mute.gain.value = 0; // keeps the graph running without playing anything back
    source.connect(node).connect(mute).connect(ctx.destination);

    const wanted = Math.round(seconds * ctx.sampleRate);
    const samples = new Float32Array(wanted);
    let filled = 0;
    await new Promise((resolve) => {
      node.port.onmessage = ({ data }) => {
        const take = Math.min(data.length, wanted - filled);
        samples.set(data.subarray(0, take), filled);
        filled += take;
        let sum = 0;
        for (let i = 0; i < data.length; i++) sum += data[i] * data[i];
        onLevel(Math.sqrt(sum / data.length));
        onProgress(filled / wanted);
        if (filled >= wanted) resolve();
      };
    });
    return { samples, rate: ctx.sampleRate };
  } finally {
    stream.getTracks().forEach((t) => t.stop());
    ctx.close();
  }
}

// ------------------------------------------------------------------ processing
async function toModelRate(samples, rate) {
  const length = Math.ceil((samples.length * MODEL_RATE) / rate);
  const offline = new OfflineAudioContext(1, length, MODEL_RATE);
  const buffer = offline.createBuffer(1, samples.length, rate);
  buffer.copyToChannel(samples, 0);
  const src = offline.createBufferSource();
  src.buffer = buffer;
  src.connect(offline.destination);
  src.start();
  return (await offline.startRendering()).getChannelData(0);
}

function toPCM16(f32) {
  const view = new DataView(new ArrayBuffer(f32.length * 2));
  for (let i = 0; i < f32.length; i++) {
    const s = Math.max(-1, Math.min(1, f32[i]));
    view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return view.buffer;
}

class UserError extends Error {}

async function identify(pcm) {
  for (let attempt = 0; attempt < 4; attempt++) {
    let res;
    try {
      res = await fetch(`${API}/api/identify`, {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream", "X-Sample-Rate": String(MODEL_RATE) },
        body: pcm,
      });
    } catch {
      throw new UserError("Pas de connexion au serveur. Vérifiez le réseau.");
    }
    if (res.status === 503) { // index still loading (server just woke up)
      setStatus("Réveil du serveur…");
      await new Promise((r) => setTimeout(r, 3000));
      continue;
    }
    if (!res.ok) throw new UserError(`Erreur du serveur (${res.status}).`);
    return res.json();
  }
  throw new UserError("Le serveur met trop de temps à démarrer. Réessayez dans une minute.");
}

// ------------------------------------------------------------------ flow
async function run(getAudio) {
  if (busy) return;
  hideResults();
  setBusy(true, "Écoute…");
  try {
    const { samples, rate } = await getAudio();
    const tEnd = performance.now();
    setBusy(true, "…");
    setStatus("Recherche…");
    const pcm = toPCM16(await toModelRate(samples, rate));
    const tSent = performance.now();
    const answer = await identify(pcm);
    const tShown = performance.now();
    if (answer.match) {
      showFound(answer);
      setStatus("Trouvé !");
    } else {
      ui.notfound.hidden = false;
      setStatus("");
    }
    debugLog({
      latency_ms: Math.round(tShown - tEnd),
      resample_ms: Math.round(tSent - tEnd),
      network_and_server_ms: Math.round(tShown - tSent),
      server: answer.timing_ms,
      confidence: answer.confidence,
      capture_rate: rate,
    });
  } catch (e) {
    setStatus(e instanceof UserError ? e.message : "Une erreur est survenue. Réessayez.");
    if (!(e instanceof UserError)) console.error(e);
  } finally {
    setBusy(false);
  }
}

const fromMicrophone = () =>
  captureMicrophone(SECONDS, setProgress, setLevel);

async function fromFile(file) {
  // debug: a file instead of the microphone, 10 s from one third in
  const ctx = new AudioContext();
  try {
    const audio = await ctx.decodeAudioData(await file.arrayBuffer());
    const data = audio.getChannelData(0);
    const start = Math.floor(data.length / 3);
    return { samples: data.slice(start, start + SECONDS * audio.sampleRate), rate: audio.sampleRate };
  } finally {
    ctx.close();
  }
}

ui.listen.addEventListener("click", () => run(fromMicrophone));
ui.retry.addEventListener("click", () => run(fromMicrophone));
if (DEBUG) {
  ui.debug.hidden = false;
  ui.file.addEventListener("change", () => ui.file.files[0] && run(() => fromFile(ui.file.files[0])));
}

// wake the server early (free hosting sleeps when idle) and say so if it is slow
(async () => {
  const slow = setTimeout(() => setStatus("Réveil du serveur…"), 2000);
  try {
    const health = await (await fetch(`${API}/api/health`)).json();
    if (health.status !== "ok") setStatus("Le catalogue se charge, un instant…");
    else if (!busy) setStatus("Approchez le téléphone de la musique, puis appuyez.");
  } catch { /* offline: the shell still works, errors are shown on use */ }
  clearTimeout(slow);
})();

if ("serviceWorker" in navigator) navigator.serviceWorker.register("sw.js").catch(() => {});

// exposed for automated checks in debug mode
if (DEBUG) window.earworm = { toModelRate, toPCM16, identify, MODEL_RATE, showFound, openSong, openPropose };
