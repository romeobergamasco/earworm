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
};
const RING = 2 * Math.PI * 54;
let busy = false;
// a cover that fails to load leaves the gradient placeholder, never a broken image
ui.cover.addEventListener("error", () => ui.cover.removeAttribute("src"));

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
function hideResults() { ui.result.hidden = true; ui.notfound.hidden = true; }
function debugLog(obj) { if (DEBUG) ui.debugOut.textContent = JSON.stringify(obj, null, 2); }

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
  ui.result.hidden = false;
}

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
      showSong(answer.song);
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
if (DEBUG) window.earworm = { toModelRate, toPCM16, identify, MODEL_RATE };
