/* MasterEngine web panel — Pyodide bridge to the Python engine. */
const $ = s => document.querySelector(s);
const status = (s, err) => { const el = $("#status"); el.textContent = s; el.className = err ? "err" : ""; };

let PY = null, engineReady = false;
let audioLoaded = false, sr = 0, nSamples = 0, lastRes = null;

const GLUE = `
import sys, os, json, wave
sys.path.append("/engine")
os.makedirs("/session", exist_ok=True)
import numpy as np

_X = None
_SR = 44100.0
STEMS = None

def set_audio(buf, n, sr_):
    global _X, _SR, STEMS
    _SR = float(sr_)
    _X = np.frombuffer(buf, dtype=np.float32).astype(np.float64).reshape(2, int(n)).T.copy()
    STEMS = None

def _wav(path, y, sr_):
    y = np.clip(y, -1.0, 1.0)
    w = wave.open(path, "wb")
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(int(sr_))
    w.writeframes((y * 32767).astype("<i2").tobytes())
    w.close()

def run_master(cfg_json):
    from masterengine import master, stems, analysis
    global STEMS
    cfg = json.loads(cfg_json)
    res, y = master.process(_X, _SR, params=cfg.get("params"),
                            refine_iters=int(cfg.get("iters", 6)), save_experience=True)
    _wav("/session/mastered.wav", y, _SR)
    out = {"before": res["before"], "after": res["after"], "params": res["params"],
           "loss": res["loss"], "pre_gain_db": res["pre_gain_db"],
           "frames": ["/session/mastered.wav"]}
    if cfg.get("stems"):
        x = _X
        if STEMS is None:
            raw = stems.separate(x, _SR)
            STEMS = {k: v.astype(np.float64) for k, v in raw.items()}
        sm = {}
        mix = np.zeros_like(x)
        for name, s in STEMS.items():
            f_in = analysis.analyze(s, _SR)
            p = stems.build_stem_params(name, f_in)
            ys = master.apply_chain(s, _SR, p)
            ft = analysis.analyze(ys, _SR)
            for _ in range(2):
                p = master.MODEL.refine_step(p, ft)
                ys = master.safe_render(s, _SR, p)
                ft = analysis.analyze(ys, _SR)
            _wav(f"/session/stem_{name}.wav", s, _SR)
            _wav(f"/session/stemm_{name}.wav", ys, _SR)
            sm[name] = {"before": master.public_feats(f_in), "after": master.public_feats(ft), "params": p}
            out["frames"].append(f"/session/stemm_{name}.wav")
            mix += ys
        _wav("/session/stem_mix.wav", mix, _SR)
        out["frames"].append("/session/stem_mix.wav")
        out["stems"] = sm
    _wav("/session/original.wav", _X, _SR)
    out["frames"].append("/session/original.wav")
    return json.dumps(out)
`;

async function boot() {
  try {
    status("Гружу рантайм Pyodide (~25 МБ, один раз на вкладку)…");
    const pyodide = await loadPyodide({ indexURL: "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/" });
    status("Гружу numpy + scipy (WASM)…");
    await pyodide.loadPackage(["numpy", "scipy"]);
    status("Гружу движок MasterEngine…");
    const files = ["__init__.py", "dsp.py", "analysis.py", "ml.py", "master.py", "stems.py"];
    pyodide.FS.mkdirTree("/engine");
    for (const f of files) {
      const r = await fetch("masterengine/" + f);
      if (!r.ok) throw new Error("fetch " + f + ": " + r.status);
      pyodide.FS.writeFile("/engine/" + f, new Uint8Array(await r.arrayBuffer()));
    }
    await pyodide.runPythonAsync(GLUE);
    PY = pyodide;
    engineReady = true;
    $("#stemsOn").disabled = false;
    $("#drop").classList.add("ready");
    status("Движок готов. Бросай трек.");
  } catch (e) {
    status("Не удалось запустить движок: " + e.message, true);
  }
}
boot();

const drop = $("#drop");
drop.onclick = () => engineReady && $("#file").click();
drop.ondragover = e => { e.preventDefault(); drop.classList.add("on"); };
drop.ondragleave = () => drop.classList.remove("on");
drop.ondrop = e => { e.preventDefault(); drop.classList.remove("on"); if (engineReady && e.dataTransfer.files[0]) load(e.dataTransfer.files[0]); };
$("#file").onchange = e => { if (e.target.files[0]) load(e.target.files[0]); };

async function load(file) {
  try {
    status("Декодирую " + file.name + "…");
    const buf = await file.arrayBuffer();
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const ab = await ctx.decodeAudioData(buf);
    sr = ab.sampleRate;
    const L = ab.getChannelData(0);
    const R = ab.numberOfChannels > 1 ? ab.getChannelData(1) : L;
    nSamples = ab.length;
    const interleaved = new Float32Array(nSamples * 2);
    interleaved.set(L, 0); interleaved.set(R, nSamples);
    await new Promise(r => setTimeout(r, 30));
    PY.globals.set("me_in", new Uint8Array(interleaved.buffer));
    await PY.runPythonAsync(`set_audio(me_in.to_bytes(), ${nSamples}, ${sr})`);
    audioLoaded = true;
    status("Трек " + Math.round(ab.duration) + "s · " + sr + " Hz. Мастерю (не закрывай вкладку)…");
    await new Promise(r => setTimeout(r, 30));
    await process({ stems: $("#stemsOn").checked });
  } catch (e) {
    status("Ошибка: " + e.message, true);
  }
}

async function process(opts) {
  if (!audioLoaded) return;
  $("#rerun").disabled = true;
  PY.globals.set("me_cfg", JSON.stringify(opts));
  const out = PY.runPython("run_master(me_cfg)");
  lastRes = JSON.parse(out);
  $("#rerun").disabled = false;
  status("Готово ✅ Жми A ⇄ B и слушай в наушниках. loss = " + lastRes.loss.toFixed(2));
  show();
}

function fmt(v) { return typeof v === "number" ? (Math.abs(v) > 1000 ? v.toFixed(0) : v.toFixed(2)) : v; }

function show() {
  $("#results").style.display = "block";
  const mt = $("#metrics"); mt.innerHTML = "<tr><th></th><th>До</th><th>После</th></tr>";
  const M = [["lufs", "LUFS (громкость)"], ["peak_dbtp", "True Peak, dBTP"], ["centroid_hz", "Спектр. центр, Hz"], ["stereo_width_db", "Стерео-ширина, dB"], ["crest_db", "Крест-фактор, dB"]];
  for (const [k, l] of M) {
    const a = lastRes.before[k] ?? 0, b = lastRes.after[k] ?? 0;
    mt.insertAdjacentHTML("beforeend", `<tr><td>${l}</td><td>${fmt(a)}</td><td class="${b > a ? 'good' : 'warn'}">${fmt(b)}</td></tr>`);
  }
  $("#meters").innerHTML = `<div class="meter"><div class="v">${lastRes.after.lufs.toFixed(1)}</div><div class="l">После · LUFS</div></div>
    <div class="meter"><div class="v">${lastRes.after.peak_dbtp.toFixed(1)}</div><div class="l">После · dBTP</div></div>
    <div class="meter"><div class="v">${lastRes.loss.toFixed(2)}</div><div class="l">Loss сходимости</div></div>`;

  $("#au0").src = blobOf("/session/original.wav");
  $("#au1").src = blobOf("/session/mastered.wav");
  drawWave(bufOf("/session/original.wav"), 0);
  drawWave(bufOf("/session/mastered.wav"), 1);
  buildParams(lastRes.params);
  drawStems();
  downloads();
  setTimeout(spectrum, 100);
  $("#stemsCard").style.display = lastRes.stems ? "block" : "none";
}

const urlCache = {};
function blobOf(path) {
  if (!urlCache[path]) { urlCache[path] = URL.createObjectURL(new Blob([bufOf(path)], { type: "audio/wav" })); }
  return urlCache[path];
}
function bufOf(path) { return PY.FS.readFile(path); }

function buildParams(p) {
  const box = $("#params"); box.innerHTML = "";
  addSl(box, "Цель громкости (LUFS)", p.target_lufs, -22, -8, 0.5, v => p.target_lufs = v);
  addSl(box, "Потолок (dBTP)", p.ceiling_dbtp, -2.5, -0.2, 0.1, v => p.ceiling_dbtp = v);
  addSl(box, "Стерео-ширина", p.stereo_width, 0.6, 1.7, 0.05, v => p.stereo_width = v);
  addSl(box, "HPF, Hz", p.hpf_hz, 16, 80, 1, v => p.hpf_hz = v);
  addSl(box, "Сатурация", p.saturation, 0, 0.8, 0.02, v => p.saturation = v);
  addSl(box, "Эксайтер", p.exciter.mix, 0, 0.35, 0.01, v => p.exciter.mix = v);
  addSl(box, "Де-эссер", p.de_ess.amount, 0, 0.8, 0.05, v => p.de_ess.amount = v);
  const h = document.createElement("div");
  h.textContent = "EQ по полосам (dB):"; h.style.cssText = "color:var(--dim);margin:10px 0 4px";
  box.appendChild(h);
  for (const k of Object.keys(p.band_gains_db).sort((a, b) => parseFloat(a) - parseFloat(b)))
    addSl(box, k + " Hz", p.band_gains_db[k], -6, 6, 0.25, v => p.band_gains_db[k] = v);
}
function addSl(box, label, val, min, max, step, set) {
  const l = document.createElement("label"); l.innerHTML = `<span>${label}</span>`;
  const i = document.createElement("input"); i.type = "range"; i.min = min; i.max = max; i.step = step; i.value = val;
  const s = document.createElement("span"); s.textContent = (+val).toFixed(2); s.style.cssText = "min-width:52px;text-align:right";
  i.oninput = () => { s.textContent = (+i.value).toFixed(2); set(+i.value); };
  l.append(i, s); box.appendChild(l);
}

$("#rerun").onclick = async () => {
  status("Пересчитываю с твоими настройками…");
  await new Promise(r => setTimeout(r, 30));
  await process({ params: lastRes.params, stems: false });
};

const N2RU = { drums: "🥁 Барабаны", bass: "🎸 Бас", vocals: "🎤 Вокал", other: "🎹 Остальное" };
function drawStems() {
  const L = $("#stemsList"); if (!lastRes.stems) { L.innerHTML = ""; return; }
  L.innerHTML = "";
  for (const [k, v] of Object.entries(lastRes.stems)) {
    const d = document.createElement("div"); d.className = "stem";
    d.innerHTML = `<h3>${N2RU[k] || k}</h3>
      <div style="display:flex;gap:10px;flex-wrap:wrap">
        <div class="pl"><h4>Сырой стем</h4><audio controls src="${blobOf("/session/stem_" + k + ".wav")}"></audio></div>
        <div class="pl"><h4>Мастер</h4><audio controls src="${blobOf("/session/stemm_" + k + ".wav")}"></audio></div>
      </div>
      <small style="color:var(--dim)">LUFS ${v.before.lufs.toFixed(1)} → <b class="good">${v.after.lufs.toFixed(1)}</b> · TP ${v.before.peak_dbtp.toFixed(1)} → ${v.after.peak_dbtp.toFixed(1)} dBTP</small>`;
    L.appendChild(d);
  }
}

function downloads() {
  const d = $("#dls"); d.innerHTML = "";
  const items = [["Мастер", "/session/mastered.wav"], ["Оригинал", "/session/original.wav"]];
  if (lastRes.stems) items.unshift(["Стем-микс", "/session/stem_mix.wav"]);
  for (const [name, p] of items)
    d.insertAdjacentHTML("beforeend", `<a class="dl" href="${blobOf(p)}" download="${name.replace(/ /g, "_")}.wav">⬇ ${name}</a>`);
  for (const k of Object.keys(lastRes.stems || {}))
    d.insertAdjacentHTML("beforeend", `<a class="dl" href="${blobOf("/session/stemm_" + k + ".wav")}" download="stem_${k}_master.wav">⬇ ${N2RU[k]} (мастер)</a>`);
}

let isA = false;
$("#ab").onclick = () => {
  const a = $("#au0"), b = $("#au1");
  if (!isA) { b.currentTime = a.currentTime || 0; b.play(); a.pause(); isA = true; }
  else { a.currentTime = b.currentTime || 0; a.play(); b.pause(); isA = false; }
};

function drawWave(u8, idx) {
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const dataOff = 44;
  const nSmp = Math.floor((u8.length - dataOff) / 4);
  const n = 600, blk = Math.max(1, Math.floor(nSmp / n));
  const pk = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let m = 0; const off = dataOff + i * blk * 4;
    for (let j = 0; j < blk && off + j * 4 + 3 < u8.length; j++) {
      const v = Math.abs(dv.getInt16(off + j * 4, true) + dv.getInt16(off + j * 4 + 2, true)) / 65534;
      if (v > m) m = v;
    }
    pk[i] = m;
  }
  const c = document.createElement("canvas"); c.width = 1000; c.height = 44; c.style.height = "44px"; c.style.marginBottom = "6px";
  const g = c.getContext("2d"); g.fillStyle = idx ? "#31d0aa" : "#38507a";
  pk.forEach((v, i) => { const h = Math.min(1, v) * 42; g.fillRect(i * (1000 / n), (44 - h) / 2, 1000 / n, h); });
  $("#waves").appendChild(c);
}

function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let j2 = 0; j2 < len / 2; j2++) {
        const ur = re[i + j2], ui = im[i + j2];
        const xr = re[i + j2 + len / 2], xi = im[i + j2 + len / 2];
        const vr = xr * cr - xi * ci, vi = xr * ci + xi * cr;
        re[i + j2] = ur + vr; im[i + j2] = ui + vi;
        re[i + j2 + len / 2] = ur - vr; im[i + j2 + len / 2] = ui - vi;
        const ncr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = ncr;
      }
    }
  }
}

async function spectrum() {
  try {
    const a0 = bufOf("/session/original.wav"), a1 = bufOf("/session/mastered.wav");
    const dv0 = new DataView(a0.buffer, a0.byteOffset, a0.byteLength);
    const dv1 = new DataView(a1.buffer, a1.byteOffset, a1.byteLength);
    const readStereo = (dv, t) => { const off = 44 + Math.min(t, dv.byteLength / 4 - 1) * 4; return [dv.getInt16(off, true), dv.getInt16(off + 2, true)]; };
    const N = 4096, frames = 24, step = Math.max(N, Math.floor((a0.length - 44) / 4 / frames));
    const bands = n => { const f = (n * sr / 2 / (N / 2)); return Math.max(0, Math.min(55, Math.floor(Math.log2(f / 31) * 10))); };
    const spec = dv => {
      const out = new Float32Array(56).fill(-120);
      const w = new Float32Array(N); for (let i = 0; i < N; i++) w[i] = 0.54 - 0.46 * Math.cos(2 * Math.PI * i / (N - 1));
      for (let k = 0; k < frames; k++) {
        const base = 44 + Math.min(k * step * 2, dv.byteLength - N * 8);
        if (base < 0) break;
        const re = new Float32Array(N), im = new Float32Array(N);
        for (let i = 0; i < N; i++) {
          const o = base + i * 4;
          if (o + 3 >= dv.byteLength) break;
          re[i] = ((dv.getInt16(o, true) + dv.getInt16(o + 2, true)) / 65534) * w[i];
        }
        fft(re, im);
        for (let i = 1; i < N / 2; i++) {
          const b = bands(i);
          const m = 20 * Math.log10(Math.hypot(re[i], im[i]) / (N / 4) + 1e-9);
          if (m > out[b]) out[b] = m;
        }
      }
      return out;
    };
    const A = spec(dv0), B = spec(dv1);
    const c = $("#spec"), g = c.getContext("2d");
    g.fillStyle = "#0d1220"; g.fillRect(0, 0, c.width, c.height);
    const nb = A.length, W = c.width / nb;
    for (let i = 0; i < nb; i++) {
      const hA = Math.max(2, (A[i] + 100) / 100 * c.height), hB = Math.max(2, (B[i] + 100) / 100 * c.height);
      g.fillStyle = "#38507a"; g.fillRect(i * W + 1, c.height - hA, W - 2, hA);
      g.fillStyle = "#31d0aa"; g.fillRect(i * W + 1, c.height - hB, W - 2, hB);
    }
  } catch (e) { /* спектр — косметика */ }
}
