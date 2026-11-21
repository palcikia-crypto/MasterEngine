/* MasterEngine worker — pure JS engine, everything off the main thread. */
importScripts("js-engine.js");
const E = self.MasterEngine;

const post = (m, tr) => self.postMessage(m, tr || []);
const say = (msg, pct) => post({ t: "status", msg, pct: pct == null ? null : pct });

self.onmessage = async (e) => {
  const d = e.data;
  if (d.t !== "load") return;
  try {
    const t0 = Date.now();
    const f32 = new Float32Array(d.buf);
    const n = d.n | 0;
    const x = E.planar64(n);
    x.L.set(f32.subarray(0, n));
    x.R.set(f32.subarray(n, 2 * n));
    say("Трек принят: " + n + " сэмплов, " + d.sr + " Hz");
    const main = await E.processAsync(x, d.sr, { params: d.params || null, iters: d.iters }, (msg, pct) => say(msg, pct == null ? null : pct));
    const files = {};
    const res = main.result;
    if (d.stems) {
      const onStatus = (msg) => say(msg);
      const ms = await E.masterStemsAsync(x, d.sr, onStatus);
      const mix = E.planar64(n);
      res.stems = {};
      for (const name of Object.keys(ms.stems)) {
        const st = ms.stems[name];
        res.stems[name] = {
          before: E.publicFeats(ms.before[name]),
          after: E.publicFeats(ms.after[name]),
          params: st.params,
        };
        files["/session/stem_" + name + ".wav"] = E.encodeWav(st.raw, d.sr).buffer;
        files["/session/stemm_" + name + ".wav"] = E.encodeWav(st.master, d.sr).buffer;
        for (let i = 0; i < n; i++) { mix.L[i] += st.master.L[i]; mix.R[i] += st.master.R[i]; }
      }
      files["/session/stem_mix.wav"] = E.encodeWav(mix, d.sr).buffer;
    }
    files["/session/mastered.wav"] = E.encodeWav(main.y, d.sr).buffer;
    files["/session/original.wav"] = E.encodeWav(x, d.sr).buffer;
    res.frames = Object.keys(files);
    say("Готово за " + ((Date.now() - t0) / 1000).toFixed(1) + " с.", null);
    post({ t: "result", res, files }, Object.values(files));
  } catch (err) {
    post({ t: "error", msg: String((err && err.stack) || err).split("\n").slice(-4).join(" | ") });
  }
};

post({ t: "ready" });
