/* MasterEngine worker: Pyodide runtime + heavy processing off the main thread. */
importScripts("https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js");

let py = null;
const post = (m, tr) => self.postMessage(m, tr || []);

async function boot() {
  try {
    post({ t: "status", msg: "Гружу рантайм Pyodide (~25 МБ) — один раз на страницу…", pct: 10 });
    py = await loadPyodide();
    post({ t: "status", msg: "Гружу numpy + scipy (WASM)…", pct: 45 });
    await py.loadPackage(["numpy", "scipy"]);
    post({ t: "status", msg: "Гружу движок MasterEngine…", pct: 85 });
    py.FS.mkdirTree("/engine/masterengine");
    for (const f of ["__init__.py", "dsp.py", "analysis.py", "ml.py", "master.py", "stems.py"]) {
      const r = await fetch("masterengine/" + f);
      if (!r.ok) throw new Error("fetch " + f + ": " + r.status);
      py.FS.writeFile("/engine/masterengine/" + f, new Uint8Array(await r.arrayBuffer()));
    }
    const glue = await (await fetch("engine_glue.py")).text();
    await py.runPythonAsync(glue);
    py.setStdout({ batched: (s) => {
      const m = s.match(/^\[ME\]\s*(.+)$/);
      if (m) post({ t: "status", msg: m[1], pct: null });
      const r = s.match(/^\[RESULT\](.*)$/);
      if (r) pendingResult = r[1];
    } });
    post({ t: "ready" });
  } catch (e) {
    post({ t: "error", msg: String(e.message || e) });
  }
}
boot();

let pendingResult = null;

self.onmessage = async (e) => {
  const d = e.data;
  try {
    if (d.t === "load") {
      if (!py) throw new Error("движок ещё не готов");
      py.globals.set("me_in", new Uint8Array(d.buf));
      await py.runPythonAsync(`set_audio(me_in.to_bytes(), ${d.n}, ${d.sr})`);
      if (!py.globals.has("_X")) throw new Error("движок не принял аудио");
      py.globals.set("me_cfg", JSON.stringify({ stems: !!d.stems, iters: d.iters || 6 }));
      pendingResult = null;
      const out = await py.runPythonAsync("await run_master_async()");
      let json = pendingResult || out;
      if (json && json.startsWith("[RESULT]")) json = json.slice(8);
      const res = JSON.parse(json);
      const files = {};
      for (const p of res.frames) {
        files[p] = py.FS.readFile(p).slice().buffer;
      }
      post({ t: "result", res, files }, Object.values(files));
    }
  } catch (err) {
    post({ t: "error", msg: String(err.message || err).split("\n").slice(-4).join(" | ") });
  }
};
