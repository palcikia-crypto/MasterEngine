"""MasterEngine browser glue (runs inside a Pyodide web worker)."""
import asyncio
import json
import os
import sys
import wave

sys.path.append("/engine")
os.makedirs("/session", exist_ok=True)
import numpy as np

_X = None
_SR = 44100.0
_STEMS = None


def set_audio(buf, n, sr_):
    global _X, _SR, _STEMS
    _SR = float(sr_)
    _STEMS = None
    _X = np.frombuffer(buf, dtype=np.float32).astype(np.float64).reshape(2, int(n)).T.copy()
    print(f"[ME] Трек принят: {int(n)} сэмплов, {sr_:.0f} Hz", flush=True)


def _wav(path, y, sr_):
    y = np.clip(y, -1.0, 1.0)
    w = wave.open(path, "wb")
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(int(sr_))
    w.writeframes((y * 32767).astype("<i2").tobytes())
    w.close()


def _pub(f):
    return {k: v for k, v in f.items() if k != "peak_lin"}


async def _separate_stems():
    from masterengine import stems
    global _STEMS
    if _STEMS is None:
        raw = stems.separate(_X, _SR, progress_cb=lambda i, n: print(
            f"[ME] Разделение на дорожки: блок {i}/{n}", flush=True))
        _STEMS = {k: v.astype(np.float32) for k, v in raw.items()}
    return _STEMS


async def run_master_async():
    from masterengine import analysis, master
    cfg = json.loads(globals()["me_cfg"])
    x, sr_ = _X, _SR
    out = {"frames": []}

    print("[ME] Анализ трека…", flush=True)
    await asyncio.sleep(0)
    params = cfg.get("params")
    iters = int(cfg.get("iters", 6))
    res, y = master.process(x, sr_, params=params, refine_iters=iters, save_experience=True)
    _wav("/session/mastered.wav", y, sr_)
    out["before"] = res["before"]
    out["after"] = res["after"]
    out["params"] = res["params"]
    out["loss"] = res["loss"]
    out["pre_gain_db"] = res["pre_gain_db"]
    out["frames"].append("/session/mastered.wav")

    if cfg.get("stems"):
        print("[ME] Разделение на дорожки…", flush=True)
        await asyncio.sleep(0)
        cache = await _separate_stems()
        sm = {}
        mix = np.zeros_like(x)
        idx = 0
        for name, s in cache.items():
            idx += 1
            print(f"[ME] Мастеринг дорожки {idx}/4: {name}…", flush=True)
            await asyncio.sleep(0)
            sf64 = s.astype(np.float64)
            f_in = analysis.analyze(sf64, sr_)
            p = stems.build_stem_params(name, f_in)
            ys = master.apply_chain(sf64, sr_, p)
            ft = analysis.analyze(ys, sr_)
            for _ in range(2):
                p = master.MODEL.refine_step(p, ft)
                ys = master.safe_render(sf64, sr_, p)
                ft = analysis.analyze(ys, sr_)
            _wav(f"/session/stem_{name}.wav", sf64, sr_)
            _wav(f"/session/stemm_{name}.wav", ys, sr_)
            sm[name] = {"before": _pub(f_in), "after": _pub(ft), "params": p}
            out["frames"].append(f"/session/stemm_{name}.wav")
            mix += ys
            await asyncio.sleep(0)
        _wav("/session/stem_mix.wav", mix, sr_)
        out["frames"].append("/session/stem_mix.wav")
        out["stems"] = sm

    _wav("/session/original.wav", x, sr_)
    out["frames"].append("/session/original.wav")
    print("[ME] Готово.", flush=True)
    print("[RESULT]" + json.dumps(out), flush=True)
    return json.dumps(out)
