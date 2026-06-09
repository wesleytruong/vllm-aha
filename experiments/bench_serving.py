#!/usr/bin/env python3
"""Online serving benchmark: Poisson arrivals against a LIVE vLLM server.

This is the realistic-serving counterpart to the Amdahl/ITL microbenchmarks.
Those measure a single static synchronous batch; this launches an actual
`vllm serve` (continuous/in-flight batching, chunked prefill, real HTTP +
streaming) and fires requests at a Poisson request rate, measuring the metrics
that production serving cares about:

  - TTFT  (time to first token, prefill-bound)
  - TPOT  (time per output token = inter-token latency, decode-bound -- where
           AHA's gate helps), reported p50/p99
  - output throughput (tok/s) and request throughput (req/s)

It compares routing modes by RELAUNCHING the server per mode with the cache-safe
gate override (VLLM_AHA_GATE_OVERRIDE): `real-gate` (default, no override) vs
`all-global` (forced full attention) and optionally `all-local`.

Prompts are real PG-19 prose (the gate is content-dependent; random tokens route
unrepresentatively). Each request gets a distinct shifted slice so the prefix
cache doesn't collapse them.

RUNTIME: bounded by --num-prompts (truncates the load) and --max-seconds-per-run
(hard wall-clock cap; pending requests are cancelled and only completed ones are
scored). Defaults target a single Poisson rate x 2 modes in ~10-15 min on a 5090.
Add rates with --request-rates "2,4,8,inf" for a fuller TPOT-vs-QPS curve.

Usage (from repo root, with the venv active):
    python experiments/bench_serving.py
    python experiments/bench_serving.py --input-len 8192 --request-rates "4,inf" \
        --num-prompts 100 --modes all-global real-gate
"""
import argparse
import asyncio
import json
import os
import random
import signal
import statistics
import subprocess
import sys
import time
import urllib.request

import aiohttp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pg19_loader import load_pg19_prompt  # noqa: E402

# Routing mode -> value for VLLM_AHA_GATE_OVERRIDE (None = real learned gate).
MODE_ENV = {"real-gate": None, "all-global": "global", "all-local": "local"}


def wait_for_server(port, proc, timeout=420):
    """Poll /health until the server is ready (or it dies / times out)."""
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False  # server process exited during startup
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def launch_server(args, mode, log_path):
    """Start `vllm serve` for one routing mode; return the Popen (own pgroup)."""
    env = os.environ.copy()
    env.pop("VLLM_AHA_GATE_OVERRIDE", None)
    if MODE_ENV[mode] is not None:
        env["VLLM_AHA_GATE_OVERRIDE"] = MODE_ENV[mode]
    hf_overrides = json.dumps({
        "attention_implementation": "flashinfer",
        "max_position_embeddings": args.model_max,
    })
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--port", str(args.port),
        "--trust-remote-code",
        "--dtype", "half",
        "--hf-overrides", hf_overrides,
        "--max-model-len", str(args.model_max),
        "--gpu-memory-utilization", str(args.gpu_mem),
        "--max-num-seqs", str(args.max_num_seqs),
        "--disable-log-stats",
    ]
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)  # own process group for clean kill
    return proc


def stop_server(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


async def one_request(session, url, model, prompt_ids, output_len):
    """Fire one streaming completion; return per-request latency metrics."""
    payload = {
        "model": model,
        "prompt": prompt_ids,           # token-id prompt -> exact context length
        "max_tokens": output_len,
        "temperature": 0.0,
        "ignore_eos": True,             # force exactly output_len decode steps
        "stream": True,
    }
    t0 = time.perf_counter()
    ttft = None
    prev = t0
    itls = []  # inter-token latencies after the first token
    ntok = 0
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                return {"ok": False, "err": f"http {resp.status}"}
            async for raw in resp.content:
                if not raw.startswith(b"data:"):
                    continue
                data = raw[len(b"data:"):].strip()
                if data == b"[DONE]":
                    break
                now = time.perf_counter()
                if ttft is None:
                    ttft = now - t0
                else:
                    itls.append(now - prev)
                prev = now
                ntok += 1
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "err": repr(e)}
    if ttft is None or ntok == 0:
        return {"ok": False, "err": "no tokens"}
    tpot = statistics.mean(itls) if itls else 0.0  # mean inter-token latency
    return {"ok": True, "ttft": ttft, "tpot": tpot, "ntok": ntok,
            "latency": prev - t0}


async def run_load(args, prompts, rate):
    """Poisson(rate) arrivals (rate='inf' => all at once). Hard wall-clock cap."""
    url = f"http://127.0.0.1:{args.port}/v1/completions"
    is_inf = rate == float("inf")
    rng = random.Random(1234)  # reproducible arrival process
    conn = aiohttp.TCPConnector(limit=0)  # no client-side cap; server batches
    timeout = aiohttp.ClientTimeout(total=None, sock_read=None)
    results = []
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        wall0 = time.perf_counter()
        tasks = []
        for p in prompts:
            tasks.append(asyncio.create_task(
                one_request(session, url, args.model, p, args.output_len)))
            if not is_inf:
                await asyncio.sleep(rng.expovariate(rate))
        # Drain with a hard cap; cancel stragglers so one run can't overrun.
        done, pending = await asyncio.wait(
            tasks, timeout=args.max_seconds_per_run)
        for t in pending:
            t.cancel()
        wall = time.perf_counter() - wall0
        for t in done:
            try:
                results.append(t.result())
            except Exception:  # noqa: BLE001
                pass
        n_cancelled = len(pending)
    return results, wall, n_cancelled


def summarize(results, wall, n_cancelled):
    ok = [r for r in results if r.get("ok")]
    if not ok:
        return None
    ttft = sorted(r["ttft"] for r in ok)
    tpot = sorted(r["tpot"] for r in ok)
    out_tok = sum(r["ntok"] for r in ok)

    def pct(xs, q):
        if not xs:
            return 0.0
        i = min(len(xs) - 1, int(q * len(xs)))
        return xs[i]

    return {
        "completed": len(ok),
        "failed": len(results) - len(ok),
        "cancelled": n_cancelled,
        "wall_s": round(wall, 1),
        "ttft_p50_ms": round(pct(ttft, 0.50) * 1e3, 1),
        "ttft_p99_ms": round(pct(ttft, 0.99) * 1e3, 1),
        "tpot_p50_ms": round(pct(tpot, 0.50) * 1e3, 2),
        "tpot_p99_ms": round(pct(tpot, 0.99) * 1e3, 2),
        "out_tok_s": round(out_tok / wall, 1) if wall else 0.0,
        "req_s": round(len(ok) / wall, 2) if wall else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xuan-luo/AHA-OLMO2")
    ap.add_argument("--input-len", type=int, default=8192,
                    help="prompt/context length in tokens")
    ap.add_argument("--output-len", type=int, default=128,
                    help="decode tokens per request (TPOT is measured over these)")
    ap.add_argument("--num-prompts", type=int, default=100,
                    help="TRUNCATES the load; lower = faster run")
    ap.add_argument("--request-rates", default="4",
                    help="comma list of req/s; 'inf' = fire all at once (saturation)")
    ap.add_argument("--modes", nargs="+", default=["all-global", "real-gate"],
                    choices=list(MODE_ENV))
    ap.add_argument("--port", type=int, default=8147)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--model-max", type=int, default=0,
                    help="max_model_len; default = input_len + output_len + 256")
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--max-seconds-per-run", type=float, default=300.0,
                    help="hard wall-clock cap per (mode,rate); stragglers cancelled")
    ap.add_argument("--out-dir", default="results/serving")
    args = ap.parse_args()
    if args.model_max == 0:
        args.model_max = args.input_len + args.output_len + 256
    rates = [float("inf") if x.strip() in ("inf", "0") else float(x)
             for x in args.request_rates.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)

    # Distinct shifted PG-19 slices, one per request (exact length = input_len).
    base = load_pg19_prompt(args.input_len + args.num_prompts, args.model)
    prompts = [base[i:i + args.input_len] for i in range(args.num_prompts)]

    print(f"# serving bench: model={args.model} ctx={args.input_len} "
          f"out={args.output_len} N={args.num_prompts} rates={rates} "
          f"modes={args.modes} max_num_seqs={args.max_num_seqs}", flush=True)

    all_rows = []
    for mode in args.modes:
        log_path = os.path.join(args.out_dir, f"server_{mode}.log")
        print(f"\n=== launching server [{mode}] (log: {log_path}) ===", flush=True)
        proc = launch_server(args, mode, log_path)
        try:
            if not wait_for_server(args.port, proc):
                print(f"  SERVER FAILED to start for {mode}; see {log_path}",
                      flush=True)
                continue
            print("  server ready.", flush=True)
            for rate in rates:
                tag = "inf" if rate == float("inf") else f"{rate:g}"
                print(f"  -> load: rate={tag} req/s ...", flush=True)
                results, wall, n_cancel = asyncio.run(run_load(args, prompts, rate))
                s = summarize(results, wall, n_cancel)
                if s is None:
                    print(f"     (no completed requests at rate={tag})", flush=True)
                    continue
                s.update(mode=mode, rate=tag, ctx=args.input_len,
                         output_len=args.output_len)
                all_rows.append(s)
                print(f"     done={s['completed']} cancelled={s['cancelled']} "
                      f"TTFT p50/p99={s['ttft_p50_ms']}/{s['ttft_p99_ms']}ms "
                      f"TPOT p50/p99={s['tpot_p50_ms']}/{s['tpot_p99_ms']}ms "
                      f"out={s['out_tok_s']}tok/s", flush=True)
        finally:
            stop_server(proc)
            time.sleep(3)  # let the port free before relaunch

    out_json = os.path.join(args.out_dir,
                            f"serving_ctx{args.input_len}.json")
    with open(out_json, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nwrote {out_json} ({len(all_rows)} rows)", flush=True)

    # Comparison table + real-vs-global TPOT speedup per rate.
    print(f"\n{'mode':>11} {'rate':>5} {'done':>5} {'TTFT p50':>9} "
          f"{'TTFT p99':>9} {'TPOT p50':>9} {'TPOT p99':>9} {'out tok/s':>10}")
    for r in all_rows:
        print(f"{r['mode']:>11} {r['rate']:>5} {r['completed']:>5} "
              f"{r['ttft_p50_ms']:>8.0f}m {r['ttft_p99_ms']:>8.0f}m "
              f"{r['tpot_p50_ms']:>8.2f}m {r['tpot_p99_ms']:>8.2f}m "
              f"{r['out_tok_s']:>10.1f}")
    by_rate = {}
    for r in all_rows:
        by_rate.setdefault(r["rate"], {})[r["mode"]] = r
    print("\nreal-gate vs all-global (lower TPOT = better):")
    for rate, m in by_rate.items():
        if "real-gate" in m and "all-global" in m:
            g, rl = m["all-global"]["tpot_p50_ms"], m["real-gate"]["tpot_p50_ms"]
            thr = m["real-gate"]["out_tok_s"] / max(m["all-global"]["out_tok_s"], 1e-9)
            print(f"  rate={rate}: TPOT p50 {g:.2f} -> {rl:.2f} ms "
                  f"({g / max(rl, 1e-9):.2f}x faster), "
                  f"throughput {thr:.2f}x")


if __name__ == "__main__":
    main()
