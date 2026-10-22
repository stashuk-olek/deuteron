#!/usr/bin/env python3
"""Compile and run the open problems; save PTX, LLVM IR and actual GPU outputs."""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent
SERVER = "http://runner.riftstack.ai:48701"


def request(server, endpoint, body, http_client):
    if http_client == "curl":
        command = [
            "curl",
            "--fail-with-body",
            "--silent",
            "--show-error",
            "--connect-timeout",
            "10",
            "--max-time",
            "300",
            "-H",
            "Content-Type: application/json",
            "--data-binary",
            "@-",
            server.rstrip("/") + endpoint,
        ]
        response = subprocess.run(command, input=json.dumps(body), text=True, capture_output=True, timeout=310)
        if response.returncode:
            raise RuntimeError(response.stderr or response.stdout)
        return json.loads(response.stdout)
    response = requests.post(server.rstrip("/") + endpoint, json=body, timeout=(10, 300))
    response.raise_for_status()
    return response.json()


def check_problem(problem, args):
    metadata = json.loads((problem / "meta.json").read_text())
    destination = args.artifacts / problem.name
    destination.mkdir(parents=True, exist_ok=True)
    produced = {}
    records = []
    for launch in metadata["kernels"]:
        entry = launch["entry"]
        command = [
            sys.executable,
            str(ROOT / "compile.py"),
            str(problem / (entry + ".ir.json")),
            "--opt-level",
            str(args.opt_level),
            "--dump-llvm",
            str(destination / (entry + ".ll")),
            "--dump-unoptimized",
            str(destination / (entry + ".unoptimized.ll")),
        ]
        start = time.perf_counter()
        compiled = subprocess.run(command, capture_output=True, text=True, timeout=60)
        compile_seconds = time.perf_counter() - start
        if compiled.returncode:
            raise RuntimeError(compiled.stderr)
        (destination / (entry + ".ptx")).write_text(compiled.stdout)
        assembled = request(args.server, "/compile", {"ptx": compiled.stdout}, args.http_client)
        if not assembled.get("ok") or entry not in assembled.get("entry_points", []):
            raise RuntimeError(f"assembly failed: {assembled}")
        by_name = {arg["name"]: arg for arg in launch["args"]}
        run_args = []
        for name in launch["arg_order"]:
            arg = dict(by_name[name])
            if arg.pop("intermediate", False):
                arg["data"] = produced[name].reshape(-1).tolist()
            run_args.append(arg)
        body = {
            "cubin_id": assembled["cubin_id"],
            "entry": entry,
            "grid": launch["grid"],
            "block": launch["block"],
            "smem_bytes": launch["smem_bytes"],
            "zero_outputs": launch["zero_outputs"],
            "args": run_args,
            "warmup": args.warmup,
            "iters": args.iters,
            "max_return": max(4096, sum(math.prod(a["shape"]) for a in run_args if a.get("output"))),
        }
        for _ in range(60):
            result = request(args.server, "/run", body, args.http_client)
            if result.get("stage") != "busy":
                break
            time.sleep(2)
        if not result.get("ok"):
            raise RuntimeError(f"execution failed: {result}")
        for name, output in result["outputs"].items():
            if output.get("truncated"):
                raise RuntimeError(f"runner truncated {name}")
            produced[name] = np.array(output["values"], dtype=np.float32).reshape(output["shape"])
        records.append(
            {
                "entry": entry,
                "compile_seconds": compile_seconds,
                "ptx_bytes": len(compiled.stdout),
                "registers": assembled.get("registers", {}).get(entry),
                "median_us": result.get("median_us"),
                "cubin_id": assembled["cubin_id"],
                "arch": assembled.get("arch"),
            }
        )
    # These are actual runner results, never the expected arrays or CPU substitutes.
    np.savez(destination / "actual.npz", **produced)
    errors = {}
    with np.load(problem / "expected.npz") as expected:
        for name in expected.files:
            if name not in produced:
                raise RuntimeError(f"runner did not return required output {name}")
            actual, reference = produced[name].reshape(expected[name].shape), expected[name]
            errors[name] = float(np.max(np.abs(actual - reference))) if actual.size else 0.0
            np.testing.assert_allclose(actual, reference, rtol=metadata["rtol"], atol=metadata["atol"], equal_nan=False)
    report = {
        "problem": problem.name,
        "passed": True,
        "rtol": metadata["rtol"],
        "atol": metadata["atol"],
        "max_abs_error": errors,
        "kernels": records,
        "iters": args.iters,
    }
    (destination / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problems", nargs="*", help="problem names; defaults to every open problem")
    parser.add_argument("--server", default=SERVER)
    parser.add_argument("--http-client", choices=("requests", "curl"), default="requests")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--opt-level", type=int, choices=(0, 1, 2, 3), default=2)
    parser.add_argument("--iters", type=int, default=1, choices=range(1, 201), metavar="1..200")
    parser.add_argument("--warmup", type=int, default=0, choices=range(21), metavar="0..20")
    args = parser.parse_args()
    problems = (
        [ROOT / "problems" / p for p in args.problems]
        if args.problems
        else sorted(p for p in (ROOT / "problems").iterdir() if (p / "meta.json").exists())
    )
    reports = []
    for problem in problems:
        try:
            report = check_problem(problem, args)
            print(f"PASS {problem.name}: max abs error {max(report['max_abs_error'].values()):.3e}", flush=True)
        except (
            RuntimeError,
            OSError,
            ValueError,
            AssertionError,
            subprocess.TimeoutExpired,
            requests.RequestException,
        ) as exc:
            report = {"problem": problem.name, "passed": False, "error": str(exc)}
            print(f"FAIL {problem.name}: {exc}", file=sys.stderr, flush=True)
        reports.append(report)
    args.artifacts.mkdir(parents=True, exist_ok=True)
    (args.artifacts / "results.json").write_text(json.dumps(reports, indent=2) + "\n")
    passed = sum(r["passed"] for r in reports)
    print(f"{passed}/{len(reports)} passed")
    return int(passed != len(reports))


if __name__ == "__main__":
    sys.exit(main())
