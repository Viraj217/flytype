"""Run full timed Monkeytype sessions with teacher-scored visual predictions.

Predictions come from the production FlyType pipeline (screenshot → crop →
pad → brain → classify → keyboard press).  Teacher labels from the DOM score
the run but never correct the typed keys.

Usage:
    python flytype_scored_run.py --sessions 3 --headed --output scratch/flytype_scored_runs
"""
import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid

import numpy as np
from playwright.async_api import async_playwright

from flytype import FlyType, MODEL_CROP_HEIGHT, STEPS
from flytype_benchmark import metrics
from flytype_collect_validation import GEOMETRY
from flytype_vision import prepare_monkeytype, focus_words, hide_caret
from PIL import Image

ROOT = Path(__file__).resolve().parent


async def check_test_over(page):
    """Return True when the Monkeytype result screen is visible."""
    return await page.evaluate(
        """() => {
            const r = document.querySelector('#result');
            return r && r.style.display !== 'none' && r.offsetParent !== null;
        }"""
    )


async def get_monkeytype_result(page):
    """Try to scrape Monkeytype's displayed stats from the result screen."""
    try:
        return await page.evaluate(
            """() => {
                const r = document.querySelector('#result');
                if (!r || r.style.display === 'none') return null;
                const wpm = r.querySelector('.group.wpm .bottom');
                const acc = r.querySelector('.group.acc .bottom');
                const raw = r.querySelector('.group.raw .bottom');
                const chars = r.querySelector('.group.chars .bottom');
                return {
                    wpm: wpm ? wpm.textContent.trim() : null,
                    accuracy: acc ? acc.textContent.trim() : null,
                    raw_wpm: raw ? raw.textContent.trim() : null,
                    chars: chars ? chars.textContent.trim() : null,
                };
            }"""
        )
    except Exception:
        return None


async def run_session(fly, session_id, output_dir, headed):
    """Use the batch-aware verifier so scoring follows production word boundaries."""
    from flytype_verify_runtime import live_check
    result = await live_check(fly, words=10000, headed=headed,
                              errors_dir=output_dir / f"errors_{session_id}")
    rows = result["predictions"]
    elapsed = result["elapsed_seconds"]
    word_times = [ms for batch in result["batches"] for ms in batch.get("word_ms", [])]
    timing = dict(result["timing"])
    timing["mean_word_ms"] = round(float(np.mean(word_times)), 2) if word_times else None
    return dict(
        session_id=session_id,
        mode="full timed session, visual predictions typed, teacher scoring only",
        words_typed=result["words_typed"],
        characters_typed=len(rows),
        elapsed_seconds=round(elapsed, 2),
        stop_reason=result["stop_reason"],
        stale_geometry_events=result["stale_geometry_warnings"],
        monkeytype_displayed=result["monkeytype_displayed"],
        crop_sizes=result["crop_sizes"],
        metrics=result["metrics"],
        timing=timing,
        screenshot_count=result["screenshot_count"],
        batches=result["batches"],
        first_word=result["first_word"],
        our_wpm=round((len(rows) / elapsed) * 60 / 5, 1) if elapsed else None,
        failure_count=sum(r["actual"] != r["predicted"] for r in rows),
        predictions=rows,
    )


async def main_async(args):
    fly = FlyType()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    all_reports = []
    for si in range(args.sessions):
        session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        print(f"\n=== Session {si + 1}/{args.sessions}: {session_id} ===", flush=True)

        report = await run_session(fly, session_id, output, args.headed)
        all_reports.append(report)

        session_file = output / f"{session_id}.json"
        session_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(
            f"  Result: {report['metrics']['accuracy']:.2%} accuracy, "
            f"{report['words_typed']} words, {report['characters_typed']} chars, "
            f"~{report['our_wpm']} WPM, stop={report['stop_reason']}",
            flush=True,
        )
        if report["monkeytype_displayed"]:
            mt = report["monkeytype_displayed"]
            print(f"  Monkeytype says: WPM={mt.get('wpm')}, acc={mt.get('accuracy')}", flush=True)

        if si < args.sessions - 1:
            print("  Waiting 3s before next session...", flush=True)
            await asyncio.sleep(3)

    # Summary across sessions.
    summary_lines = ["# FlyType Scored Run Summary", ""]
    summary_lines.append("| Session | Words | Chars | Accuracy | Our WPM | MT WPM | MT Acc | Stop | Failures |")
    summary_lines.append("|---|---:|---:|---:|---:|---|---|---|---:|")
    for r in all_reports:
        mt = r.get("monkeytype_displayed") or {}
        summary_lines.append(
            f"| {r['session_id'][:15]}… "
            f"| {r['words_typed']} | {r['characters_typed']} "
            f"| {r['metrics']['accuracy']:.2%} | {r['our_wpm']} "
            f"| {mt.get('wpm', 'N/A')} | {mt.get('accuracy', 'N/A')} "
            f"| {r['stop_reason']} | {r['failure_count']} |"
        )
    total_chars = sum(r["characters_typed"] for r in all_reports)
    total_correct = sum(
        sum(1 for p in r["predictions"] if p["actual"] == p["predicted"])
        for r in all_reports
    )
    total_failures = sum(r["failure_count"] for r in all_reports)
    summary_lines.extend([
        "",
        f"**Aggregate:** {total_correct}/{total_chars} characters correct "
        f"({total_correct / total_chars:.2%}), {total_failures} failures.",
    ])
    summary = "\n".join(summary_lines) + "\n"
    (output / "summary.md").write_text(summary, encoding="utf-8")
    print(f"\n{summary}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "scratch/flytype_scored_runs")
    args = parser.parse_args()
    if args.sessions < 1:
        parser.error("--sessions must be at least 1")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
