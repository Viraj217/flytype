"""Validate the production predictor on saved fresh crops or real visual typing."""
import argparse
import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from flytype import FlyType, MODEL_CROP_HEIGHT, MODEL_PATH, FEATURES_PATH
from flytype_benchmark import metrics, read_samples
from flytype_vision import image_to_array, pad_to_training_height, prepare_monkeytype, focus_words
from flytype_collect_validation import GEOMETRY
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent


def evaluate(fly, folder, errors_dir=None):
    samples = read_samples(folder)
    unique = {s["digest"]: s for s in samples}
    predictions = {}
    for i, s in enumerate(unique.values(), 1):
        with Image.open(s["path"]) as image:
            arr = image_to_array(image.convert("L"))
        original = str(fly.model.predict(fly.run_brain(arr)[fly.top_idx].reshape(1, -1))[0])
        started = time.perf_counter()
        corrected = fly.predict(arr)
        predictions[s["digest"]] = (original, corrected, (time.perf_counter() - started) * 1000)
        if i % 25 == 0:
            print(f"{folder.name}: {i}/{len(unique)} distinct crops", flush=True)
    y = np.array([s["label"] for s in samples])
    labels = list("abcdefghijklmnopqrstuvwxyz")
    before = np.array([predictions[s["digest"]][0] for s in samples])
    after = np.array([predictions[s["digest"]][1] for s in samples])
    # Save failure crops for visual inspection.
    failures = []
    for s, pred_before, pred_after in zip(samples, before, after):
        if pred_after != s["label"]:
            failures.append(dict(path=s["path"], actual=s["label"],
                                 before=str(pred_before), corrected=str(pred_after),
                                 crop_size=f"{s['size'][0]}x{s['size'][1]}"))
    if failures and errors_dir:
        errors_dir.mkdir(parents=True, exist_ok=True)
        for j, f in enumerate(failures):
            with Image.open(f["path"]) as src:
                raw = src.convert("L")
                raw.save(errors_dir / f"{j:03d}_{f['actual']}_as_{f['corrected']}_raw.png")
                padded_arr = pad_to_training_height(image_to_array(raw), MODEL_CROP_HEIGHT)
                padded_img = Image.fromarray((padded_arr * 255).astype(np.uint8), mode="L")
                padded_img.save(errors_dir / f"{j:03d}_{f['actual']}_as_{f['corrected']}_padded.png")
        (errors_dir / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
        print(f"  Saved {len(failures)} failure crops to {errors_dir}", flush=True)
    result = dict(folder=str(folder.resolve()), images=len(samples), unique_pixels=len(unique),
                  crop_sizes=dict(Counter(f"{s['size'][0]}x{s['size'][1]}" for s in samples)),
                  before=metrics(y, before, labels), corrected=metrics(y, after, labels),
                  missing_letters=sorted(set(labels) - set(y)),
                  failure_count=len(failures),
                  mean_prediction_ms=float(np.mean([p[2] for p in predictions.values()])),
                  predictions=[dict(path=s["path"], actual=s["label"], before=str(a), corrected=str(b))
                               for s, a, b in zip(samples, before, after)])
    print(f"{folder.name}: {result['before']['accuracy']:.2%} -> {result['corrected']['accuracy']:.2%}", flush=True)
    return result


async def live_check(fly, words, headed, errors_dir=None):
    """Score the same visual batches and state guards used by production."""
    from playwright.async_api import async_playwright
    from flytype_vision import test_finished
    rows, batches = [], []
    stale_count = 0
    stop_reason = "completed"
    captures_before = getattr(fly, "capture_count", 0)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=not headed)
        try:
            page = await browser.new_page(viewport={"width": 1280, "height": 720}, device_scale_factor=1)
            await page.goto("https://monkeytype.com", wait_until="domcontentloaded")
            await prepare_monkeytype(page)
            first_cache = await fly.build_stable_startup_cache(page)
            started = time.perf_counter()
            words_done = 0
            empty_attempts = 0
            while words_done < words:
                if await test_finished(page):
                    stop_reason = "end_of_test"
                    break
                if time.perf_counter() - started > 90:
                    stop_reason = "verification_time_limit"
                    break
                await focus_words(page)
                cache_start = time.perf_counter()
                if first_cache is not None:
                    cache, first_cache = first_cache, None
                else:
                    cache = await fly.build_visual_cache(page)
                if not cache:
                    empty_attempts += 1
                    if empty_attempts >= 20:
                        stop_reason = "no_readable_active_word"
                        break
                    await page.wait_for_timeout(100)
                    continue
                # Labels are read only by this scorer; the production cache has none.
                teacher = await fly.get_geometry(page, include_labels=True)
                if not fly.geometry_matches(fly.cached_geometry, teacher):
                    empty_attempts += 1
                    if empty_attempts >= 20:
                        stop_reason = "unstable_scoring_geometry"
                        break
                    continue
                empty_attempts = 0
                cache_ms = (time.perf_counter() - cache_start) * 1000
                word_images = fly.split_cache_words(cache)
                if len(word_images) != len(teacher):
                    raise RuntimeError("Cache word boundaries differ from scoring geometry")
                batch = dict(index=len(batches), visible_words=len(teacher),
                             visible_rows=len({round(w["row_y"], 1) for w in teacher}),
                             words_typed=0, cache_ms=round(cache_ms, 2))
                batches.append(batch)
                for word, inputs in zip(teacher, word_images):
                    if words_done >= words:
                        break
                    if len(inputs) != len(word["boxes"]):
                        raise RuntimeError("Cache letters differ from scoring geometry")
                    if not await fly.cached_word_is_current(page, word):
                        stale_count += 1
                        break
                    interrupted = False
                    word_started = time.perf_counter()
                    for li, (box, arr) in enumerate(zip(word["boxes"], inputs)):
                        if not await fly.cached_word_is_current(page, word):
                            interrupted = True
                            break
                        brain_start = time.perf_counter()
                        predicted = fly.predict(arr)
                        brain_ms = (time.perf_counter() - brain_start) * 1000
                        if not await fly.cached_word_is_current(page, word):
                            interrupted = True
                            break
                        key_start = time.perf_counter()
                        await page.keyboard.press(predicted)
                        key_ms = (time.perf_counter() - key_start) * 1000
                        actual = box["char"]
                        rows.append(dict(actual=actual, predicted=predicted, word=words_done,
                                         source_word_id=word["word_id"], batch=batch["index"],
                                         letter_index=li, crop_h=arr.shape[0], crop_w=arr.shape[1],
                                         brain_ms=round(brain_ms, 2), key_ms=round(key_ms, 2)))
                        if errors_dir and (predicted != actual or words_done == 0):
                            errors_dir.mkdir(parents=True, exist_ok=True)
                            raw = Image.fromarray((arr * 255).astype(np.uint8))
                            raw.save(errors_dir / f"word{words_done:03d}_letter{li:02d}_{actual}_as_{predicted}.png")
                    if interrupted or not await fly.cached_word_is_current(page, word):
                        break
                    await page.keyboard.press("Space")
                    batch["words_typed"] += 1
                    batch.setdefault("word_ms", []).append((time.perf_counter() - word_started) * 1000)
                    words_done += 1
                print(f"Visual typing: {words_done}/{words} words; batch {batch['index'] + 1}: "
                      f"{batch['visible_words']} visible words across {batch['visible_rows']} rows", flush=True)
            elapsed = time.perf_counter() - started
            if not rows:
                raise RuntimeError(f"No predictions were made: {stop_reason}")
            displayed = None
            if await test_finished(page):
                if stop_reason != "completed":
                    stop_reason = "end_of_test"
                await page.wait_for_timeout(300)
                displayed = await page.evaluate("""() => {
                    const r = document.querySelector('#result');
                    const value = name => r?.querySelector('.group.' + name + ' .bottom')?.textContent.trim() ?? null;
                    return {wpm:value('wpm'), accuracy:value('acc'), raw_wpm:value('raw'), chars:value('chars')};
                }""")
            y = np.array([r["actual"] for r in rows])
            pred = np.array([r["predicted"] for r in rows])
            brain = [r["brain_ms"] for r in rows]
            first = [r for r in rows if r["word"] == 0]
            return dict(mode="actual visual predictions typed, teacher scoring only",
                        images=len(rows), words_typed=words_done,
                        metrics=metrics(y, pred, list("abcdefghijklmnopqrstuvwxyz")),
                        elapsed_seconds=elapsed, stop_reason=stop_reason,
                        stale_geometry_warnings=stale_count, monkeytype_displayed=displayed,
                        screenshot_count=getattr(fly, "capture_count", 0) - captures_before,
                        batches=batches,
                        first_word=dict(actual="".join(r["actual"] for r in first),
                                        predicted="".join(r["predicted"] for r in first),
                                        correct=all(r["actual"] == r["predicted"] for r in first)),
                        crop_sizes=dict(Counter(f"{r['crop_w']}x{r['crop_h']}" for r in rows)),
                        timing=dict(mean_brain_ms=round(float(np.mean(brain)), 2),
                                    median_brain_ms=round(float(np.median(brain)), 2),
                                    p95_brain_ms=round(float(np.percentile(brain, 95)), 2),
                                    mean_key_ms=round(float(np.mean([r["key_ms"] for r in rows])), 2)),
                        predictions=rows)
        finally:
            await browser.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, nargs="*", default=[])
    parser.add_argument("--live-words", type=int, default=0)
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.validation and args.live_words <= 0:
        parser.error("Provide validation folders or --live-words")
    fly = FlyType()
    report = dict(crop_height=MODEL_CROP_HEIGHT,
                  model_sha256=hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
                  feature_indices_sha256=hashlib.sha256(FEATURES_PATH.read_bytes()).hexdigest(),
                  sessions=[])
    errors_base = args.output.parent / (args.output.stem + "_errors")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for folder in args.validation:
        errors_dir = errors_base / folder.name
        report["sessions"].append(evaluate(fly, folder, errors_dir=errors_dir))
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.live_words:
        live_errors = errors_base / "live"
        report["live"] = asyncio.run(live_check(fly, args.live_words, args.headed, errors_dir=live_errors))
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        live_result = report['live']
        print(f"Live visual typing accuracy: {live_result['metrics']['accuracy']:.2%}")
        print(f"Stop reason: {live_result['stop_reason']}")
        if live_result['stale_geometry_warnings']:
            print(f"WARNING: {live_result['stale_geometry_warnings']} stale geometry events")


if __name__ == "__main__":
    main()
