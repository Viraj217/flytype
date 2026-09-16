import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import asyncio
import contextlib
import hashlib
import inspect
import json
import time
from pathlib import Path

import joblib
import numpy as np
from playwright.async_api import async_playwright

from flysim import FlyBrain
from flyeye import FlyEye

from flytype_vision import (
    hide_caret,
    capture_words,
    crop_letter,
    image_to_array,
    pad_to_training_height,
    prepare_monkeytype,
    focus_words,
    test_finished,
)


ROOT = Path(__file__).parent

MODEL_PATH = (
    ROOT
    / "models"
    / "flytype_model.joblib"
)

FEATURES_PATH = (
    ROOT
    / "models"
    / "top_idx.npy"
)

MANIFEST_PATH = (
    ROOT
    / "models"
    / "model_manifest.json"
)

STEPS = 50

# Canvas height of the PNGs used to fit models/flytype_model.joblib.
# Live 59px crops require 5px of background above and below the glyph.
MODEL_CROP_HEIGHT = 69

MAX_HZ = 180.0

# Zero means every complete word that is actually visible in the viewport.
# Hidden DOM words are rejected with hit-testing before they enter the cache.
MAX_VISIBLE_WORDS = 0

STARTUP_STABILITY_ATTEMPTS = 8
STARTUP_STABILITY_DELAY_MS = 120

DEBUG_TIMING = True


class FlyType:
    def __init__(self, telemetry=None):
        self.telemetry = telemetry
        self.capture_count = 0
        self.cached_geometry = None
        self.cached_frame = None
        self._last_retina_rates = np.array([], dtype=np.float32)
        print("Loading fly brain...")

        self.fb = FlyBrain()
        self.eye = FlyEye(
            self.fb
        )

        print(
            "Loading classifier..."
        )

        self.model = joblib.load(
            MODEL_PATH
        )

        self.top_idx = np.load(
            FEATURES_PATH
        )

        self._validate_manifest()

        print()
        print("Fly ready.")
        print(
            "Selected neurons:",
            len(self.top_idx)
        )
        print(
            "Brain steps:",
            STEPS
        )
        print(
            "Screenshot batch:",
            "all visible words"
            if MAX_VISIBLE_WORDS == 0
            else f"up to {MAX_VISIBLE_WORDS} visible words"
        )
        print()

    async def emit(self, event_type, **payload):
        """Publish optional runtime telemetry without coupling inference to a UI."""
        if self.telemetry is None:
            return
        try:
            result = self.telemetry({"type": event_type, **payload})
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            print(f"Telemetry disconnected: {exc}")
            self.telemetry = None

    def _validate_manifest(self):
        """Check loaded model against manifest metadata, if present."""
        if not MANIFEST_PATH.exists():
            print(
                "No model manifest found; "
                "skipping validation."
            )
            return
        manifest = json.loads(
            MANIFEST_PATH.read_text(
                encoding="utf-8"
            )
        )
        errors = []
        if manifest.get("crop_height") != MODEL_CROP_HEIGHT:
            errors.append(
                f"crop_height: manifest says {manifest.get('crop_height')}, "
                f"code says {MODEL_CROP_HEIGHT}"
            )
        if manifest.get("simulator_steps") != STEPS:
            errors.append(
                f"simulator_steps: manifest says {manifest.get('simulator_steps')}, "
                f"code says {STEPS}"
            )
        if manifest.get("n_selected_features") != len(self.top_idx):
            errors.append(
                f"n_selected_features: manifest says {manifest.get('n_selected_features')}, "
                f"loaded {len(self.top_idx)}"
            )
        model_hash = hashlib.sha256(
            MODEL_PATH.read_bytes()
        ).hexdigest()
        if manifest.get("model_sha256") and manifest["model_sha256"] != model_hash:
            errors.append(
                f"model_sha256 mismatch: manifest={manifest['model_sha256'][:16]}... "
                f"actual={model_hash[:16]}..."
            )
        feat_hash = hashlib.sha256(
            FEATURES_PATH.read_bytes()
        ).hexdigest()
        if manifest.get("feature_indices_sha256") and manifest["feature_indices_sha256"] != feat_hash:
            errors.append(
                f"feature_indices_sha256 mismatch: manifest={manifest['feature_indices_sha256'][:16]}... "
                f"actual={feat_hash[:16]}..."
            )
        if errors:
            raise RuntimeError(
                "Model manifest validation failed:\n  "
                + "\n  ".join(errors)
            )
        print(
            "Model manifest validated."
        )

    def make_l1_drive(
        self,
        arr
    ):
        h, w = arr.shape

        drive = self.eye.look(
            arr,
            cx=w // 2,
            cy=h // 2,
            fov_w=w,
            fov_h=h,
            max_hz=MAX_HZ
        )

        items = list(
            drive.items()
        )

        l1_key, l1_rates = (
            items[0]
        )

        self._last_retina_rates = np.asarray(l1_rates, dtype=np.float32).copy()
        return {l1_key: l1_rates}

    def run_brain(
        self,
        arr
    ):
        drive = (
            self.make_l1_drive(
                arr
            )
        )

        result = self.fb.run(
            drive=drive,
            steps=STEPS,
            seed=42,
            spike_log=True
        )

        spikes = np.zeros(
            self.fb.n,
            dtype=np.float32
        )

        for fired in result["_spikes"]:
            if len(fired):
                np.add.at(
                    spikes,
                    fired,
                    1
                )

        return spikes

    def infer(self, arr):
        """Return the prediction and the exact activity used to produce it."""
        normalized = pad_to_training_height(arr, MODEL_CROP_HEIGHT)
        spikes = self.run_brain(normalized)
        features = spikes[self.top_idx].reshape(1, -1)
        prediction = str(self.model.predict(features)[0])

        alternatives = []
        confidence = None
        try:
            probabilities = np.asarray(self.model.predict_proba(features)[0])
            classes = np.asarray(self.model.classes_).astype(str)
            if probabilities.ndim != 1 or classes.ndim != 1:
                raise TypeError("Classifier probabilities are not one-dimensional")
            order = np.argsort(probabilities)[::-1][:3]
            alternatives = [
                {"letter": str(classes[i]), "probability": float(probabilities[i])}
                for i in order
            ]
            match = np.flatnonzero(classes == prediction)
            if len(match):
                confidence = float(probabilities[match[0]])
        except (AttributeError, TypeError, ValueError, IndexError):
            # Some compatible classifiers expose only predict().
            pass

        return {
            "prediction": prediction,
            "confidence": confidence,
            "alternatives": alternatives,
            "spikes": spikes,
            "crop": normalized,
            "retina_rates": getattr(
                self,
                "_last_retina_rates",
                np.array([], dtype=np.float32),
            ),
        }

    def predict(self, arr):
        return self.infer(arr)["prediction"]

    async def get_geometry(
        self,
        page,
        include_labels=False,
    ):
        return await page.evaluate(
            """
            ({maxWords, includeLabels}) => {
                const container = document.querySelector('#words');
                if (!container || container.classList.contains('blurred')) return null;
                // Cache identities survive ordinary layout shifts and DOM trimming.
                const ids = window.__flytypeWordIds ||= {map: new WeakMap(), next: 1};
                const wordId = word => {
                    if (!ids.map.has(word)) ids.map.set(word, ids.next++);
                    return ids.map.get(word);
                };
                let left = 0, top = 0, right = innerWidth, bottom = innerHeight;
                for (let el = container; el; el = el.parentElement) {
                    const style = getComputedStyle(el);
                    if (style.display === 'none' || style.visibility === 'hidden' ||
                        Number(style.opacity) < 0.99 ||
                        (style.filter !== 'none' && style.filter !== 'blur(0px)')) return null;
                    const r = el.getBoundingClientRect();
                    if (['hidden', 'clip', 'auto', 'scroll'].includes(style.overflowX)) {
                        left = Math.max(left, r.left + el.clientLeft);
                        right = Math.min(right, r.left + el.clientLeft + el.clientWidth);
                    }
                    if (['hidden', 'clip', 'auto', 'scroll'].includes(style.overflowY)) {
                        top = Math.max(top, r.top + el.clientTop);
                        bottom = Math.min(bottom, r.top + el.clientTop + el.clientHeight);
                    }
                }
                const words = [
                    ...document.querySelectorAll(
                        "#words .word"
                    )
                ];

                const activeIndex =
                    words.findIndex(
                        w =>
                            w.classList.contains(
                                "active"
                            )
                    );

                if (activeIndex < 0) {
                    return null;
                }

                const result = [];

                for (
                    let wi = activeIndex;
                    wi < words.length;
                    wi++
                ) {
                    const word = words[wi];

                    const letters = [
                        ...word.querySelectorAll(
                            "letter"
                        )
                    ];

                    const boxes = [];
                    let completeAndVisible = true;

                    for (
                        let li = 0;
                        li < letters.length;
                        li++
                    ) {
                        const letter =
                            letters[li];

                        if (
                            wi === activeIndex
                            &&
                            (
                                letter.classList.contains(
                                    "correct"
                                )
                                ||
                                letter.classList.contains(
                                    "incorrect"
                                )
                            )
                        ) {
                            continue;
                        }

                        const r =
                            letter.getBoundingClientRect();

                        if (
                            r.width <= 0
                            ||
                            r.height <= 0
                            ||
                            Math.floor(r.left) < Math.floor(left)
                            ||
                            Math.floor(r.top) < Math.floor(top)
                            ||
                            Math.ceil(r.right) > Math.ceil(right)
                            ||
                            Math.ceil(r.bottom) > Math.ceil(bottom)
                        ) {
                            completeAndVisible = false;
                            break;
                        }

                        const cx = r.left + r.width / 2;
                        const cy = r.top + r.height / 2;
                        const hit = document.elementFromPoint(cx, cy);

                        if (!(hit === letter || letter.contains(hit))) {
                            completeAndVisible = false;
                            break;
                        }

                        const box = {
                            x: r.x,
                            y: r.y,
                            width: r.width,
                            height: r.height
                        };

                        if (includeLabels) {
                            box.char = (letter.textContent || "").toLowerCase();
                        }

                        boxes.push(box);
                    }

                    if (!completeAndVisible || letters.length === 0) {
                        // Never skip an unread word: later predictions would be
                        // typed into the wrong active word.
                        break;
                    }

                    result.push({
                        word_index: wi,
                        word_id: wordId(word),
                        row_y: boxes.length ? boxes[0].y : word.getBoundingClientRect().y,
                        boxes: boxes
                    });

                    if (maxWords > 0 && result.length >= maxWords) {
                        break;
                    }
                }

                return result;
            }
            """,
            {
                "maxWords": MAX_VISIBLE_WORDS,
                "includeLabels": include_labels,
            }
        )

    def geometry_matches(
        self,
        a,
        b,
        tolerance=0.25
    ):
        if (
            a is None
            or b is None
        ):
            return False

        if len(a) != len(b):
            return False

        for wa, wb in zip(
            a,
            b
        ):
            if (wa.get("word_index") != wb.get("word_index") or
                    wa.get("word_id") != wb.get("word_id")):
                return False

            ba = wa["boxes"]
            bb = wb["boxes"]

            if len(ba) != len(bb):
                return False

            for xa, xb in zip(
                ba,
                bb
            ):
                if (
                    abs(
                        xa["x"]
                        - xb["x"]
                    ) > tolerance
                    or
                    abs(
                        xa["y"]
                        - xb["y"]
                    ) > tolerance
                    or
                    abs(
                        xa["width"]
                        - xb["width"]
                    ) > tolerance
                    or
                    abs(
                        xa["height"]
                        - xb["height"]
                    ) > tolerance
                ):
                    return False

        return True

    def visual_cache_matches(self, a, b):
        if not a or not b or len(a) != len(b):
            return False
        for (kind_a, arr_a), (kind_b, arr_b) in zip(a, b):
            if kind_a != kind_b:
                return False
            if kind_a == "char" and not np.array_equal(arr_a, arr_b):
                return False
        return True

    async def build_stable_startup_cache(self, page):
        """Wait out focus/font transitions and return the first usable cache."""
        await focus_words(page)
        previous = None
        previous_geometry = None
        for _ in range(STARTUP_STABILITY_ATTEMPTS):
            current = await self.build_visual_cache(page, debug=False)
            geometry = getattr(self, "cached_geometry", None)
            if (self.visual_cache_matches(previous, current) and
                    self.geometry_matches(previous_geometry, geometry)):
                if DEBUG_TIMING:
                    print("Startup glyphs stable across consecutive captures.")
                return current
            previous = current
            previous_geometry = geometry
            await page.wait_for_timeout(STARTUP_STABILITY_DELAY_MS)
        raise RuntimeError("Monkeytype glyphs did not become visually stable at startup")

    @staticmethod
    def split_cache_words(cache):
        words, letters = [], []
        for kind, arr in cache:
            if kind == "char":
                letters.append(arr)
            elif kind == "space":
                words.append(letters)
                letters = []
        if letters:
            raise ValueError("Cache ends without a word boundary")
        return words

    async def cached_word_is_current(self, page, word):
        return await page.evaluate("""expected => {
            const result = document.querySelector('#result');
            if (result && result.getClientRects().length &&
                getComputedStyle(result).visibility !== 'hidden') return false;
            const words = document.querySelector('#words');
            const active = words?.querySelector('.word.active');
            if (!active || words.classList.contains('blurred')) return false;
            return window.__flytypeWordIds?.map.get(active) === expected;
        }""", word["word_id"])

    async def build_visual_cache(
        self,
        page,
        debug=True,
    ):
        start = time.perf_counter()
        self.cached_geometry = None

        before = (
            await self.get_geometry(
                page
            )
        )

        if not before:
            return []

        img, viewport, frame = (
            await capture_words(
                page,
                include_bytes=True,
            )
        )
        self.capture_count = getattr(self, "capture_count", 0) + 1

        if img is None:
            return []

        after = (
            await self.get_geometry(
                page
            )
        )

        if not self.geometry_matches(
            before,
            after
        ):
            print(
                "Layout changed during "
                "capture. Retrying..."
            )

            return []

        tokens = []

        for word in after:
            boxes = word["boxes"]

            for box in boxes:
                crop = crop_letter(
                    img,
                    viewport,
                    box
                )

                arr = image_to_array(
                    crop
                )

                # Stable blur is still blur: reject near-blank captures rather
                # than feeding them to a classifier as valid glyphs.
                if float(np.ptp(arr)) < 8 / 255:
                    return []

                tokens.append(
                    (
                        "char",
                        arr
                    )
                )

            tokens.append(
                (
                    "space",
                    None
                )
            )

        elapsed = (
            time.perf_counter()
            - start
        )
        self.cached_geometry = after
        self.cached_frame = frame

        if DEBUG_TIMING and debug:
            char_count = sum(
                1
                for kind, _ in tokens
                if kind == "char"
            )

            word_count = sum(
                1
                for kind, _ in tokens
                if kind == "space"
            )

            print()
            print(
                f"Cache: "
                f"{char_count} chars, "
                f"{word_count} words, "
                f"{len({round(word['row_y'], 1) for word in after})} rows, "
                f"{elapsed * 1000:.1f}ms"
            )

        return tokens

    async def run(self, headless=False, telemetry=None, max_words=0):
        """Run Flytype and optionally stream evidence from the production path."""
        if telemetry is not None:
            self.telemetry = telemetry

        captures_before = self.capture_count
        total_keys = 0
        total_letters = 0
        total_words = 0
        empty_captures = 0
        batch_index = 0
        stop_reason = "completed"
        displayed = None
        stop_requested = False
        start_time = time.perf_counter()

        await self.emit(
            "session_started",
            total_neurons=int(self.fb.n),
            selected_neurons=int(len(self.top_idx)),
            brain_steps=STEPS,
            model_crop_height=MODEL_CROP_HEIGHT,
        )

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=headless,
                    args=["--disable-dev-shm-usage", "--no-sandbox"]
                )
                screencast_task = None
                cdp = None
                try:
                    context = await browser.new_context(
                        viewport={"width": 1280, "height": 720},
                        device_scale_factor=1,
                    )
                    page = await context.new_page()

                    # Chromium's screencast is presentation-only. It pushes a
                    # frame when the page changes and never feeds the model or
                    # increments the inference screenshot counter.
                    if self.telemetry is not None:
                        latest_frame = {"data": None, "sequence": 0}
                        cdp = await context.new_cdp_session(page)
                        loop = asyncio.get_running_loop()

                        def on_screencast_frame(params):
                            latest_frame["data"] = params.get("data")
                            latest_frame["sequence"] += 1
                            asyncio.run_coroutine_threadsafe(
                                cdp.send(
                                    "Page.screencastFrameAck",
                                    {"sessionId": params["sessionId"]},
                                ),
                                loop,
                            )

                        cdp.on("Page.screencastFrame", on_screencast_frame)
                        await cdp.send(
                            "Page.startScreencast",
                            {
                                "format": "jpeg",
                                "quality": 62,
                                "maxWidth": 1280,
                                "maxHeight": 720,
                                "everyNthFrame": 1,
                            },
                        )

                        async def pump_screencast():
                            sent = -1
                            while not page.is_closed():
                                if (
                                    latest_frame["data"] is not None
                                    and latest_frame["sequence"] != sent
                                ):
                                    sent = latest_frame["sequence"]
                                    await self.emit(
                                        "browser_frame",
                                        frame=latest_frame["data"],
                                    )
                                await asyncio.sleep(0.1)

                        screencast_task = asyncio.create_task(pump_screencast())

                    print("Opening Monkeytype...")
                    await self.emit("status", message="Opening Monkeytype")
                    await page.goto(
                        "https://monkeytype.com",
                        wait_until="domcontentloaded",
                        timeout=60000,
                    )
                    await page.wait_for_timeout(1500)
                    await prepare_monkeytype(page)
                    await page.locator("#words").wait_for(state="visible")
                    await hide_caret(page)

                    print("\nStarting FlyType...\n")
                    await self.emit("status", message="Waiting for stable glyphs")
                    first_cache = await self.build_stable_startup_cache(page)
                    start_time = time.perf_counter()

                    while True:
                        if stop_requested:
                            break
                        if await test_finished(page):
                            stop_reason = "end_of_test"
                            print("Monkeytype test finished.")
                            break

                        await focus_words(page)
                        cache_start = time.perf_counter()
                        if first_cache is not None:
                            cache = first_cache
                            first_cache = None
                        else:
                            cache = await self.build_visual_cache(page)
                        cache_time = time.perf_counter() - cache_start

                        if not cache:
                            empty_captures += 1
                            if empty_captures >= 20:
                                raise RuntimeError(
                                    "No readable active word after 20 capture attempts"
                                )
                            await asyncio.sleep(0.1)
                            continue

                        empty_captures = 0
                        captured_words = self.cached_geometry
                        cached_word_index = 0
                        letter_index = 0
                        predicted_word = []
                        batch_index += 1
                        rows = len({round(word["row_y"], 1) for word in captured_words})
                        await self.emit(
                            "batch_captured",
                            batch=batch_index,
                            screenshot=self.cached_frame,
                            geometry=captured_words,
                            visible_words=len(captured_words),
                            visible_rows=rows,
                            cache_ms=round(cache_time * 1000, 2),
                            inference_captures=self.capture_count - captures_before,
                        )

                        for token_type, arr in cache:
                            if cached_word_index >= len(captured_words):
                                break
                            current_word = captured_words[cached_word_index]
                            if not await self.cached_word_is_current(page, current_word):
                                await self.emit(
                                    "cache_invalidated",
                                    reason="active word changed",
                                    batch=batch_index,
                                )
                                break

                            if token_type == "space":
                                key_start = time.perf_counter()
                                await page.keyboard.press("Space")
                                key_time = time.perf_counter() - key_start
                                total_keys += 1
                                total_words += 1
                                elapsed = time.perf_counter() - start_time
                                wpm = total_keys / max(elapsed, 0.001) * 60 / 5
                                await self.emit(
                                    "word_completed",
                                    batch=batch_index,
                                    word_index=cached_word_index,
                                    prediction="".join(predicted_word),
                                    words=total_words,
                                    characters=total_letters,
                                    wpm=round(wpm, 1),
                                    key_ms=round(key_time * 1000, 2),
                                )
                                if max_words and total_words >= max_words:
                                    stop_reason = "word_limit"
                                    stop_requested = True
                                    break
                                cached_word_index += 1
                                letter_index = 0
                                predicted_word = []
                                if DEBUG_TIMING:
                                    print(
                                        f"[space] key={key_time * 1000:5.1f}ms "
                                        f"~{wpm:6.1f} WPM"
                                    )
                                continue

                            cycle_start = time.perf_counter()
                            brain_start = time.perf_counter()
                            inference = self.infer(arr)
                            brain_time = time.perf_counter() - brain_start
                            prediction = inference["prediction"]

                            if not await self.cached_word_is_current(page, current_word):
                                await self.emit(
                                    "cache_invalidated",
                                    reason="layout changed after inference",
                                    batch=batch_index,
                                )
                                break

                            await self.emit(
                                "inference",
                                batch=batch_index,
                                word_index=cached_word_index,
                                letter_index=letter_index,
                                prediction=prediction,
                                confidence=inference["confidence"],
                                alternatives=inference["alternatives"],
                                spikes=inference["spikes"],
                                crop=inference["crop"],
                                retina_rates=inference["retina_rates"],
                                brain_ms=round(brain_time * 1000, 2),
                            )

                            key_start = time.perf_counter()
                            await page.keyboard.press(prediction)
                            key_time = time.perf_counter() - key_start
                            cycle_time = time.perf_counter() - cycle_start
                            predicted_word.append(prediction)
                            total_letters += 1
                            total_keys += 1
                            elapsed = time.perf_counter() - start_time
                            wpm = total_keys / max(elapsed, 0.001) * 60 / 5
                            await self.emit(
                                "key_sent",
                                prediction=prediction,
                                batch=batch_index,
                                word_index=cached_word_index,
                                letter_index=letter_index,
                                characters=total_letters,
                                words=total_words,
                                wpm=round(wpm, 1),
                                key_ms=round(key_time * 1000, 2),
                            )
                            letter_index += 1

                            if DEBUG_TIMING:
                                print(
                                    f"{prediction} | brain={brain_time * 1000:6.1f}ms "
                                    f"key={key_time * 1000:5.1f}ms "
                                    f"cycle={cycle_time * 1000:6.1f}ms "
                                    f"~{wpm:6.1f} WPM"
                                )

                    if await test_finished(page):
                        await page.wait_for_timeout(300)
                        displayed = await page.evaluate(
                            """() => {
                                const r = document.querySelector('#result');
                                const value = name => r?.querySelector('.group.' + name + ' .bottom')?.textContent.trim() ?? null;
                                return {wpm:value('wpm'), accuracy:value('acc'), raw_wpm:value('raw'), chars:value('chars')};
                            }"""
                        )
                finally:
                    if screencast_task is not None:
                        screencast_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await screencast_task
                    if cdp is not None:
                        with contextlib.suppress(Exception):
                            await cdp.send("Page.stopScreencast")
                    await browser.close()
        except Exception as exc:
            stop_reason = "runtime_error"
            print("Runtime error:", exc)
            await self.emit("error", message=str(exc))

        elapsed = time.perf_counter() - start_time
        summary = {
            "stop_reason": stop_reason,
            "words": total_words,
            "characters": total_letters,
            "inference_captures": self.capture_count - captures_before,
            "elapsed_seconds": round(elapsed, 2),
            "wpm": round(total_keys / max(elapsed, 0.001) * 60 / 5, 1),
            "monkeytype": displayed,
        }
        await self.emit("session_finished", **summary)
        return summary


async def main():
    fly = FlyType()

    await fly.run()


if __name__ == "__main__":
    asyncio.run(
        main()
    )
