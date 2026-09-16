"""Save every active-word glyph from a new Monkeytype session, without a classifier."""
import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import random
import uuid

import numpy as np
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

from flytype_vision import capture_words, crop_letter, hide_caret, prepare_monkeytype, focus_words

ROOT = Path(__file__).resolve().parent

GEOMETRY = """() => {
  const words = [...document.querySelectorAll('#words .word')];
  const index = words.findIndex(w => w.classList.contains('active'));
  if (index < 0) return null;
  return {index, letters: [...words[index].querySelectorAll('letter')].map(el => {
    const r = el.getBoundingClientRect();
    return {char: el.textContent, x: r.x, y: r.y, width: r.width, height: r.height,
            classes: el.className};
  })};
}"""


def stable(before, after):
    if not before or not after or before["index"] != after["index"]:
        return False
    if len(before["letters"]) != len(after["letters"]):
        return False
    for a, b in zip(before["letters"], after["letters"]):
        if a["char"] != b["char"] or a["classes"] != b["classes"]:
            return False
        if any(abs(a[key] - b[key]) > .01 for key in ("x", "y", "width", "height")):
            return False
    return True


async def set_alphabet_text(page, words, seed):
    """Use the site's custom-text UI to cover all letters in a fresh session."""
    rng = random.Random(seed)
    chars = list("abcdefghijklmnopqrstuvwxyz" * ((words * 5 + 25) // 26))
    rng.shuffle(chars)
    chars = chars[:words * 5]
    text = " ".join("".join(chars[i:i + 5]) for i in range(0, len(chars), 5))
    for command in ("custom", "change custom text"):
        await page.keyboard.press("Escape")
        await page.wait_for_timeout(200)
        await page.keyboard.type(command)
        await page.wait_for_timeout(200)
        await page.keyboard.press("Enter")
        await page.wait_for_timeout(700)
    textarea = page.locator("textarea:visible")
    await textarea.first.fill(text)
    ok = page.get_by_text("ok", exact=True)
    if await ok.count() and await ok.first.is_visible():
        await ok.first.click()
    else:
        await page.keyboard.press("Control+Enter")
    await page.wait_for_timeout(800)
    await hide_caret(page)
    await focus_words(page)
    return text


async def collect(args):
    session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    output = args.output / session
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(session=session, purpose="independent_validation", requested_words=args.words,
                    viewport=dict(width=1280, height=720), device_scale_factor=1,
                    selection="all lowercase glyphs in consecutive active words; no model or error filtering",
                    headless=not args.headed, status="incomplete")
    metadata["mode"] = "custom_alphabet" if args.alphabet else "default_timed_english"
    count = Counter()
    crop_sizes = {}
    processed = 0
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=not args.headed)
        try:
            context = await browser.new_context(viewport=metadata["viewport"], device_scale_factor=1)
            page = await context.new_page()
            await page.goto("https://monkeytype.com", wait_until="domcontentloaded")
            await prepare_monkeytype(page)
            if args.alphabet:
                metadata["custom_text"] = await set_alphabet_text(page, args.words, args.seed)
                metadata["custom_seed"] = args.seed
            await page.wait_for_timeout(1000)
            metadata["browser_version"] = browser.version
            metadata["rendering"] = await page.evaluate("""() => {
                const el = document.querySelector('#words letter');
                const s = getComputedStyle(el);
                return {font: s.font, color: s.color, lineHeight: s.lineHeight,
                        background: getComputedStyle(document.body).backgroundColor,
                        dpr: devicePixelRatio, userAgent: navigator.userAgent};
            }""")
            with (output / "manifest.jsonl").open("w", encoding="utf-8") as manifest:
                retries = 0
                while processed < args.words:
                    await focus_words(page)
                    before = await page.evaluate(GEOMETRY)
                    if not before:
                        metadata["stop_reason"] = "No active word (test may have finished)"
                        break
                    image, viewport = await capture_words(page)
                    if processed == 0:
                        image.save(output / "first_viewport.png")
                    after = await page.evaluate(GEOMETRY)
                    if not stable(before, after):
                        retries += 1
                        if retries > 40:
                            raise RuntimeError("Layout did not settle after 40 captures")
                        await page.wait_for_timeout(50)
                        continue
                    retries = 0
                    letters = after["letters"]
                    if any("correct" in s["classes"] or "incorrect" in s["classes"] for s in letters):
                        raise RuntimeError("Active word was already typed; refusing repeated or colored samples")
                    pending = []
                    for i, info in enumerate(letters):
                        char = info["char"]
                        if len(char) != 1 or char not in "abcdefghijklmnopqrstuvwxyz":
                            continue
                        if (info["x"] < 0 or info["y"] < 0 or info["width"] <= 0 or info["height"] <= 0
                                or info["x"] + info["width"] > image.width
                                or info["y"] + info["height"] > image.height):
                            raise RuntimeError("Letter is clipped by viewport; collection stopped")
                        crop = crop_letter(image, viewport, info)
                        if np.ptp(np.asarray(crop).astype(float)) < 8:
                            crop.save(output / "blank_crop.png")
                            raise RuntimeError(f"Blank crop detected at {info}; collection stopped")
                        pending.append((i, info, crop))
                    if processed == 0:
                        image.save(output / "first_viewport.png")
                    for i, info, crop in pending:
                        folder = output / info["char"]
                        folder.mkdir(exist_ok=True)
                        path = folder / f"{processed:05d}_{i:02d}.png"
                        crop.save(path)
                        crop_dim = f"{crop.width}x{crop.height}"
                        crop_sizes[crop_dim] = crop_sizes.get(crop_dim, 0) + 1
                        manifest.write(json.dumps(dict(path=str(path.relative_to(output)), label=info["char"],
                                                       session=session, word=processed, geometry=info,
                                                       crop_width=crop.width, crop_height=crop.height)) + "\n")
                        count[info["char"]] += 1
                    manifest.flush()
                    processed += 1
                    # DOM teacher text only advances the collection browser; no prediction uses it.
                    await page.keyboard.type("".join(s["char"] for s in letters), delay=0)
                    await page.keyboard.press("Space")
                    await page.wait_for_timeout(30)
                    if processed % 20 == 0:
                        print(f"Collected {processed} words, {sum(count.values())} glyphs", flush=True)
            metadata["status"] = "complete" if processed == args.words else "partial"
        finally:
            metadata.update(collected_words=processed, per_letter=dict(count), total_images=sum(count.values()),
                            crop_sizes=crop_sizes)
            (output / "session.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            missing = sorted(set("abcdefghijklmnopqrstuvwxyz") - set(count))
            print(f"Validation folder: {output}", flush=True)
            print(f"Coverage: {len(count)}/26 letters, {sum(count.values())} glyphs", flush=True)
            print(f"Crop sizes: {crop_sizes}", flush=True)
            if missing:
                print(f"Missing letters: {', '.join(missing)}", flush=True)
            await browser.close()
    if not count:
        raise RuntimeError("No glyphs collected")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--words", type=int, default=100)
    parser.add_argument("--output", type=Path, default=ROOT / "scratch/flytype_validation")
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--alphabet", action="store_true", help="Use balanced custom text (104 words = 20 samples per letter)")
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()
    if args.words < 1:
        parser.error("--words must be positive")
    asyncio.run(collect(args))


if __name__ == "__main__":
    main()
