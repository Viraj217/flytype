"""Record the Flytype dashboard as a clean WebM and optional MP4.

Start ``py flytype_live.py`` first, then run this file.  Use ``--words`` for
a short bounded clip; zero records until the Monkeytype test finishes.
"""
import argparse
import asyncio
import shutil
import subprocess
import time
from pathlib import Path

import requests
from playwright.async_api import async_playwright


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "build" / "recordings"
FINAL_STATES = {"end_of_test", "word_limit", "runtime_error"}


async def record(args):
    base = f"http://127.0.0.1:{args.port}"
    status = requests.get(f"{base}/status", timeout=10).json()
    if not status.get("ready"):
        raise RuntimeError("Flytype dashboard is not ready")
    if status.get("running"):
        raise RuntimeError("A Flytype run is already active")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    url = base + (f"/?words={args.words}" if args.words else "/")

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": args.width, "height": args.height},
            record_video_dir=str(OUTPUT),
            record_video_size={"width": args.width, "height": args.height},
        )
        page = await context.new_page()
        page.on("pageerror", lambda error: print(f"dashboard error: {error}"))
        await page.goto(url, wait_until="networkidle", timeout=45000)
        await page.wait_for_function(
            "() => !document.getElementById('start').disabled",
            timeout=60000,
        )
        await page.wait_for_timeout(1500)
        await page.click("#start")
        print("Recording Flytype...")

        deadline = time.time() + args.timeout
        outcome = None
        while time.time() < deadline:
            await page.wait_for_timeout(1000)
            outcome = (await page.locator("#batchLabel").inner_text()).lower()
            if outcome in FINAL_STATES:
                break
        if outcome not in FINAL_STATES:
            raise TimeoutError(f"Dashboard did not finish within {args.timeout}s")

        await page.wait_for_timeout(args.hold * 1000)
        video = page.video
        await context.close()
        await browser.close()
        temporary = Path(await video.path())

    webm = OUTPUT / f"flytype-{stamp}.webm"
    temporary.replace(webm)
    print(f"Wrote {webm}")

    mp4 = None
    if shutil.which("ffmpeg"):
        mp4 = OUTPUT / f"flytype-{stamp}.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(webm),
                "-c:v", "libx264", "-preset", "slow", "-crf", "20",
                "-pix_fmt", "yuv420p", str(mp4),
            ],
            check=True,
        )
        print(f"Wrote {mp4}")
    return mp4 or webm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=4652)
    parser.add_argument("--words", type=int, default=0)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--hold", type=int, default=4)
    args = parser.parse_args()
    asyncio.run(record(args))


if __name__ == "__main__":
    main()
