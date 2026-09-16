from io import BytesIO
import math
import re

import numpy as np
from PIL import Image


PADDING = 12
FOCUS_SETTLE_MS = 800


def pad_to_training_height(arr, target_height):
    """Restore the model's vertical canvas without resizing any glyph pixels."""
    if arr.ndim != 2 or not arr.size:
        raise ValueError("Expected a nonempty grayscale glyph array")
    height, width = arr.shape
    if height > target_height:
        raise ValueError(
            f"Glyph crop height {height} exceeds model height {target_height}; "
            "use the font size used for training"
        )
    if height == target_height:
        return arr
    output = np.full((target_height, width), np.median(arr), dtype=arr.dtype)
    # A 58px row needs 6px above / 5px below to share the training baseline.
    # Giving the spare row to the bottom shifts short crops up by one pixel.
    top = (target_height - height + 1) // 2
    output[top:top + height] = arr
    return output


async def focus_words(page):
    """Dismiss Monkeytype's blur overlay using its normal focus affordance."""
    await page.bring_to_front()
    overlay = page.get_by_text("Click here or press any key to focus", exact=False)
    if await overlay.count() and await overlay.first.is_visible():
        # The overlay text is pointer-transparent; click its screen position so
        # Monkeytype's containing test area receives the normal focus event.
        box = await overlay.first.bounding_box()
        if box:
            await page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    await page.wait_for_function(
        """() => { const words = document.querySelector('#words');
        if (!words || words.classList.contains('blurred')) return false;
        for (let el = words; el; el = el.parentElement) {
            const style = getComputedStyle(el);
            if (style.visibility === 'hidden' || Number(style.opacity) < 0.99) return false;
            if (style.filter !== 'none' && style.filter !== 'blur(0px)') return false;
        }
        return true; }""",
        timeout=10000,
    )


async def test_finished(page):
    if page.is_closed():
        return True
    return await page.evaluate("""() => {
        const result = document.querySelector('#result');
        return !!result && result.getClientRects().length > 0 &&
               getComputedStyle(result).visibility !== 'hidden';
    }""")


async def prepare_monkeytype(page):
    from playwright.async_api import TimeoutError as PlaywrightTimeoutError
    reject = page.get_by_role(
        "button",
        name=re.compile(r"reject\s+(all|non-essential)", re.I),
    )
    try:
        await reject.first.wait_for(state="visible", timeout=10000)
    except PlaywrightTimeoutError:
        pass
    else:
        await reject.first.click(force=True)
        try:
            await reject.first.wait_for(state="hidden", timeout=5000)
        except PlaywrightTimeoutError:
            pass
    await page.locator("#words").wait_for(state="visible")
    await page.evaluate("document.fonts.ready")
    
    # Set to a 30-second time test using the command palette
    await page.keyboard.press("Escape")
    await page.wait_for_timeout(200)
    await page.keyboard.type("time 30")
    await page.wait_for_timeout(200)
    await page.keyboard.press("Enter")
    await page.wait_for_timeout(500)
    
    await hide_caret(page)
    await focus_words(page)
    # Monkeytype removes the `blurred` class before its opacity/filter transition
    # has visually completed. Capturing immediately makes the first word faint.
    await page.wait_for_timeout(FOCUS_SETTLE_MS)


async def hide_caret(page):
    await page.add_style_tag(
        content="""
        #caret,
        .caret {
            opacity: 0 !important;
            visibility: hidden !important;
        }
        """
    )


async def capture_words(page, include_bytes=False):
    data = await page.screenshot(
        type="png",
        full_page=False,
        scale="css"
    )

    img = Image.open(
        BytesIO(data)
    ).convert("L")

    viewport = {
        "x": 0,
        "y": 0,
        "width": img.width,
        "height": img.height
    }

    if include_bytes:
        return img, viewport, data

    return img, viewport


def crop_letter(img, viewport, letter_box):
    x1 = math.floor(
        letter_box["x"]
    )

    y1 = math.floor(
        letter_box["y"]
    )

    x2 = math.ceil(
        letter_box["x"]
        + letter_box["width"]
    )

    y2 = math.ceil(
        letter_box["y"]
        + letter_box["height"]
    )

    x1 = max(
        0,
        min(x1, img.width)
    )

    y1 = max(
        0,
        min(y1, img.height)
    )

    x2 = max(
        0,
        min(x2, img.width)
    )

    y2 = max(
        0,
        min(y2, img.height)
    )

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Invalid crop: "
            f"{x1},{y1},{x2},{y2}"
        )

    letter = img.crop(
        (
            x1,
            y1,
            x2,
            y2
        )
    )

    pixels = np.asarray(
        img,
        dtype=np.uint8
    )

    background = int(
        np.median(pixels)
    )

    output = Image.new(
        "L",
        (
            letter.width + 2 * PADDING,
            letter.height + 2 * PADDING
        ),
        color=background
    )

    output.paste(
        letter,
        (
            PADDING,
            PADDING
        )
    )

    return output


def image_to_array(img):
    return (
        np.asarray(
            img,
            dtype=np.float32
        )
        / 255.0
    )
