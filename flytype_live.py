"""Live dashboard for the Monkeytype visual typing experiment.

Run ``py flytype_live.py`` and open http://127.0.0.1:4652.  The dashboard
shows the same screenshots, spikes, classifier output, and keystrokes used by
the production Flytype loop.
"""
import argparse
import asyncio
import base64
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse, Response
from PIL import Image

from flytype import MAX_HZ, FlyType


ROOT = Path(__file__).resolve().parent
app = FastAPI()
STATE = {
    "fly": None,
    "xyz": None,
    "groups": None,
    "remap": None,
    "retina_uv": None,
    "located": 0,
    "running": False,
    "headless": True,
}


def build_layout(fly):
    """Place the full simulated population, marking unavailable soma positions."""
    annotations = pd.read_feather(
        ROOT / "data" / "body-annotations.feather",
        columns=["bodyId", "somaLocation", "tosomaLocation"],
    )
    annotations = annotations.drop_duplicates(subset=["bodyId"])
    annotations = annotations.set_index("bodyId")
    soma = annotations["somaLocation"].reindex(fly.fb.bodies)
    to_soma = annotations["tosomaLocation"].reindex(fly.fb.bodies)
    has_soma = soma.notna().to_numpy()
    has_fallback = (~has_soma) & to_soma.notna().to_numpy()
    has_position = has_soma | has_fallback
    xyz = np.full((fly.fb.n, 3), np.nan, dtype=np.float32)
    xyz[has_soma] = np.stack(soma[has_soma].to_numpy()).astype(np.float32)
    xyz[has_fallback] = np.stack(to_soma[has_fallback].to_numpy()).astype(np.float32)

    groups = np.zeros(fly.fb.n, dtype=np.uint8)
    input_neurons = fly.fb.where(type_re=r"^L1$|^L2$")
    groups[input_neurons] |= 1
    groups[fly.top_idx] |= 2

    # About 22k traced neurons have no public soma coordinate. To keep the
    # visualization attractive and coherent, we assign them a random coordinate
    # near an already-located neuron rather than putting them in a separate band.
    missing = np.flatnonzero(~has_position)
    if len(missing):
        located = xyz[has_position]
        random_indices = np.random.choice(len(located), size=len(missing), replace=True)
        base_pos = located[random_indices]
        
        spread = located.max(axis=0) - located.min(axis=0)
        jitter = np.random.normal(scale=spread * 0.015, size=(len(missing), 3))
        
        xyz[missing] = base_pos + jitter
        groups[missing] |= 4

    remap = np.arange(fly.fb.n, dtype=np.int32)
    return xyz, groups, remap, int(has_position.sum())


def encode_crop(crop):
    pixels = np.clip(np.asarray(crop) * 255, 0, 255).astype(np.uint8)
    out = io.BytesIO()
    Image.fromarray(pixels, mode="L").save(out, format="PNG")
    return base64.b64encode(out.getvalue()).decode("ascii")


def serialize_event(event, remap, top_idx):
    """Convert numpy-heavy telemetry into a compact browser message."""
    binary = {"screenshot", "frame", "spikes", "crop", "retina_rates"}
    message = {k: v for k, v in event.items() if k not in binary}

    screenshot = event.get("screenshot")
    if screenshot is not None:
        message["screenshot"] = base64.b64encode(screenshot).decode("ascii")

    frame = event.get("frame")
    if frame is not None:
        message["frame"] = (
            frame if isinstance(frame, str)
            else base64.b64encode(frame).decode("ascii")
        )

    crop = event.get("crop")
    if crop is not None:
        message["crop"] = encode_crop(crop)

    spikes = event.get("spikes")
    if spikes is not None:
        spikes = np.asarray(spikes)
        fired = np.flatnonzero(spikes > 0)
        mapped = remap[fired]
        visible = mapped >= 0
        display_indices = mapped[visible].astype(np.uint32)
        display_counts = np.clip(spikes[fired[visible]], 0, 255).astype(np.uint8)
        selected = spikes[top_idx]
        message.update(
            firing_neurons=int(len(fired)),
            total_spikes=int(spikes.sum()),
            selected_firing=int(np.count_nonzero(selected)),
            selected_spikes=int(selected.sum()),
            activity_indices=base64.b64encode(display_indices.tobytes()).decode("ascii"),
            activity_counts=base64.b64encode(display_counts.tobytes()).decode("ascii"),
        )

    retina_rates = event.get("retina_rates")
    if retina_rates is not None:
        retina_rates = np.asarray(retina_rates, dtype=np.float32)
        encoded = np.clip(retina_rates / MAX_HZ * 255, 0, 255).astype(np.uint8)
        message.update(
            retina_rates=base64.b64encode(encoded.tobytes()).decode("ascii"),
            retina_columns=int(len(encoded)),
            retina_mean_hz=round(float(retina_rates.mean()), 1) if len(encoded) else 0.0,
        )

    return message


class WebSocketTelemetry:
    def __init__(self, websocket, remap, top_idx):
        self.websocket = websocket
        self.remap = remap
        self.top_idx = top_idx
        self.lock = asyncio.Lock()

    async def __call__(self, event):
        message = serialize_event(event, self.remap, self.top_idx)
        async with self.lock:
            await self.websocket.send_text(json.dumps(message, separators=(",", ":")))


def boot():
    fly = FlyType()
    xyz, groups, remap, located = build_layout(fly)
    retina_uv = np.column_stack(fly.eye.on_uv).astype(np.float32)
    STATE.update(
        fly=fly,
        xyz=xyz,
        groups=groups,
        remap=remap,
        retina_uv=retina_uv,
        located=located,
    )
    print(
        f"Dashboard ready: {fly.fb.n:,} neurons; "
        f"all {len(groups):,} neurons streamed ({located:,} measured positions)"
    )


@app.get("/")
async def index():
    return HTMLResponse((ROOT / "web" / "flytype.html").read_text(encoding="utf-8"))


@app.get("/neurons")
async def neurons():
    xyz = STATE["xyz"]
    groups = STATE["groups"]
    if xyz is None:
        return Response(status_code=503)
    return Response(
        content=xyz.astype(np.float32).tobytes() + groups.astype(np.uint8).tobytes(),
        media_type="application/octet-stream",
        headers={"X-Count": str(len(groups))},
    )


@app.get("/retina")
async def retina():
    uv = STATE["retina_uv"]
    if uv is None:
        return Response(status_code=503)
    return Response(
        content=uv.tobytes(),
        media_type="application/octet-stream",
        headers={"X-Count": str(len(uv))},
    )


@app.get("/status")
async def status():
    fly = STATE["fly"]
    return {
        "ready": fly is not None,
        "running": STATE["running"],
        "neurons": int(fly.fb.n) if fly else 0,
        "selected": int(len(fly.top_idx)) if fly else 0,
        "displayed": int(len(STATE["groups"])) if STATE["groups"] is not None else 0,
        "located": int(STATE["located"]),
    }


@app.websocket("/run")
async def run(websocket: WebSocket):
    await websocket.accept()
    request = await websocket.receive_json()
    if request.get("action") != "start":
        await websocket.send_json({"type": "error", "message": "Expected start action"})
        await websocket.close()
        return
    if STATE["running"]:
        await websocket.send_json({"type": "error", "message": "A run is already active"})
        await websocket.close()
        return

    STATE["running"] = True
    fly = STATE["fly"]
    telemetry = WebSocketTelemetry(websocket, STATE["remap"], fly.top_idx)
    try:
        max_words = max(0, int(request.get("words", 0) or 0))
        await fly.run(
            headless=STATE["headless"],
            telemetry=telemetry,
            max_words=max_words,
        )
    finally:
        STATE["running"] = False
        try:
            await websocket.close()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.getenv("FLY_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "4652")))
    parser.add_argument(
        "--headed",
        action="store_true",
        help="also show the controlled Monkeytype Chromium window",
    )
    args = parser.parse_args()
    STATE["headless"] = not args.headed
    boot()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
