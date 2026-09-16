"""Test training-derived vertical padding with the unchanged production classifier."""
import argparse
import json
from pathlib import Path

from flytype_benchmark import ROOT, Extractor, checkpoint_counts, metrics, read_samples
from flytype_vision import image_to_array
import joblib
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "scratch/flytype_benchmark")
    args = parser.parse_args()
    train = read_samples(ROOT / "monkeytype_dataset")
    samples = read_samples(args.validation)
    heights = {s["size"][1] for s in train}
    if len(heights) != 1:
        raise ValueError("Training crop height is not constant")
    target_height = heights.pop()
    model = joblib.load(ROOT / "models/flytype_model.joblib")
    selected = np.load(ROOT / "models/top_idx.npy")
    extractor = Extractor(args.output / "cache")
    predictions = {}
    for i, sample in enumerate({s["digest"]: s for s in samples}.values(), 1):
        original = extractor.features(sample)["brain50"]
        with Image.open(sample["path"]) as source:
            image = source.convert("L")
        if image.height > target_height:
            raise ValueError("Cannot restore background padding to a smaller canvas")
        padded = Image.new("L", (image.width, target_height), int(np.median(np.asarray(image))))
        padded.paste(image, (0, (target_height - image.height) // 2))
        arr = image_to_array(padded)
        h, w = arr.shape
        drive = extractor.eye.look(arr, cx=w // 2, cy=h // 2, fov_w=w, fov_h=h, max_hz=180)
        key = tuple(extractor.eye.on_idx)
        log = extractor.fb.run({key: drive[key]}, steps=50, seed=42, spike_log=True)["_spikes"]
        padded_features = checkpoint_counts(log, extractor.fb.n, (50,))[50]
        predictions[sample["digest"]] = tuple(str(p) for p in model.predict(
            np.stack([original[selected], padded_features[selected]]).astype(np.float32)))
        if i % 25 == 0:
            print(f"Padding control: {i} distinct images", flush=True)
    y = np.array([s["label"] for s in samples])
    labels = sorted({s["label"] for s in train})
    baseline = np.array([predictions[s["digest"]][0] for s in samples])
    padded = np.array([predictions[s["digest"]][1] for s in samples])
    report = dict(target_height_from_training=target_height, images=len(samples),
                  protocol="Existing 50-step model and feature indices, seed 42; only add vertical background padding; no refitting",
                  original=metrics(y, baseline, labels), padded=metrics(y, padded, labels),
                  missing_letters=sorted(set(labels) - set(y)),
                  predictions=[dict(path=s["path"], actual=s["label"], original=str(a), padded=str(b))
                               for s, a, b in zip(samples, baseline, padded)])
    (args.output / "padding_control.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Unchanged production model: original {report['original']['accuracy']:.2%}; padded {report['padded']['accuracy']:.2%}")


if __name__ == "__main__":
    main()
