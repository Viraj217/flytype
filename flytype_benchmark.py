"""Controlled visual representation comparison; never replaces the live model."""
import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import time
import warnings

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
LETTERS = "abcdefghijklmnopqrstuvwxyz"


def digest_image(image):
    return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def read_samples(folder):
    samples = []
    for label in LETTERS:
        for path in sorted((folder / label).glob("*.png")):
            with Image.open(path) as source:
                image = source.convert("L")
            samples.append(dict(path=str(path.resolve()), label=label,
                                digest=digest_image(image), size=image.size))
    if not samples:
        raise ValueError(f"No letter-folder PNGs found in {folder}")
    labels_by_hash = {}
    for row in samples:
        previous = labels_by_hash.setdefault(row["digest"], row["label"])
        if previous != row["label"]:
            raise ValueError("Identical pixels have conflicting teacher labels")
    return samples


def audit(samples):
    return {"images": len(samples), "unique_pixels": len({s["digest"] for s in samples}),
            "per_letter": {c: {"images": sum(s["label"] == c for s in samples),
                                "unique_pixels": len({s["digest"] for s in samples if s["label"] == c})}
                           for c in LETTERS}}


def raw_features(image, size):
    # Preserve source pixels. Only the raw baseline needs a fixed feature width.
    if image.width > size[0] or image.height > size[1]:
        raise ValueError(f"Validation crop {image.size} exceeds training canvas {size}")
    canvas = Image.new("L", size, int(np.median(np.asarray(image))))
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return np.asarray(canvas, dtype=np.float32).ravel() / 255.0


def checkpoint_counts(log, n, checkpoints):
    counts = np.zeros(n, dtype=np.uint16)
    result = {}
    for step, fired in enumerate(log, 1):
        np.add.at(counts, fired, 1)
        if step in checkpoints:
            result[step] = counts.copy()
    if set(result) != set(checkpoints):
        raise ValueError("Incomplete simulator spike log")
    return result


def fingerprint():
    digest = hashlib.sha256()
    for relative in ("flytype_benchmark.py", "flytype_vision.py", "flyeye.py", "flysim.py",
                     "build/graph.npz", "data/body-annotations.feather"):
        with (ROOT / relative).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


class Extractor:
    def __init__(self, cache):
        from flysim import FlyBrain
        from flyeye import FlyEye
        self.fb = FlyBrain()
        self.eye = FlyEye(self.fb, str(ROOT / "data/body-annotations.feather"))
        self.cache = cache / fingerprint()
        self.cache.mkdir(parents=True, exist_ok=True)

    def features(self, sample):
        cache_file = self.cache / (sample["digest"] + ".npz")
        if cache_file.exists():
            with np.load(cache_file, allow_pickle=False) as saved:
                return {key: saved[key] for key in saved.files}
        from flytype_vision import image_to_array
        with Image.open(sample["path"]) as source:
            arr = image_to_array(source.convert("L"))
        h, w = arr.shape
        drive = self.eye.look(arr, cx=w // 2, cy=h // 2, fov_w=w, fov_h=h, max_hz=180.0)
        key = tuple(self.eye.on_idx)
        l1 = drive[key]
        started = time.perf_counter()
        result = self.fb.run({key: l1}, steps=200, seed=42, spike_log=True)
        elapsed = time.perf_counter() - started
        checkpoints = checkpoint_counts(result["_spikes"], self.fb.n, (50, 100, 200))
        output = {"l1": l1, "brain_ms_200": np.array(elapsed * 1000)}
        output.update({f"brain{s}": value for s, value in checkpoints.items()})
        temporary = cache_file.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **output)
        temporary.replace(cache_file)
        return output


def select_features(X, y, k):
    from sklearn.feature_selection import f_classif
    # Constant neurons provide no discrimination; retain genuinely infinite F scores.
    varying = np.flatnonzero(np.asarray(X.max(axis=0).toarray()).ravel() !=
                             np.asarray(X.min(axis=0).toarray()).ravel())
    if not len(varying):
        raise ValueError("All training features are constant")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        scores, _ = f_classif(X[:, varying], y)
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    return varying[np.argsort(scores, kind="stable")[-min(k, len(varying)):]]


def metrics(y, predicted, labels):
    from sklearn.metrics import accuracy_score, confusion_matrix
    per_letter = {}
    for c in labels:
        mask = y == c
        per_letter[c] = {"count": int(mask.sum()),
                         "accuracy": float(np.mean(predicted[mask] == y[mask])) if mask.any() else None}
    return {"accuracy": float(accuracy_score(y, predicted)),
            "macro_accuracy_observed_letters": float(np.mean([r["accuracy"] for r in per_letter.values()
                                                              if r["count"]])),
            "per_letter": per_letter,
            "confusion_matrix": confusion_matrix(y, predicted, labels=labels).tolist()}


def benchmark(args):
    import joblib
    import scipy.sparse as sp
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.exceptions import ConvergenceWarning

    train = read_samples(args.train)
    report = {"train": audit(train)}
    a, b = train_test_split(np.arange(len(train)), test_size=.2, random_state=42,
                            stratify=[s["label"] for s in train])
    seen = {train[i]["digest"] for i in a}
    report["legacy_random_holdout_overlap"] = {
        "test_images": len(b), "exact_matches_in_train": sum(train[i]["digest"] in seen for i in b)}
    args.output.mkdir(parents=True, exist_ok=True)
    if args.audit_only:
        (args.output / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return
    if args.validation is None:
        raise ValueError("--validation is required; error-only crops are not a validation set")
    if args.train.resolve() == args.validation.resolve():
        raise ValueError("Training and validation must be separate collections")
    validation = read_samples(args.validation)
    train_hashes = {s["digest"] for s in train}
    novel = np.array([s["digest"] not in train_hashes for s in validation])
    report.update(validation=audit(validation), validation_exact_matches_in_train=int((~novel).sum()))
    labels = sorted({s["label"] for s in train})
    if set(s["label"] for s in validation) - set(labels):
        raise ValueError("Validation includes letters absent from training")
    report["missing_validation_letters"] = sorted(set(labels) - {s["label"] for s in validation})
    report["protocol"] = {"C": args.c, "top_k": args.top_k,
                          "scaling": "StandardScaler fitted on training only",
                          "selection": "ANOVA fitted on training only; fixed k and C; no validation tuning",
                          "seed": 42, "max_hz": 180, "steps": [50, 100, 200],
                          "raw": "native grayscale pixels, centered background padding to training maximum size"}
    all_samples = train + validation
    (args.output / "samples.json").write_text(json.dumps({"train": train, "validation": validation}, indent=2), encoding="utf-8")
    canvas = tuple(max(s["size"][axis] for s in train) for axis in (0, 1))
    # Check every raw canvas before the expensive simulation.
    raw = []
    for sample in all_samples:
        with Image.open(sample["path"]) as source:
            raw.append(raw_features(source.convert("L"), canvas))
    extractor = Extractor(args.output / "cache")
    report["extractor_fingerprint"] = extractor.cache.name
    unique = {s["digest"]: s for s in all_samples}
    for i, sample in enumerate(unique.values(), 1):
        extractor.features(sample)
        if i % 25 == 0 or i == len(unique):
            print(f"Representations: {i}/{len(unique)} distinct images", flush=True)
    y_train = np.array([s["label"] for s in train])
    y_valid = np.array([s["label"] for s in validation])
    report["results"] = {}
    predictions = []
    for representation in ("raw", "l1", "brain50", "brain100", "brain200"):
        if representation == "raw":
            X = sp.csr_matrix(np.stack(raw))
        else:
            X = sp.vstack([sp.csr_matrix(extractor.features(s)[representation].astype(np.float32).reshape(1, -1))
                           for s in all_samples], format="csr")
        X_train, X_valid = X[:len(train)], X[len(train):]
        selected = select_features(X_train, y_train, args.top_k)
        scaler = StandardScaler()
        fit_X = scaler.fit_transform(X_train[:, selected].toarray())
        test_X = scaler.transform(X_valid[:, selected].toarray())
        classifier = LogisticRegression(C=args.c, max_iter=5000, random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            classifier.fit(fit_X, y_train)
        predicted = classifier.predict(test_X)
        result = metrics(y_valid, predicted, labels)
        result["novel_pixels"] = metrics(y_valid[novel], predicted[novel], labels) if novel.any() else None
        result["selected_features"] = len(selected)
        report["results"][representation] = result
        joblib.dump(dict(classifier=classifier, scaler=scaler, selected=selected,
                         representation=representation, canvas=canvas, protocol=report["protocol"]),
                    args.output / f"{representation}.joblib")
        predictions.extend(dict(representation=representation, path=s["path"], actual=s["label"],
                                predicted=str(p), novel_pixels=bool(n))
                           for s, p, n in zip(validation, predicted, novel))
        (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"{representation}: {result['accuracy']:.2%}; macro {result['macro_accuracy_observed_letters']:.2%}", flush=True)
        del X, X_train, X_valid
    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    lines = ["# FlyType representation benchmark", "", "| Representation | Accuracy | Macro accuracy (observed letters) | Novel-pixel accuracy |",
             "|---|---:|---:|---:|"]
    for name, result in report["results"].items():
        novel_result = result["novel_pixels"]
        novel_text = f"{novel_result['accuracy']:.2%}" if novel_result else "N/A"
        lines.append(f"| {name} | {result['accuracy']:.2%} | {result['macro_accuracy_observed_letters']:.2%} | {novel_text} |")
    lines += ["", f"Training: {len(train)} images, {report['train']['unique_pixels']} distinct pixel images.",
              f"Validation: {len(validation)} images; {int((~novel).sum())} exactly match training pixels.",
              f"Missing validation letters: {', '.join(report['missing_validation_letters']) or 'none'}.",
              "", "This validation set informs architecture choice. Confirm the chosen design on another untouched session.",
              "Raw pixels and L1 are diagnostic baselines. Production inference remains visual and unchanged."]
    (args.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "monkeytype_dataset")
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "scratch/flytype_benchmark")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--c", type=float, default=.1)
    args = parser.parse_args()
    if args.top_k < 1 or args.c <= 0:
        parser.error("--top-k and --c must be positive")
    benchmark(args)


if __name__ == "__main__":
    main()
