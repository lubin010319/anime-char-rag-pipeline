"""Verify pixels, apply image QA rules and drop pHash near-duplicates."""
import argparse
from collections import Counter

import imagehash
from PIL import Image

from common import ASSETS, BLUE, CLEAN, RAW, TEAL, open_rgb, plotting, read_rows, write_rows


def draw(rows, rejects, kept, decoded, size_ok):
    plt = plotting()
    fig = plt.figure(figsize=(13, 4.5), layout="constrained")
    axes = fig.subplots(1, 4, gridspec_kw={"width_ratios": [1.45, 1, 1, 1]})
    counts = [len(rows), decoded, size_ok, len(kept)]
    bars = axes[0].barh(["Input", "Decoded", "Size & ratio", "Unique"], counts,
                        color=[BLUE] * 3 + [TEAL], height=0.59)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, max(counts) + 4)
    axes[0].bar_label(bars, padding=5, weight="bold")
    axes[0].set_title("Quality funnel", loc="left", weight="bold")
    axes[0].tick_params(left=False, bottom=False, labelbottom=False)
    repeat = next((row for row in rejects if row["id"] == "repeat"), None)
    repeat = repeat or next((row for row in rejects if "duplicate_of" in row), None)
    low = next((row for row in rejects if row["id"] == "lowres"), None)
    if repeat:
        original = next(row for row in rows if row["id"] == repeat["duplicate_of"])
        examples = [(original, "A   Source image"),
                    (repeat, f'B   Near duplicate\npHash distance: {repeat["distance"]}')]
        if low:
            examples.append((low, f'C   Low-resolution control\n{low["size"][0]} × {low["size"][1]} px'))
        for ax, (row, label) in zip(axes[1:], examples):
            ax.imshow(open_rgb(RAW / row["file"]))
            ax.set_title(label, weight="bold", fontsize=11)
    for ax in axes[1:]:
        ax.axis("off")
    fig.suptitle("Validate before indexing", x=0.02, ha="left", weight="bold", fontsize=18)
    fig.savefig(ASSETS / "dedup_pipeline_summary.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-side", type=int, default=192)
    parser.add_argument("--max-ratio", type=float, default=3)
    parser.add_argument("--hash-distance", type=int, default=5)
    parser.add_argument("--demo-defects", action="store_true")
    args = parser.parse_args()
    if args.min_side < 1 or args.max_ratio < 1 or not 0 <= args.hash_distance <= 64:
        parser.error("Use a positive min-side, ratio >= 1 and a pHash threshold in 0..64")
    rows = read_rows(RAW / "active.jsonl")
    if args.demo_defects:
        source = rows[0]
        image = open_rgb(RAW / source["file"])
        for kind, scale in [("repeat", 1.25), ("lowres", 0.18)]:
            filename = f"control_{kind}.jpg"
            image.resize(tuple(max(1, round(side * scale)) for side in image.size),
                         Image.Resampling.LANCZOS).save(RAW / filename, quality=76)
            rows.append({**source, "id": kind, "file": filename, "control_of": source["id"]})
        (RAW / "control_broken.png").write_bytes(b"deliberately invalid PNG")
        rows.append({**source, "id": "broken", "file": "control_broken.png", "control_of": source["id"]})
    kept, rejects, hashes = [], [], []
    decoded = size_ok = 0
    CLEAN.mkdir(parents=True, exist_ok=True)
    for row in rows:
        path = RAW / row["file"]
        try:
            with Image.open(path) as probe:
                probe.verify()
            image = open_rgb(path)
        except (OSError, Image.DecompressionBombError):
            rejects.append({**row, "reason": "unreadable"})
            continue
        decoded += 1
        width, height = image.size
        if min(width, height) < args.min_side or max(width, height) / min(width, height) > args.max_ratio:
            rejects.append({**row, "reason": "low resolution or extreme ratio", "size": image.size})
            continue
        size_ok += 1
        fingerprint = imagehash.phash(image)
        nearest = min(((fingerprint - prior, uid) for prior, uid in hashes), default=(65, None))
        if nearest[0] <= args.hash_distance:
            rejects.append({**row, "reason": "near duplicate",
                            "duplicate_of": nearest[1], "distance": int(nearest[0])})
            continue
        hashes.append((fingerprint, row["id"]))
        name = f'{row["id"]}.jpg'
        image.thumbnail((960, 960))
        image.save(CLEAN / name, quality=93)
        kept.append({**row, "raw_file": row["file"], "file": name,
                     "source_size": [width, height], "size": list(image.size), "phash": str(fingerprint)})
    if not kept:
        raise SystemExit("No images passed the QA rules.")
    write_rows(CLEAN / "metadata.jsonl", kept)
    write_rows(CLEAN / "rejected.jsonl", rejects)
    draw(rows, rejects, kept, decoded, size_ok)
    print(f"Kept {len(kept)}/{len(rows)}. Reject reasons: {dict(Counter(r['reason'] for r in rejects))}")


if __name__ == "__main__":
    main()
