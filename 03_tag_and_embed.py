"""Use WD14 ONNX, clean character tags and build two exact cosine Faiss indexes."""
import argparse
import json
from collections import Counter

import faiss
import numpy as np

from common import ASSETS, BLUE, CLEAN, INDEX, JA, TEAL, open_rgb, plotting, read_rows, write_rows
from models import EMBED_MODEL, WD_MODEL, WD_REVISION, Embedder, WDTagger, character_tags


def plot_tags(raw, clean):
    plt = plotting()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.7), layout="constrained")
    for ax, data, color, title in zip(axes, (raw, clean), (BLUE, TEAL),
                                     ("Raw WD14 general tags", "Final character attributes")):
        items = data.most_common(15)[::-1]
        names = [name.replace("_", " ") for name, _ in items]
        counts = [count for _, count in items]
        bars = ax.barh(names, counts, color=color)
        ax.bar_label(bars, padding=4, fontsize=9)
        ax.set_xlim(0, max(counts, default=1) + 3)
        ax.tick_params(axis="y", length=0, labelsize=9)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Images with tag")
    fig.suptitle("Keep tags useful for character search", x=0.02, ha="left", weight="bold", fontsize=18)
    fig.savefig(ASSETS / "tag_distribution.png", dpi=150)
    plt.close(fig)


def write_index(name, vectors):
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(INDEX / f"{name}.faiss"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.30)
    args = parser.parse_args()
    if args.batch_size < 1 or not 0 < args.threshold < 1:
        parser.error("Use a positive batch size and threshold between 0 and 1")
    rows = read_rows(CLEAN / "metadata.jsonl")
    tagger = WDTagger(args.threshold)
    raw_count, clean_count = Counter(), Counter()
    for row in rows:
        raw = tagger(open_rgb(CLEAN / row["file"]))
        wd_tags = character_tags(raw)
        final = sorted(set(wd_tags) | set(row["reviewed_tags"]))
        raw_count.update(raw.keys())
        clean_count.update(final)
        row.update(wd14_scores=raw, wd14_character_tags=wd_tags, character_tags=final,
                   caption_en=row["character"] + ". " + ", ".join(x.replace("_", " ") for x in final),
                   caption_ja="、".join(JA[x] for x in final))
    model = Embedder(args.device)
    image_vectors, text_vectors = [], []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        image_vectors.append(model.vector(images=[open_rgb(CLEAN / row["file"]) for row in batch]))
        text_vectors.append(model.vector(texts=[row["caption_en"].lower() for row in batch]))
    INDEX.mkdir(parents=True, exist_ok=True)
    write_index("images", np.concatenate(image_vectors))
    write_index("captions", np.concatenate(text_vectors))
    write_rows(INDEX / "items.jsonl", rows)
    (INDEX / "config.json").write_text(json.dumps({
        "embedding_model": EMBED_MODEL, "embedding_revision": model.revision,
        "tagger": WD_MODEL, "tagger_revision": WD_REVISION, "wd_threshold": args.threshold,
        "rows": len(rows), "embedding_device": model.device,
    }, indent=2), encoding="utf-8")
    plot_tags(raw_count, clean_count)
    print(f"WD14 CPU + SigLIP 2 ({model.device}) + two Faiss indexes: {len(rows)} images")


if __name__ == "__main__":
    main()
