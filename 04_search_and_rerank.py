"""Search by English/Japanese text or image, then rerank the recalled candidates."""
import argparse
import json
import re
from pathlib import Path

import faiss

from common import ASSETS, BLUE, CLEAN, INDEX, JA, TEAL, open_rgb, plotting, read_rows
from models import Embedder, WDTagger, character_tags

DEMOS = {
    "青髪のメイド、エプロン、髪飾り": ("blue_hair", "maid", "apron", "hair_ornament"),
    "anime girl in a red jacket": ("red_jacket",),
    "制服、セーラー襟": ("school_uniform", "sailor_collar"),
}


def query_tags(text):
    english = re.sub(r"[-_]", " ", text.lower())
    return {tag for tag, ja in JA.items()
            if ja in text or re.search(r"\b" + re.escape(tag.replace("_", " ")) + r"\b", english)}


def search(vector, index, rows, tags, top_k, weight, exclude=None):
    scores, ids = index.search(vector, min(index.ntotal, top_k + int(exclude is not None)))
    before = []
    for score, row_id in zip(scores[0], ids[0]):
        if row_id < 0 or rows[row_id]["id"] == exclude:
            continue
        item = rows[row_id]
        overlap = len(tags.intersection(item["character_tags"])) / len(tags) if tags else 0.0
        cosine = float(score)
        combined = (1 - weight) * cosine + weight * overlap if tags else cosine
        before.append({"id": item["id"], "file": item["file"], "character": item["character"],
                       "cosine": cosine, "overlap": overlap, "rerank": combined})
    before = before[:top_k]
    return before, sorted(before, key=lambda row: row["rerank"], reverse=True)


def draw(label, before, after, weight):
    plt = plotting()
    fig, axes = plt.subplots(2, 5, figsize=(12.6, 6.8))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.76, bottom=0.05, wspace=0.08, hspace=0.48)
    fig.text(0.02, 0.935, f"Search / {label}", weight="bold", fontsize=18)
    fig.text(0.02, 0.88, f"Faiss cosine → attribute overlap  ·  attribute weight {weight:.2f}", fontsize=11)
    fig.text(0.02, 0.81, "A   VECTOR RETRIEVAL", color=BLUE, weight="bold")
    fig.text(0.02, 0.405, "B   AFTER RERANK", color=TEAL, weight="bold")
    for y, entries in enumerate((before[:5], after[:5])):
        for x, ax in enumerate(axes[y]):
            ax.axis("off")
            if x >= len(entries):
                continue
            row = entries[x]
            ax.imshow(open_rgb(CLEAN / row["file"]))
            score = f'cos {row["cosine"]:.2f}' if y == 0 else f'score {row["rerank"]:.2f} · tags {row["overlap"]:.0%}'
            ax.set_title(f'{x + 1:02d}  {row["character"]}\n{score}', fontsize=9, loc="left")
    fig.savefig(ASSETS / "retrieval_rerank_demo.png", dpi=155)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    query = parser.add_mutually_exclusive_group(required=True)
    query.add_argument("--text")
    query.add_argument("--image", type=Path)
    query.add_argument("--demo", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--tag-weight", type=float, default=0.25)
    parser.add_argument("--save-figure", action="store_true")
    args = parser.parse_args()
    if args.top_k < 5 or not 0 <= args.tag_weight <= 1:
        parser.error("Use top-k >= 5 and a tag weight between 0 and 1")
    rows = read_rows(INDEX / "items.jsonl")
    config = json.loads((INDEX / "config.json").read_text(encoding="utf-8"))
    index = faiss.read_index(str(INDEX / "images.faiss"))
    if index.ntotal != len(rows):
        raise SystemExit("Index and metadata differ. Run stage 3 again.")
    model = Embedder(args.device, revision=config["embedding_revision"])

    def run_text(text):
        return search(model.vector(texts=[text.lower()]), index, rows, query_tags(text),
                      args.top_k, args.tag_weight)

    def run_image(path, exclude):
        image = open_rgb(path)
        tags = character_tags(WDTagger(config["wd_threshold"])(image))
        return search(model.vector(images=[image]), index, rows, set(tags),
                      args.top_k, args.tag_weight, exclude=exclude)

    if args.demo:
        results = []
        for query_index, (prompt, target) in enumerate(DEMOS.items()):
            before, after = run_text(prompt)
            relevant = {row["id"] for row in rows if set(target).issubset(row["reviewed_tags"])}
            ranks = [next((i for i, row in enumerate(result[:5], 1) if row["id"] in relevant), None)
                     for result in (before, after)]
            recall = [sum(r["id"] in relevant for r in result[:5]) / len(relevant) if relevant else None
                      for result in (before, after)]
            results.append({"query": prompt, "relevant": len(relevant), "first_relevant_rank_at_5": ranks, "recall_at_5": recall,
                            "before_ids": [r["id"] for r in before[:5]], "after_ids": [r["id"] for r in after[:5]]})
            if query_index == 1:
                draw(prompt, before, after, args.tag_weight)
        leave_one_out = next((row for row in rows if row["character"] == "Wikipe-tan"), rows[0])
        before, after = run_image(CLEAN / leave_one_out["file"], leave_one_out["id"])
        report = {"text_queries": results, "image_query": {"excluded_id": leave_one_out["id"],
                  "before": before[:5], "after": after[:5]}}
        (INDEX / "demo_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if args.text is not None:
        before, after = run_text(args.text)
        label = args.text
    else:
        exclude = next((row["id"] for row in rows if (CLEAN / row["file"]).resolve() == args.image.resolve()), None)
        before, after = run_image(args.image, exclude)
        label = args.image.name
    if args.save_figure:
        draw(label, before, after, args.tag_weight)
    print(json.dumps({"before": before[:5], "after": after[:5]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
