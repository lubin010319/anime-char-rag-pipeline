"""Shared paths, JSONL and the controlled tag vocabulary."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
CLEAN = DATA / "clean"
INDEX = DATA / "index"
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

JA = {
    "blue_hair": "青髪", "pink_hair": "ピンク髪", "black_hair": "黒髪",
    "blonde_hair": "金髪", "white_hair": "白髪", "red_hair": "赤髪",
    "brown_hair": "茶髪", "purple_hair": "紫髪", "blue_eyes": "青い目",
    "red_eyes": "赤い目", "green_eyes": "緑の目", "long_hair": "ロングヘア",
    "short_hair": "ショートヘア", "twintails": "ツインテール",
    "maid": "メイド", "apron": "エプロン", "hair_ornament": "髪飾り",
    "casual": "私服", "sailor_collar": "セーラー襟", "school_uniform": "制服",
    "red_jacket": "赤いジャケット", "glasses": "眼鏡", "ribbon": "リボン",
    "hat": "帽子", "dress": "ドレス", "bow": "蝶結び",
}
ALIASES = {"maid_headdress": "maid", "hair_bow": "ribbon", "sailor_uniform": "school_uniform"}
BLUE, TEAL, PAPER, INK = "#395ccb", "#128174", "#f4f7fc", "#152339"


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                         encoding="utf-8")
    temporary.replace(path)


def plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    choices = ["Meiryo", "Yu Gothic", "Noto Sans CJK JP", "Source Han Sans JP"]
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [name for name in choices if name in fonts] + ["DejaVu Sans"],
        "figure.facecolor": PAPER, "axes.facecolor": PAPER, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return plt


def open_rgb(path):
    from PIL import Image, ImageOps
    with Image.open(path) as source:
        rgba = ImageOps.exif_transpose(source).convert("RGBA")
    white = Image.new("RGBA", rgba.size, "white")
    return Image.alpha_composite(white, rgba).convert("RGB")
