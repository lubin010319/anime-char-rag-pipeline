"""Fetch a reviewed title list through the Wikimedia Commons Action API."""
import argparse
import html
import json
import re
import time
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from common import ASSETS, DATA, JA, RAW, read_rows, write_rows

API = "https://commons.wikimedia.org/w/api.php"
LICENSES = {"CC BY-SA 3.0", "CC0"}
MAX_BYTES = 6 * 1024 * 1024


def field(info, key):
    raw = info.get("extmetadata", {}).get(key, {}).get("value", "")
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", raw)).split())


def request(session, url, delay, **kwargs):
    time.sleep(delay)
    response = session.get(url, timeout=(10, 35), **kwargs)
    response.raise_for_status()
    return response


def image_info(session, titles, language, delay):
    with request(session, API, delay, params={
        "action": "query", "format": "json", "formatversion": 2,
        "prop": "imageinfo", "titles": "|".join(titles), "redirects": 1,
        "iiprop": "url|size|mime|extmetadata", "iiurlwidth": 500,
        "iiextmetadatalanguage": language,
        "iiextmetadatafilter": "Artist|LicenseShortName|LicenseUrl|ImageDescription|Categories",
    }) as response:
        payload = response.json()
    if "error" in payload:
        raise RuntimeError(payload["error"])
    (RAW / f"api_{language}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    query = payload["query"]
    pages = {page["title"]: page for page in query["pages"]}
    for item in reversed(query.get("normalized", []) + query.get("redirects", [])):
        if item["to"] in pages:
            pages[item["from"]] = pages[item["to"]]
    return pages


def download(session, url, path, delay):
    uri = urlparse(url)
    if uri.scheme != "https" or uri.hostname not in {"upload.wikimedia.org", "thumb.wikimedia.org"}:
        raise ValueError(f"Unexpected asset host: {uri.hostname}")
    temporary = path.with_suffix(".part")
    try:
        with request(session, url, delay, stream=True) as response, temporary.open("wb") as output:
            if not response.headers.get("Content-Type", "").startswith("image/"):
                raise ValueError("Expected an image response")
            size = 0
            for chunk in response.iter_content(65536):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise ValueError("File exceeds demo download limit")
                output.write(chunk)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=17, choices=range(1, 18), metavar="1..17")
    parser.add_argument("--delay", type=float, default=1)
    args = parser.parse_args()
    if args.delay < 0.5:
        parser.error("Use a delay of at least 0.5 seconds")
    manifest = json.loads((DATA / "sources.json").read_text(encoding="utf-8"))[:args.limit]
    RAW.mkdir(parents=True, exist_ok=True)
    state = RAW / "records.jsonl"
    cached = {row["title"]: row for row in read_rows(state)} if state.exists() else {}
    pending = [row["title"] for row in manifest
               if row["title"] not in cached or not (RAW / cached[row["title"]]["file"]).is_file()]
    with requests.Session() as session:
        session.headers["User-Agent"] = (
            "AnimeCharPortfolio/1.0 (https://github.com/lubin010319/anime-char-rag-pipeline)"
        )
        retries = Retry(total=3, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504))
        session.mount("https://", HTTPAdapter(max_retries=retries))
        english = image_info(session, pending, "en", args.delay) if pending else {}
        japanese = image_info(session, pending, "ja", args.delay) if pending else {}
        for source in manifest:
            title = source["title"]
            if title not in pending:
                continue
            page = english.get(title, {})
            if not page.get("imageinfo"):
                print(f"Skipped missing source: {title}")
                continue
            info = page["imageinfo"][0]
            license_name, artist = field(info, "LicenseShortName"), field(info, "Artist")
            if not artist or license_name not in LICENSES:
                print(f"Skipped missing author or unsupported license: {title} ({license_name})")
                continue
            ja_info = japanese.get(title, {}).get("imageinfo", [{}])[0]
            url = info["url"] if info["size"] <= 1400000 else info["thumburl"]
            filename = f'{page["pageid"]}.img'
            try:
                download(session, url, RAW / filename, args.delay)
            except (requests.RequestException, ValueError) as error:
                print(f"Download failed, resume on next run: {title}: {error}")
                continue
            cached[title] = {
                **source, "id": str(page["pageid"]), "file": filename,
                "source_url": info["descriptionurl"], "image_url": url,
                "license": license_name, "license_url": field(info, "LicenseUrl"),
                "artist": artist, "original_size": [info["width"], info["height"]],
                "tags_en": [tag.replace("_", " ") for tag in source["reviewed_tags"]],
                "tags_ja": [JA[tag] for tag in source["reviewed_tags"]],
                "commons_categories": field(info, "Categories").split("|"),
                "commons_description_en_requested": field(info, "ImageDescription"),
                "commons_description_ja_requested": field(ja_info, "ImageDescription"),
            }
            write_rows(state, cached.values())
    active = [cached[row["title"]] for row in manifest
              if row["title"] in cached and (RAW / cached[row["title"]]["file"]).is_file()]
    if not active:
        raise SystemExit("No eligible Commons images were downloaded.")
    write_rows(RAW / "active.jsonl", active)
    lines = ["# Wikimedia Commons 图像署名", "",
             "图中的源图经过缩放、加白底或拼图，清洗实验另生成了降质副本。",
             "各源图及其改编沿用相应许可证。CC BY-SA 3.0 图像的改编按同一许可证分享。", ""]
    for row in active:
        artist = row["artist"].replace("[", "［").replace("]", "］")
        lines.append(f'- [{row["title"].removeprefix("File:")}]({row["source_url"]}) · '
                     f'{artist} · [{row["license"]}]({row["license_url"]})')
    (ASSETS / "ATTRIBUTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(active)} licensed source images")


if __name__ == "__main__":
    main()
