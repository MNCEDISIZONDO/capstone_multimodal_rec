"""Inspect raw metadata structure before building the coarse pass.
Reads a bounded sample — never loads the whole file."""
import gzip, json
from collections import Counter
from pathlib import Path

from config import load_config

cfg = load_config()
path = Path(cfg["paths"]["raw_data"]) / cfg["sources"]["metadata_file"]

SAMPLE = 50_000


def image_urls(imgs):
    """Return usable image URLs, tolerating either documented shape."""
    out = []
    if isinstance(imgs, dict):                     # {"large": [...], "hi_res": [...]}
        for key in ("large", "hi_res", "thumb"):
            v = imgs.get(key)
            if isinstance(v, list):
                out += [u for u in v if u]
            elif isinstance(v, str) and v:
                out.append(v)
    elif isinstance(imgs, list):                   # [{"large": "...", ...}, ...]
        for im in imgs:
            if isinstance(im, dict):
                for key in ("large", "hi_res", "thumb"):
                    u = im.get(key)
                    if isinstance(u, str) and u:
                        out.append(u)
            elif isinstance(im, str) and im:
                out.append(im)
    return out


main_cats, cat_tokens = Counter(), Counter()
has_title = has_desc = has_feat = has_img = 0
first = None
first_with_img = None

with gzip.open(path, "rt", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= SAMPLE:
            break
        rec = json.loads(line)
        if first is None:
            first = rec
        main_cats[rec.get("main_category")] += 1
        for c in (rec.get("categories") or []):
            cat_tokens[str(c).lower()] += 1
        if rec.get("title"):
            has_title += 1
        if rec.get("description"):
            has_desc += 1
        if rec.get("features"):
            has_feat += 1
        if image_urls(rec.get("images")):
            has_img += 1
            if first_with_img is None:
                first_with_img = rec

n = min(SAMPLE, i + 1)
print(f"=== sampled {n:,} records ===\n")
print("FIELDS PRESENT:", sorted(first.keys()), "\n")

print("RAW images TYPE:", type(first.get("images")).__name__)
print("RAW images SAMPLE:")
print(json.dumps((first_with_img or first).get("images"), indent=2)[:700], "\n")

print("RAW categories SAMPLE:", json.dumps(first.get("categories"))[:300])
print("RAW main_category  :", repr(first.get("main_category")), "\n")

print("CONTENT AVAILABILITY (share of sample):")
for label, c in [("title", has_title), ("description", has_desc),
                 ("features", has_feat), ("image url", has_img)]:
    print(f"  {label:<12} {c/n:6.1%}")

print("\nTOP 25 main_category:")
for v, c in main_cats.most_common(25):
    print(f"  {c:>7,}  {v}")

print("\nTOP 40 category tokens:")
for v, c in cat_tokens.most_common(40):
    print(f"  {c:>7,}  {v}")