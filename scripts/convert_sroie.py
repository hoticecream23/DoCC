"""Turn the SROIE receipts dataset into inputs and gold the harness can score.

SROIE (ICDAR 2019, CC-BY-4.0) ships as Parquet: a receipt image, its text
lines as transcribed by the organisers, and four key fields. For each split
this writes two inputs, each with its own gold file:

  <out>/<split>/images/<key>.jpg   the pipeline runs its own OCR on these
  <out>/<split>/text/<key>.txt     the organisers' lines, so OCR errors are
                                   taken out and only extraction is scored
  <out>/<split>/gold_images.jsonl
  <out>/<split>/gold_text.jsonl

Gold keys on doc_id, which is a hash of the file, so each input needs its own
gold file. Field values go through the baseline's own normalizers, so a date
or amount is compared in the same form the pipeline emits it.

Usage:
  python scripts/convert_sroie.py --parquet datasets/sroie/test.parquet --split test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from baseline.config import load_config
from baseline.metadata.normalizers import NormCtx, normalize
from baseline.schema import compute_doc_id

# SROIE field -> (our field, our normalizer). SROIE's fourth field, address,
# has no field in our schema, so it is not scored.
FIELDS = {
    "company": ("vendor_name", "text_norm"),
    "date": ("invoice_date", "date_iso"),
    "total": ("total_amount", "currency"),
}


def gold_fields(entities: dict, norm_opts: dict) -> tuple[list[dict], list[str]]:
    """Map SROIE entities to gold metadata rows. Returns rows and skipped fields."""
    rows, skipped = [], []
    for sroie_name, (field, normalizer) in FIELDS.items():
        value = (entities.get(sroie_name) or "").strip()
        if not value:
            continue
        ctx = NormCtx(full=value, start=0, end=len(value), opts=norm_opts)
        normalized, _ = normalize(normalizer, value, ctx)
        if normalized is None:
            # A date our normalizer cannot parse cannot be compared fairly.
            skipped.append(f"{field}={value!r}")
            continue
        rows.append({"field": field, "normalized_value": normalized})
    return rows, skipped


def write_gold_row(fh, path: Path, fields: list[dict]) -> None:
    row = {
        "doc_id": compute_doc_id(path),
        "filename": path.name,
        "path": path.as_posix(),
        "label": "receipt",
        "metadata": fields,
        "tags": [],
    }
    fh.write(json.dumps(row, sort_keys=True) + "\n")


def convert(parquet: Path, split: str, out: Path) -> None:
    norm_opts = load_config().fields.get("normalizer_options", {}) or {}
    images_dir = out / split / "images"
    text_dir = out / split / "text"
    images_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(parquet, columns=["key", "image", "entities", "words"])
    n_docs, all_skipped = 0, []
    with (
        open(out / split / "gold_images.jsonl", "w", encoding="utf-8") as gold_images,
        open(out / split / "gold_text.jsonl", "w", encoding="utf-8") as gold_text,
    ):
        for batch in table.to_batches(max_chunksize=64):
            for row in batch.to_pylist():
                key = row["key"]
                fields, skipped = gold_fields(row["entities"], norm_opts)
                all_skipped += [f"{key}: {s}" for s in skipped]

                image_path = images_dir / f"{key}.jpg"
                image_path.write_bytes(row["image"]["bytes"])
                write_gold_row(gold_images, image_path, fields)

                text_path = text_dir / f"{key}.txt"
                text_path.write_text("\n".join(row["words"]) + "\n", encoding="utf-8")
                write_gold_row(gold_text, text_path, fields)
                n_docs += 1

    print(f"{split}: {n_docs} documents written to {out / split}")
    print(f"{len(all_skipped)} gold values skipped because they would not normalize")
    for s in all_skipped:
        print("  ", s)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parquet", required=True, type=Path)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", default=Path("datasets/sroie"), type=Path)
    args = ap.parse_args()
    convert(args.parquet, args.split, args.out)


if __name__ == "__main__":
    main()
