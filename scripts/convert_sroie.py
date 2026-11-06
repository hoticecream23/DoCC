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
  python scripts/convert_sroie.py --parquet datasets/sroie/test.parquet --split test [--dry-run]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import IO, Any

import pyarrow.parquet as pq

from baseline.config import load_config
from baseline.io import atomic_open, report_dry_run
from baseline.logging_setup import setup_logging
from baseline.metadata.normalizers import NormCtx, normalize
from baseline.schema import compute_doc_id

# Rows per Parquet batch, so a split is never held in memory twice.
PARQUET_BATCH_ROWS = 64

# SROIE field -> (our field, our normalizer). SROIE's fourth field, address,
# has no field in our schema, so it is not scored.
FIELDS = {
    "company": ("vendor_name", "text_norm"),
    "date": ("invoice_date", "date_iso"),
    "total": ("total_amount", "currency"),
}


def gold_fields(entities: dict[str, str | None], norm_opts: dict[str, Any]) -> tuple[list[dict[str, str]], list[str]]:
    """Map SROIE entities to gold metadata rows.

    Args:
        entities: SROIE key fields for one receipt.
        norm_opts: Normalizer options from fields.yaml.

    Returns:
        Gold rows, and the fields skipped because they would not normalize.
    """
    rows: list[dict[str, str]] = []
    skipped: list[str] = []
    for sroie_name, (field, normalizer) in FIELDS.items():
        raw = (entities.get(sroie_name) or "").strip()
        if not raw:
            continue
        ctx = NormCtx(full=raw, start=0, end=len(raw), opts=norm_opts)
        normalized, _ = normalize(normalizer, raw, ctx)
        if normalized is None:
            # A date our normalizer cannot parse cannot be compared fairly.
            skipped.append(f"{field}={raw!r}")
            continue
        rows.append({"field": field, "normalized_value": normalized})
    return rows, skipped


def write_gold_row(fh: IO[str], path: Path, fields: list[dict[str, str]]) -> None:
    """Write one gold line, keyed on the input file's doc_id.

    Args:
        fh: The open gold file.
        path: The input file the gold describes.
        fields: Gold metadata rows.
    """
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
    """Write the images, text files and gold files for one split.

    Args:
        parquet: The SROIE Parquet file.
        split: Split name, used as the output sub directory.
        out: Output root.
    """
    norm_opts = load_config().fields.get("normalizer_options", {}) or {}
    images_dir = out / split / "images"
    text_dir = out / split / "text"
    images_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(parquet, columns=["key", "image", "entities", "words"])
    n_docs = 0
    all_skipped: list[str] = []
    with (
        atomic_open(out / split / "gold_images.jsonl") as gold_images,
        atomic_open(out / split / "gold_text.jsonl") as gold_text,
    ):
        for batch in table.to_batches(max_chunksize=PARQUET_BATCH_ROWS):
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
    """Parse arguments and convert, or report what would be written."""
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--parquet", required=True, type=Path)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", default=Path("datasets/sroie"), type=Path)
    ap.add_argument("--dry-run", action="store_true", help="log what would be written and stop")
    args = ap.parse_args()
    setup_logging()
    if args.dry_run:
        split_dir = args.out / args.split
        outputs = [
            split_dir / "images",
            split_dir / "text",
            split_dir / "gold_images.jsonl",
            split_dir / "gold_text.jsonl",
        ]
        report_dry_run(outputs, pq.read_metadata(args.parquet).num_rows)
        return
    convert(args.parquet, args.split, args.out)


if __name__ == "__main__":
    main()
