# Public datasets

Checked 2026-09-17 for use in company work, which is commercial use. Not legal
advice: confirm with legal before training anything that ships to a client.

| Dataset | Licence | Usable |
| --- | --- | --- |
| SROIE, ICDAR 2019 receipts | CC-BY-4.0, per the Hugging Face mirror `jsdnrs/ICDAR2019-SROIE`, which attributes the official challenge | Yes, with attribution |
| CORD, receipts with line items | CC-BY-4.0, per `github.com/clovaai/cord` | Yes, with attribution |
| DocILE, business documents | Non-commercial research only, no third party access | No |
| FUNSD, forms | "solely for non-commercial, research and educational purposes" | No |
| RVL-CDIP, 16 document classes | "other", defers to the UCSF Industry Documents copyright page | Unclear, treat as no |

SROIE and CORD are both receipts, so they exercise field extraction and line
items. Neither says anything about classification or invoice layouts.

## SROIE

Downloaded to `datasets/sroie/` (gitignored, about 510 MB). Reading it needs
`pyarrow`, which is a tool dependency, not a pipeline one.

```bash
python scripts/convert_sroie.py --parquet datasets/sroie/test.parquet --split test
python -m baseline run --input datasets/sroie/test/images --output datasets/sroie/test/pred_images.jsonl --workers 4 --no-resume
python -m baseline eval --gold datasets/sroie/test/gold_images.jsonl --pred datasets/sroie/test/pred_images.jsonl
```

The same commands with `text` in place of `images` score the organisers' own
text lines, which removes OCR from the measurement.

Mapping: `company` to `vendor_name`, `date` to `invoice_date`, `total` to
`total_amount`, class `receipt`. `address` has no field in our schema and is
not scored. 15 of 361 gold dates are skipped because the baseline's date
parser cannot read them (`27/MAR/2018`, `22 MAR 18`, `2018/03/27`,
`23.03.18`), which is a gap in the baseline worth fixing separately.

### Baseline on the 361 test receipts

| Input | Field | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| organisers' text | `vendor_name` | 0.000 | 0.000 | 0.000 |
| organisers' text | `invoice_date` | 0.992 | 0.749 | 0.853 |
| organisers' text | `total_amount` | 0.490 | 0.479 | 0.485 |
| images, Tesseract | `vendor_name` | 0.000 | 0.000 | 0.000 |
| images, Tesseract | `invoice_date` | 0.829 | 0.506 | 0.628 |
| images, Tesseract | `total_amount` | 0.339 | 0.307 | 0.323 |

What the numbers say:

- **`vendor_name` is never found.** Its anchors are labels like "sold by" and
  "supplier name". A receipt prints no label: the vendor is the first line.
- **`total_amount` is wrong half the time even on clean text.** A receipt
  prints several totals (subtotal, rounding, total incl. GST, cash, change) and
  the anchor picks the wrong one. This is the anchor field weakness
  PROJECT_STATE already records, measured at scale.
- **OCR costs about a third of the date and total scores.** Date F1 falls from
  0.853 to 0.628, total from 0.485 to 0.323.
- **Classification is not meaningful here.** 310 of 361 receipts print
  "TAX INVOICE" (Malaysian GST receipts) and are labelled `invoice`. The gold
  label is `receipt`, but the document's own title is a fair argument for
  either, so the classification score is left out.

Cost on this laptop: median 1.3 s per receipt image end to end, 95th
percentile 2.8 s, almost all of it Tesseract. On text input, median 51 ms.
