# Pack mode: how a cabinet run decides class and fields

Pack mode runs DoCC with one client cabinet's own class list and index fields
(see `client-pack.md`, "Running a cabinet"). Code: `baseline/packmode.py`.
This page states the rules the run follows.

## How the class is decided

The trained classifier does not know the pack's classes. Pack mode uses a
keyword prior only.

1. Each class has terms: its name and its aliases. A class named "other", in
   any case, is the abstain label and is never a class.
2. Both the terms and the text are NFKC-normalised and case-folded. A term
   matches as whole words, in order, with any punctuation or spacing between
   them. Word characters are Unicode word characters, so Arabic or other
   non-Latin class names match.
3. Containment rule: if a term of one class lies inside a longer matched term
   of another class at the same place, the shorter match is discarded. For
   example, "Payment Voucher Transfer" in the text counts only for the class
   "Payment Voucher Transfer", not for "Payment Voucher".
4. Tiers: a term in the header (the first `classification.header_chars`
   characters of page 1) scores 0.8. On page 1 it scores 0.7. Anywhere else
   it scores 0.6. A class takes its best tier.
5. Margin: the best class must score at least 0.6 and lead the runner-up by
   at least 0.1. Otherwise the label is "other". The comparison allows a
   float tolerance (`PRIOR_EPSILON`), so 0.7 against 0.6 passes.

All tiers are below the cascade's `rule_short_circuit` (0.9), so a model
answer always overrides the prior. The cascade records the matched terms per
class in provenance `prior`.

## How fields are anchored

A field is anchored only on the client's own field name and its spelling
variants in `SYNONYMS`. The variant groups are small and explicit:

| Group | Words |
|---|---|
| number | no, no., number, num, # |
| invoice | inv, invoice |
| amount | amt, amount |
| reference | ref, reference |
| quantity | qty, quantity |

No group crosses document concepts. PO, DO, LPO, GRN, Ref, Voucher, Claim,
Policy and Cheque stay distinct. A field never inherits another extractor's
labels. It can borrow only a value pattern and a normaliser. If the label is
not in the text, the rule gives no value.

A label must not be glued to another word ("LPO No" does not match "PO No").
A one-word label, such as "Date" or "Total", must also not follow another word
on its line. So "Due Date", "Order Date" and "Posting Date" do not match a
"Date" field.

## Identifier or amount

A field is an identifier when its role is `key`, or when its name holds one of
these words: no, number, num, #, id, code, ref, reference. An identifier uses
the text normaliser: trim and collapse spaces, keep leading zeros and
separators. "00123456" stays "00123456" and "1,234" stays "1,234".

Other fields take their value shape from their dtype:

| dtype | Value pattern and normaliser from |
|---|---|
| date | `invoice_date` (date pattern, `date_iso`) |
| number | `total_amount` (amount pattern, `currency`) |
| text | rest of the label's line, `text_norm` |
| selection | no rule; the cascade asks the model |

"Vendor Name", "Supplier Name" and "Customer Name" borrow the value pattern of
`vendor_name` or `buyer_name`, but not their labels.

## Multi-frame images and the page limit

Two `extraction` settings in `pipeline.yaml`. Both are validated when the
config loads. An unknown or mistyped key under `extraction` or `client_pack`
fails the load with a `ConfigError` that names the key.

| Key | Type | Normal mode default | Pack mode default |
|---|---|---|---|
| `extraction.multi_frame_images` | bool | false: frame 0 only | true: every frame |
| `extraction.max_pages` | int or null | null: no limit | 500 |

The keys are left unset in `config/pipeline.yaml`, so each mode keeps its own
default. A value set in the file wins in both modes. Past `max_pages` the file
is still one record, and its diagnostics say how many pages were read. The
cascade renderer follows `multi_frame_images` too, so the model sees the same
frames the text came from.

A relative `client_pack.dir` resolves against the config directory, not the
working directory. A `--client-pack` path on the command line resolves against
the working directory.

## Command line

- `--client-pack DIR --cabinet NAME` runs that cabinet.
- `--cabinet NAME` alone, when `pipeline.yaml` sets a pack, picks that cabinet
  of the configured pack.
- `--cabinet NAME` with no pack anywhere stamps `cabinet_id`, as in a normal run.
- `cascade run --dry-run` in pack mode counts the manifest documents found in
  the input directory, not every file there.

## Resume

A pack run resumes by file stem, which is the manifest `doc_id`. Two manifest
documents with the same bytes therefore each get a record. A restart after a
crash writes each missing document once. Resuming into an output file that
holds another cabinet's records is refused; use a new output file or
`--no-resume`.

## Text mode budgets (`vlm.input: text`)

| Key | Default | Meaning |
|---|---|---|
| `vlm.text_page_chars` | 4000 | Characters sent at most from one page |
| `vlm.text_total_chars` | 24000 | Characters sent at most in all |

Every page sends its head, up to `text_page_chars`, while the total lasts.
When the total runs out, the first and last pages are filled first (keys and
totals live there) and pages are dropped from the middle. Each page is labelled
with its real page number.

Provenance records what was sent:

- `pages_included`: the 1-based pages whose text was sent;
- `pages_sent`: how many;
- `text_truncated`: true when any page was cut or left out;
- `text_chars_sent`: characters of page text sent;
- `reduced_text_chars`: set only after a retry (below).

When whole pages were left out, the record's diagnostics say "text mode sent N
of M pages".

On a context-length failure (HTTP 400) the request is retried once with half
of `text_total_chars`. A second failure falls back to the baseline record.

## Grounding in text mode

In text mode a model value is accepted only if it grounds in the text that was
actually sent. Grounding runs on a copy of the document in which every
character not sent is blanked, with offsets unchanged. A value printed only on
a page that was cut or left out is therefore rejected as ungrounded. Proven
(checksummed) baseline values are kept as in every run.
