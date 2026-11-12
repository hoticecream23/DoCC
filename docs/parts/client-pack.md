# Client packs

A client pack holds everything a run on one client share needs: the classes
per cabinet, the index fields per document type, and a labeled list of the
sample documents. It is built on the client machine from the two CSVs that
`scripts/inventory_share.py` writes (`inventory.csv` and `schema.csv`).

## Privacy

Packs, inventories, schema and header CSVs, materialized copies and results
stay on the client machine. Never commit them. Every one of them carries
client file names, folder names or document values. `.gitignore` covers
`client_pack*/`, `inventory.csv`, `schema.csv` and `headers.csv`; keep a pack
outside the repo in any case. Examples in code, tests and docs use made-up
names only (Acme, Borealis, Corvid, Delta).

## Format

The Pydantic models in `baseline/clientpack.py` define the format.

```
<pack>/pack.yaml                       version, created_at, share_root, cabinets: [slug, name, documents, types_with_samples]
<pack>/label_map.yaml                  per cabinet: {label -> schema type name}
<pack>/REPORT.md                       exclusions, ambiguous and unmapped labels, duplicates, oversize files, odd field types
<pack>/<slug>/taxonomy.yaml            classes: [{name, aliases, has_samples}]
<pack>/<slug>/fields.yaml              {type: [{name, dtype, date_format, role, origin}]}
<pack>/<slug>/manifest.csv             doc_id, rel_path, in_zip, case_id, label, label_source, key_gold, pages, size_kb
<pack>/<slug>/excluded.csv             rel_path, reason
```

- `doc_id` is the first 16 hex characters of SHA-256 over the cabinet name and
  the path. It does not change when the file bytes change.
- `rel_path` is relative to the share root. For a zip member it is the path the
  inventory gives: `<zip folder>/<zip stem>/<inner dirs>/<name>`.
- `label` is the schema type name, or empty. `label_source` is `folder`,
  `filename` or `none`.
- `taxonomy.yaml` lists the types that have samples plus an `other` class.
  `--all-types` lists every schema type of the cabinet. `aliases` are the
  folder labels that map to the type.
- `fields.yaml` covers the types in `taxonomy.yaml`. `dtype` is
  `text|date|number|selection`; `role` is `key|required|optional`; `origin` is
  `document|erp|manual|fixed`.
- `excluded.csv` reasons: `reference`, `duplicate`, `archive`, `unsupported`,
  `oversize`.

## Build rules

1. **Screening.** Inventory kind `archive` and `reference` are excluded with
   that reason, and kind `other` as `unsupported`. An image (`.png`, `.jpg`
   and so on) is a reference when its folder also holds `.pdf` or `.tif`
   samples, when its stem equals a folder label or type name, or when it
   contains "index field". Documents over `max_size_mb` (default 50) are
   `oversize`.
2. **Duplicates.** Documents are deduplicated across the whole share by
   (name, size_kb). The copy kept is a loose file before a zip member, then
   the shallowest path, then the first path in order. So a cabinet copied
   inside another, and a zip copied inside a sibling zip, both lose to the
   original.
3. **Labels.** A nested folder with the cabinet's own name ("Acme/Acme/...")
   is skipped. The first folder below it is the label. A file with no folder
   uses its stem as the label (`label_source: filename`). Each label resolves
   in this order: an entry in `label_map.yaml`; an exact match of the
   normalised names (case and punctuation ignored); a type name that starts
   with the label ("LGRN" -> "LGRN (Local GRN)"); a type whose words include
   all the label's words. A step that matches several types stops there: the
   label stays unmapped and REPORT.md lists the candidates.
4. **label_map.yaml.** Every label is written with its resolved type, or `''`.
   A re-run reads the file first and keeps every non-empty entry as an
   override. To fix a label, set its type. To have a label matched again, set
   it to `''`. An override that names a type outside the schema is kept and
   listed in REPORT.md.
5. **Case bundles.** A type folder with no documents of its own, whose child
   folders each hold two or more documents, holds cases. `case_id` is the
   child folder name. The type labels the case, not each file.
6. **key_gold.** Only for documents outside a case whose type has a key field:
   the file stem, with a leading `<digits>_` removed, kept only if it holds a
   digit. "6390_C111322.pdf" gives "C111322"; "Purchase Daily Invoice.tif"
   gives nothing.
7. **Fields.** `dtype` comes from the free-text data type: "date" first, then
   "text"/"alpha" (so "Text (Alpha Numeric)" is text), then selection words,
   then number words. A format in brackets, such as `(DD/MM/YYYY)`, becomes
   `date_format`. `role`: Key -> key; Required, Mandatory and Fixed ->
   required; Optional and Not Required -> optional; anything else -> optional,
   listed in REPORT.md. `origin`: Fixed -> fixed; indexing method Autofil(DB),
   or a `db_field` on a non-key row -> erp; a name with an operational word
   (box, pouch, file number, indexing person, year, month, verification,
   remarks, posted by, driver, bundle) -> manual; else document.

## Commands

```
baseline client-pack build inventory.csv schema.csv --out DIR [--all-types] [--share ROOT] [--dry-run]
baseline client-pack materialize --pack DIR --cabinet NAME --share ROOT --work DIR [--dry-run]
baseline client-eval --pack DIR --cabinet NAME --results FILE [--out report.md]
```

- `build` writes or refreshes the pack. Re-running it gives the same files,
  apart from `created_at` in pack.yaml, and keeps label_map.yaml edits.
- `materialize` copies, or extracts from its zip, each manifest document to
  `<work>/<slug>/<doc_id><ext>`. It skips files already there and refuses a
  work directory inside the share. Run the pipeline on that directory.
- `client-eval` scores a results file from that run. `--cabinet` takes the
  name or the slug.

## Scoring and its limits

Results join to the manifest by file stem, which is the `doc_id`.

- **Classification.** Accuracy, plus precision and recall per class, over
  labeled documents that are not in a case. An abstention (no label, or
  `unknown`) counts as wrong. Documents with an empty label are counted as
  unlabeled and not scored.
- **Key field.** Exact: the predicted value of the type's key field equals
  `key_gold` after case, spaces and punctuation are removed. Field names are
  matched the same way, so `pv_no` matches "PV No". Contained: `key_gold`
  appears inside any extracted value. This looser rate shows values the
  pipeline found but filed under another field.
- **Case bundles.** Scored per case, both ways. Majority: the most common
  non-empty prediction among the case's documents is the case type. Any-hit:
  at least one document is.
- **Counts.** Missing (in the manifest, not in the results), failed (the
  record has diagnostics errors) and abstained.

No field other than the key is scored. The pack has no gold for any other
field, and the report says so. Folder labels are the client's filing, not a
review, so a misfiled sample is scored as a pipeline error.
