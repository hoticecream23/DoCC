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
<pack>/pack.yaml                       version, created_at, share_root, cabinets: [slug, name, documents, types_with_samples],
                                       retired: {name -> slug} of cabinets gone from the inventory
<pack>/label_map.yaml                  per cabinet: {overrides: {label -> type}, auto: {label -> type}}
<pack>/field_overrides.yaml            optional, hand-written: {cabinet: {type: {field: {origin, role}}}}
<pack>/REPORT.md                       exclusions, ambiguous and unmapped labels, dropped aliases, skipped wrappers,
                                       duplicates dropped, possible duplicates, oversize files, odd field types,
                                       field definitions that disagree, unknown field_overrides.yaml entries,
                                       merged and renamed schema types, workbooks under no cabinet
<pack>/<slug>/taxonomy.yaml            classes: [{name, aliases, has_samples}]
<pack>/<slug>/fields.yaml              {type: [{name, dtype, date_format, role, origin}]}
<pack>/<slug>/manifest.csv             doc_id, rel_path, in_zip, case_id, label, label_source, key_gold, pages, size_kb, member,
                                       role, dup_group
<pack>/<slug>/excluded.csv             rel_path, reason, detail
```

- The slug is the cabinet name in lower case, with every other character as
  `_`. A slug that pack.yaml already records (under `cabinets` or `retired`)
  is kept for that name forever, and no other name takes it, so a pack or
  work folder never changes owner. New names are taken in name order; when a
  slug is taken ("Acme-Insurance" built first, "Acme Insurance" added later),
  the new name gets `_2`, `_3` and so on. Commands take the name or the slug;
  an exact name wins.
- `doc_id` is the first 16 hex characters of SHA-256 over the cabinet name and
  the path. It does not change when the file bytes change. When one zip holds
  two entries with the same name, the later entry hashes `<path>#<member>`.
- `rel_path` is relative to the share root. For a zip member it is the path the
  inventory gives: `<zip folder>/<zip stem>/<inner dirs>/<name>`. A member name
  that starts with `/` or holds `..` is made a safe relative path first.
- `member` is a zip member's index in its archive, so `materialize` can tell
  same-named entries apart. It is empty for a loose file.
- `label` is the schema type name, or empty. `label_source` is `folder`,
  `filename` or `none`.
- `role` is `document` or `attachment` (see Case bundles). `dup_group` is a
  short id shared by kept documents with the same name and size in different
  folders, or empty. A manifest written before these columns existed loads
  with `document` and empty.
- `detail` in excluded.csv, for a duplicate, says whether the dropped file is
  loose or a zip member (and of which zip), and names the copy kept the same
  way.
- `taxonomy.yaml` lists the types that have samples plus an `other` class.
  `--all-types` lists every schema type of the cabinet. `aliases` are the
  folder labels that map to the type. An alias equal to another class's name
  is dropped and listed in REPORT.md.
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
2. **Schema.** A workbook belongs to the cabinet named by its nearest folder
   that is a cabinet name, so "Acme/Borealis/types.xlsx" (a copied cabinet)
   gives Borealis its types. A workbook under no cabinet name is skipped and
   listed in REPORT.md. Type names that differ only in case, spacing or
   punctuation ("PAYMENT VOUCHER", "Payment Voucher ") are one type: the first
   spelling is its name, and the fields are joined by normalised field name.
   When two definitions of a field disagree, the role is the stronger one
   (key, then required, then optional), a filled data type beats a blank one,
   and a date format beats none; otherwise the first wins. REPORT.md lists
   each merge and each disagreement (type, field, both values). A type named
   "other" in any case is renamed `Other (client)`, because `other` is the
   reserved class; REPORT.md lists it.
3. **Labels.** A nested folder with the cabinet's own name ("Acme/Acme/...")
   is skipped. Wrapper folders are skipped next. A wrapper holds subfolders
   and no documents of its own, and its name matches no schema type and no
   override. With a schema, it is a wrapper only when skipping it yields a
   child folder that matches a type (directly, or through another wrapper),
   whatever its name; so a "Samples" folder of case subfolders is a label.
   With no schema, only the common wrapper names (samples, sample, documents,
   docs, exporteddocuments) are skipped. REPORT.md lists every skipped
   wrapper. The first folder after that is the label, so
   "Acme/Acme/Samples/Purchase Invoice/100/100.pdf" has the label "Purchase
   Invoice". A file with no folder uses its stem, without copy suffixes, as
   the label (`label_source: filename`). Each label resolves in this order: an
   override in `label_map.yaml`; an exact match of the normalised names (case
   and punctuation ignored); a type name that starts with the label ("LGRN" ->
   "LGRN (Local GRN)"); a type whose words include all the label's words. A
   step that matches several types stops there: the label stays unmapped and
   REPORT.md lists the candidates. A single prefix or word match is used only
   when the label holds at least 75% of the type's words, or all the type's
   words outside brackets and before a dash ("LGRN (Local GRN)", "Sales
   Invoice - Yellow"). 75% is above two of three words on purpose, so
   "Payment Voucher" does not become "Payment Voucher Transfer"; that label
   stays unmapped and REPORT.md names the candidate.
4. **Case bundles.** Decided before duplicates are dropped, for each child
   folder of a label folder that has no documents of its own. A child folder
   is a single-document folder when it holds one document, or when exactly
   one document's stem (without copy markers) equals the folder name: that
   one is the document, with `key_gold`, and the others are attachments
   (`role: attachment`, no `key_gold`, not scored). A child folder with two or
   more documents and no stem equal to its name (or several) is a case;
   `case_id` is the child folder name and the type labels the case, not each
   file. One type folder may mix both: "Inv/3/3.pdf" plus "Inv/3/3-annex.pdf"
   is one document and an attachment, and "Inv/C-1/a.pdf" plus
   "Inv/C-1/b.pdf" beside it is a case.
5. **Duplicates.** Within one cabinet, two documents are copies only when
   their path below the cabinet (after the nested folder with the cabinet's
   own name is skipped) and their size_kb are equal, or when one is a zip
   member and the other is the same member reached through a copy of that zip
   inside another zip ("Inv/new/old/x.pdf" in new.zip copies "Inv/old/x.pdf"
   in old.zip). So "Inv/c1/Invoice.pdf" and "Inv/c2/Invoice.pdf" are both
   kept; kept documents with the same name and size in different folders
   share a `dup_group` and REPORT.md lists them under "Possible duplicates
   (same name and size, different folder)". When inventory rows carry a
   `sha256` column, that content hash decides the groups instead. Two entries of one zip with the same name are both kept. Across
   cabinets, a document whose first folder is another cabinet's name
   ("Acme/Borealis/...") is a copy when that cabinet holds the same path below
   its own name with the same size_kb. The copy kept is the one in the cabinet
   that owns it, then a loose file before a zip member, then the shallowest
   path, then the first path in order.
6. **label_map.yaml.** Each cabinet has two sections. `overrides:` is
   hand-written: an entry there always wins and is kept as written. `auto:`
   is rewritten on every build with what matching gave; it is never read. To
   fix a label, add it under `overrides:`. To have it matched again, delete
   it there. An override that names a type outside the schema is kept and
   listed in REPORT.md. An old flat file (label -> type) is read once: an
   entry that differs from what matching gives now becomes an override, and
   the rest are dropped.
7. **key_gold.** Only for documents (not attachments) labelled by a folder,
   outside a case, whose type has a key field: the file stem without a
   leading "Copy of ", without a trailing copy marker, and without a leading
   `<digits>_`, kept only if it holds a digit. A copy marker is " (n)" with n
   of one or two digits after a space, or " - Copy" / " copy" with an
   optional " (n)". Brackets with no space before them, or with more digits,
   stay: "INV(2024)" and "Form 16 (2023)" are kept whole. "6390_C111322.pdf"
   gives "C111322"; "Copy of PV-14380.pdf" gives "PV-14380"; "Purchase Daily
   Invoice.tif" gives nothing; a root file "Form 16.tif" labelled by its name
   gives nothing.
8. **Fields.** `dtype` is date when the data type starts with the word
   "date" (so "Updated" is not a date). Otherwise "text"/"alpha" comes first
   (so "Text (Alpha Numeric)" is text), then selection words, then number
   words. A format in brackets becomes `date_format`, with backslashes read as
   slashes and an optional time: `(dd\MM\YYYY)` gives `DD/MM/YYYY`,
   `(DD/MM/YYYY HH:MM)` gives `DD/MM/YYYY HH:MM`. `role`: Key -> key;
   Required, Mandatory and Fixed -> required; Optional and Not Required ->
   optional; anything else -> optional, listed in REPORT.md. `origin`: Fixed
   -> fixed; indexing method Autofil(DB), or a `db_field` on a non-key row ->
   erp; manual when the whole normalised name is year, month, box, pouch,
   bundle, remark, remarks or verified, or when the name holds one of the
   phrases box and pouch no, box no, box number, pouch no, file number, file
   no, indexing person, indexed by, verification status, posted by, bundle no,
   remarks; else document. A postal box ("P.O. Box No", "Post Box No") is
   document, and so are "Model Year", "Policy Year" and "Driver Name".
9. **field_overrides.yaml.** Optional and hand-written:
   `{cabinet: {type: {field: {origin: ..., role: ...}}}}`. Type and field
   names match after normalising. Its values replace the rules' values in
   fields.yaml. Build reads it and never writes it. A value outside the
   allowed set stops the build with an error. A cabinet, type or field it
   names that the build does not have is ignored, listed in REPORT.md, and
   logged once as a warning.

## Commands

```
baseline client-pack build inventory.csv schema.csv --out DIR [--all-types] [--share ROOT] [--dry-run]
baseline client-pack materialize --pack DIR --cabinet NAME --share ROOT --work DIR [--dry-run]
baseline client-eval --pack DIR --cabinet NAME --results FILE [--out report.md]
baseline client-run --pack DIR --cabinet NAME --share ROOT --work DIR [--no-cascade] [--dry-run]
```

- `build` writes or refreshes the pack. Re-running it gives the same files,
  apart from `created_at` in pack.yaml, and keeps label_map.yaml edits.
- `materialize` copies, or extracts from its zip, each manifest document to
  `<work>/<slug>/<doc_id><ext>`. It skips files already there and refuses a
  work directory inside the share. Run the pipeline on that directory. A
  missing or unreadable source is logged and skipped, and the rest are still
  copied. In code, `materialize` returns a `MaterializeResult`: the list of
  ready paths, with `copied` and `missing` (the doc_ids that failed).
- `scripts/inventory_share.py` does not follow symbolic links; it logs and
  skips them. A zip member it cannot read (corrupt data, an unsupported
  compression) and an image too large to open safely are still listed, with
  an empty page count and a warning that names the error type only.
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
- **Case bundles.** Scored per case three ways: all files right, any file
  right, and majority right (a tie is not right). docs/parts/client-eval.md
  defines them.
- **Attachments** are counted apart and not scored.
- **Counts.** Missing (in the manifest, not in the results), failed (the
  record has diagnostics errors) and abstained.

No field other than the key is scored. The pack has no gold for any other
field, and the report says so. Folder labels are the client's filing, not a
review, so a misfiled sample is scored as a pipeline error.

## Running a cabinet

```
baseline client-run --pack DIR --cabinet NAME --share ROOT --work DIR [--no-cascade] [--dry-run]
```

The command materializes the cabinet into `<work>/<slug>/`. It then runs the
cascade in pack mode and writes `results.jsonl` and `client_eval.md` into the
same folder. The cascade runs the baseline for every document, so there is
one run. With `--no-cascade`, or when no model is set (`vlm.model` or
`DOCC_VLM_MODEL`), the baseline runs alone and the log says so. A run without
`--resume` first deletes `results.jsonl` and moves the old `client_eval.md`
to `client_eval.md.previous`. The new report is written only after scoring
succeeds. If a stage fails (materialize, run or score), `client_eval.md` is a
short stub naming the stage and the time, and the error is raised.

**Pack mode.** `--client-pack DIR --cabinet NAME` on `baseline run` and
`cascade run`, or `client_pack: {dir, cabinet}` in pipeline.yaml, turns it on.
`baseline/packmode.py` builds a config from the cabinet:

- The labels are the cabinet's classes. `other` is the abstain label.
  `cabinet_id` is the cabinet slug.
- Inputs are the manifest's `<doc_id><ext>` copies only, so results join to
  the manifest by file stem. Other files in the folder are not read.
- **Class, baseline.** The trained classifier does not know the pack's
  classes, so a keyword prior decides. A class name or alias found in the
  header scores 0.8, on page 1 scores 0.7, and anywhere else scores 0.6. The
  best class must lead the next by 0.1, else the label is `other`. The scores
  of every class are in `classification.all_scores`.
- **Class, cascade.** The model chooses from the closed list of classes plus
  `other`. All prior scores are below `rule_short_circuit` (0.9), so a valid
  model answer always overrides the prior. The provenance line records the
  prior's evidence under `prior` (the matched class names and aliases, and
  where they matched).
- **Fields.** Only `origin: document` fields are extracted, key first. Fields
  of origin erp, manual or fixed are never asked of the model. Each field is
  stored under the client's field name and scoped to its class. A name in
  `EXTRACTOR_BY_NAME` (for example "Invoice No", "Date", "Total Amount") reuses
  that extractor, with the client's label added as an anchor. Any other date,
  number or text field reuses the extractor for its dtype, anchored on the
  client's own label only. A selection field has no rule extractor; only the
  model can fill it.
- **Grounding** is unchanged. A model value is kept only if it is found in the
  page text. Dates and amounts are normalized by the existing normalizers. The
  global `dayfirst` option applies; `date_format` is a hint in the prompt only.

**Two cascade switches** in cascade.yaml:

- `vlm.input: image | text` (default `image`). `text` sends the extracted page
  text and no images, for a model that cannot read images. The text is cut
  page by page to `render.max_text_chars` across `render.max_pages` pages.
- The client already speaks `/v1/chat/completions`, which vLLM and Ollama
  (its OpenAI-compatible endpoint) both serve. Set `vlm.base_url` to
  `http://HOST:PORT/v1`. If the server wants a key, set `vlm.api_key_env` to the
  name of an environment variable that holds it. Never put the key itself in
  a config file.

**Multi-page files.** A PDF or TIFF is one document, even when it holds
several. Every TIFF frame is a page. `extraction.max_pages` (default 500) caps
the pages read. At the cap, the record's `diagnostics.errors` holds a
`page limit` line, so client-eval counts that document as failed.

**What is scored:** the class and the key field only (see above). Other
document fields are in the results but are not scored.

### Checklist for the client machine

1. Run `git pull`.
2. Run `pip install -e ".[dev,api]"`.
3. Set `extraction.ocr.binary` in `config/pipeline.yaml` to the tesseract path.
4. Set `vlm.base_url` and `vlm.model` in `config/cascade.yaml`. Add the model
   host to `vlm.allowed_hosts` if it is not loopback. Set `vlm.input: text` for
   a text-only model.
5. Keep the pack, the work directory and all outputs outside the repo. Never
   commit them.
