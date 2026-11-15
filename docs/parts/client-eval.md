# Client evaluation: what each number means

`baseline client-eval` and `baseline client-run` score one cabinet of a client
pack. Code: `baseline/evaluate/client_eval.py`. This page defines every number
in the report, its denominator, and what counts as wrong.

## What is scored

- Only the class and the key field are scored. The pack has no gold for any
  other field.
- The gold class is the folder a document sat in. Folder labels may be
  imperfect: a misfiled document is scored against the wrong label.
- The gold key comes from the file name (`key_gold` in `manifest.csv`). Not
  every type has a key, and not every file name carries one.
- Results are joined to the manifest by file stem, which is the `doc_id`.

## Document outcomes

| Outcome | Meaning | How it is scored |
|---|---|---|
| No result | The manifest lists the document, the results file has no line for it. The pipeline failed on it, or the run did not reach it. | Wrong for class, key and case. Listed by name under "No result". |
| Produced a result with errors | The line exists and its `diagnostics.errors` is not empty. | Scored as it is. A right class still counts as right. Listed by name. |
| Abstained | The line exists with no class or the label `unknown`. | Wrong. |
| Unlabelled | The manifest row has no gold label. | Not scored. Counted only. |
| Attachment | The row has `role: attachment`: a file beside the document its folder is named after. | Not a document: in no denominator or outcome count. Counted under "Attachments". |
| Duplicated | A stem is on more than one line. | The last line is used. The count of such stems is reported. |

`client-run` starts from an empty results file, so stale lines from an earlier
run are never scored, and moves the old report to `client_eval.md.previous`.
`--resume` keeps the earlier lines and processes only new documents. The
report is written only after scoring succeeds; a failed run leaves a stub that
names the failed stage and the time.

**Possible duplicates.** The overview gives the number of labelled documents
whose `dup_group` is set: the same name and size as a file in another folder.
They are scored as usual; the count shows how much the scores may lean on
copies.

## Classification (single documents)

- **Accuracy** = documents classed right / every labelled single document in
  the manifest. Documents in a case are scored as cases, not here.
- **Right** means the predicted class equals the gold class, or one of the gold
  class's aliases from `taxonomy.yaml`. The compare ignores case and collapses
  whitespace. A prediction that equals a class's own name always means that
  class, even when another class lists the same text as an alias.
- No result and abstained are wrong. The headline reads
  `Accuracy 2/4 (50.0%) — 2 documents produced no result`.
- **Per-class table**: one row per class that is gold or predicted, plus
  `(no result)` and `(abstained)` rows. `gold` is the count with that gold
  class (support). `predicted` is the count predicted as it. Precision =
  right / predicted. Recall = right / gold. The `gold` column and the
  `predicted` column each sum to the accuracy denominator.

## Key field

Denominator: every labelled document with a `key_gold`, with or without a
result. The predicted value is read from the key field of the **gold** type,
even when the predicted class is wrong. Each field's raw value and normalized
value are both compared; a match on either counts. A predicted value such as
`4502424845.0` (digits, a dot, zeros only) is compared as `4502424845` before
every test.

- **Key exact**: equal after trimming outer whitespace, case folding,
  collapsing inner whitespace, and mapping unicode dashes to `-`. Nothing else.
  `1-2345` and `12-345` are not exact.
- **Key loose (ignoring separators)**: equal after also removing every space and
  punctuation mark. `1-2345` and `12-345` are loose. Every exact match is also
  loose.
- **Key contained**: the gold appears in any extracted value of the document
  (any field) as a whole run: no letter or digit right before or after it.
  Separators between the gold's own parts are ignored. Denominator: key
  documents whose gold has at least 6 letters and digits. Shorter golds are
  reported as "too short to test". An empty gold never matches.
- **Class and key both right (exact)**: exact key and right class / every key
  document. The report also gives the exact rate among key documents whose
  class was right.

## Case bundles

- A case's gold is the majority label of its files. Cases whose files carry
  different labels are listed as "mixed".
- **All files right**: every file is predicted as the case gold. A file with
  no result or an abstention makes the case wrong.
- **Any file right**: at least one file is predicted as the case gold.
- **Majority right**: one class is predicted more often than any other across
  the case's files, and it is the case gold. No result and abstained are not
  votes. A tie for the top is not right, so the answer never depends on the
  manifest order. A case with no vote (every file missing or abstained) is
  wrong.
- Denominator for all three: every case with a label.

## Percentages

A percentage is floored to one decimal, so `100.0%` means every document is
right: 9996/10000 prints as 99.9%. A zero denominator prints `n/a`.

## Report header

`client-run` writes the run start time (UTC) and duration, the stages that ran
(`baseline only` or `baseline + model (<model name>)`) and the results file
path.

## Worked example (made-up numbers)

Cabinet "Northwind": 10 manifest documents. 1 unlabelled. 2 in one case.
7 labelled single documents, 6 with a key.

| Document | Gold | Predicted | Gold key | Key field value |
|---|---|---|---|---|
| 1 | Invoice | Invoice | INV-100201 | INV-100201 |
| 2 | Invoice | invoice | INV-100202 | INV 100202 |
| 3 | Invoice | Receipt | INV-100203 | INV-100203 |
| 4 | Receipt | Receipt | R-55 | R55 |
| 5 | Receipt | unknown | RC-700100 | none; text has "ref RC 700100" |
| 6 | Receipt | no result | RC-700101 | none |
| 7 | Memo | Memo | none | none |

- Accuracy: 1, 2 (case ignored), 4, 7 are right = 4/7 (57.1%). 1 document
  produced no result (6). 5 abstained.
- Key exact: 1, 3 = 2/6 (33.3%). Document 3 counts: the key is read from
  the Invoice key field although the class is wrong.
- Key loose: 1, 2, 3, 4 = 4/6 (66.6%).
- Key contained: document 4's gold has 3 letters and digits, too short to
  test. Denominator 5. Hits: 1, 2, 3, 5 = 4/5 (80.0%).
- Class and key both right (exact): 1 = 1/6 (16.6%). Of the 3 key documents
  whose class was right (1, 2, 4), 1/3 (33.3%).
- Per-class rows: Invoice (gold 3, predicted 2), Receipt (gold 3, predicted 2),
  Memo (1, 1), `(abstained)` (0, 1), `(no result)` (0, 1). Both columns sum
  to 7.
- The case: files predicted Claim, Claim, gold Claim. All files right 1/1,
  any file right 1/1, majority right 1/1. Had they been Claim and Invoice,
  only any file right would hold: the vote is a tie.
