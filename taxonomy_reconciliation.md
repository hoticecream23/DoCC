# Taxonomy reconciliation: POC 1 deck vs baseline pipeline

Compares the document taxonomy in `POC_1_PRESENTATION_TRIMMED.docx` (section 5,
"Standardized Banking Document Taxonomy Matrix") against the classes actually
declared in `config/classes.yaml` and the label mapping in `config/taxonomy.yaml`.

It is a written record of where the two disagree, so that the disagreement is a
decision someone makes rather than a surprise found during a demo.

The eight classes marked **Declared** below have since been added to
`classes.yaml` from the deck alone. Everything else here is unchanged: no field
schemas, no DMS routing, and no documents to tune the new classes against.

## Counts

| Source | Real classes | Fallback | Total rows |
| --- | --- | --- | --- |
| Deck, sections 1 and 3 (prose) | 15 | 1 (`Unknown`) | 16 |
| Deck, section 5 (the table) | 20 | 1 (`Unknown`) | 21 |
| `config/classes.yaml` (before) | 9 | 1 (`unknown`, from `pipeline.yaml`) | 10 |
| `config/classes.yaml` (now) | 17 | 1 | 18 |

The deck contradicts itself: the headline "15 core banking classes + 1 explicit
Unknown fallback class" appears twice in prose, but its own taxonomy table lists
20 named classes. This needs fixing in the deck before it goes to a client,
whichever number is chosen.

## Class-by-class

`Status` values:

- **Covered** - the deck class maps onto a declared class with markers behind it.
- **Collapsed** - the deck splits into classes the pipeline deliberately treats
  as one. A prediction can never recover the deck's finer label.
- **Partial** - a related class exists but was not written for this document type,
  so the markers are not tuned for it.
- **Declared** - a class was added from the deck alone. Markers were written, not measured: no document we hold is an instance of one. Verified only in the negative, that they fire on 0 of the 112 labelled documents.

| Deck class | Baseline class | Status | Note |
| --- | --- | --- | --- |
| Invoice | `invoice` | Covered | 9 markers, tuned on 25 real client invoices |
| Receipt | `receipt` | Covered | 4 markers, incl. a negative lookbehind for "due upon receipt" |
| Purchase_Order | `purchase_order` | Covered | Verified against 1 real PO |
| Salary_Slip | `salary_slip` | Covered | |
| Bank_Statement | `bank_statement` | Covered | |
| Account_Statement | `bank_statement` | Collapsed | "statement of account" is a `bank_statement` marker at 0.94 |
| Transaction_Statement | `bank_statement` | Collapsed | Same markers, no way to separate |
| Passbook | `bank_statement` | Partial | No passbook-specific marker exists |
| Identity_Proof | `id_document` | Covered | |
| PAN_Card | `id_document` | Collapsed | `pan` exists as a *field* detector, not as a class |
| Loan_Agreement | `contract` | Partial | `contract` markers are generic agreement language |
| Income_Proof | `tax_form` | Partial | Only if the proof is a tax return; pay stubs land in `salary_slip` |
| Address_Proof | `address_proof` | Declared | |
| KYC_Form | `kyc_form` | Declared | |
| Customer_Application | `customer_application` | Declared | |
| Loan_Application | `loan_application` | Declared | |
| Sanction_Letter | `sanction_letter` | Declared | |
| Repayment_Schedule | `repayment_schedule` | Declared | Table extraction finds the grid; now there is a class to file it under |
| Account_Opening_Form | `account_opening_form` | Declared | |
| Account_Closure_Form | `account_closure_form` | Declared | |
| Unknown | `unknown` | Covered | Different mechanism, see below |

Score: 6 covered outright, 3 collapsed, 3 partial, 8 declared but unmeasured.

## Classes the pipeline has that the deck does not

| Baseline class | Why it matters |
| --- | --- |
| `court_document` | 46 files, **41% of the entire labelled client set** |
| `tax_form` | 5 files under the client's "tax document" label |

This is the largest finding in this document. The deck is a banking POC end to
end: PAN cards, KYC, loans, sanction letters, DMS cabinets named
`Loan_Management_Cabinet`. The corpus we are actually measured on is a legal
set - court judgments downloaded from indiankanoon.org, named
`<party>_vs_<party>_on_<date>.PDF`, filed under seven legal labels that
`config/taxonomy.yaml` collapses into one class because nothing in the text
distinguishes them.

Two thirds of the deck's taxonomy has no representation in the corpus at all,
and the largest single block of the corpus has no representation in the deck.

## The seven-to-one legal collapse

`config/taxonomy.yaml` maps seven client labels onto `court_document`:

Client Representations, Draft Agreements, Legal Opinions, Litigation Strategy
Notes, Research Notes and Case Notes, Writ Petitions and Appeals, Written
Submissions.

All are marked `lossy: true`. Four files sit under two labels at once, and three
of those four are different files that share a name. If the deck's approach were
applied to this corpus as written, it would have to either invent seven markers
that cannot exist, or make the same collapse. The deck does not discuss the
problem, because its taxonomy never met this corpus.

## Confidence routing

| | Deck | Baseline |
| --- | --- | --- |
| Mechanism | Single gate on softmax confidence | Four-stage cascade |
| Auto-file bar | `C >= 0.90` | `rule_short_circuit: 0.9` skips the model entirely |
| Model floor | not stated | `model_min_confidence: 0.45`, below which output is discarded |
| Weak rule bar | not stated | `rule_min_confidence: 0.7`, consulted only after the model declines |
| Fallback | `Unknown` routed to HITL queue | `unknown_label: unknown` |

Same headline number, 0.9, but it means different things. In the deck it is the
bar for automating a filing decision. In `pipeline.yaml` it is the bar at which a
rule is specific enough to skip the model. The deck has no equivalent of the
0.45 floor or the 0.7 weak-rule bar, and neither the deck's 0.90 nor its
"95% operational automation" claim has measured calibration behind it.

## Index fields

The deck names roughly 37 index fields across its 21 rows. `config/fields.yaml`
declares 15.

Overlapping: `pan` (deck `pan_number`), `invoice_number` (`invoice_no`),
`invoice_date`, `total_amount` (`total`), `account_number` (`acc_no`),
`vendor_name`.

In `fields.yaml`, not in the deck: `aadhaar`, `gstin`, `ifsc`, `card_number`,
`email`, `phone`, `due_date`, `tax_amount`, `buyer_name`.

In the deck, absent from `fields.yaml` - grouped by what they would take:

- **Person and party names** - `customer_name`, `applicant_name`,
  `borrower_name`, `lender_name`, `employee_name`, `merchant_name`. Six deck
  labels for one problem the pipeline has not solved at all. `vendor_name` and
  `buyer_name` are anchor-scoped to the invoice-family classes only.
- **Loan fields** - `loan_amount`, `loan_acc_no`, `principal`, `total_emi`,
  `emi_amount`, `annual_income`. All absent, along with the classes they belong to.
- **Salary breakdown** - `gross_salary`, `net_salary`, `company`. `salary_slip`
  exists as a class but extracts only `total_amount` from it.
- **Period fields** - `start_date`, `end_date`, `period`, `dob`, `po_date`.
- **Other** - `kyc_ref_id`, `id_number`, `address_line`, `account_type`,
  `branch`, `reason`, `po_number`, `doc_type`.

Note also that `templates: []` in `fields.yaml` - the positional extraction
strategy is implemented but has no templates registered, so the deck's zonal
comparison in section 7 is arguing against a strategy this pipeline has built
and not yet used.

## What is not reconcilable

The DMS routing layer in deck section 6 - the pre-existing
Cabinet/Drawer/Folder/File path resolution - has no counterpart anywhere in
`baseline/`. There is no code that resolves a target path, and nothing to
reconcile it against. It is a whole component, not a gap in an existing one.

## Suggested decisions

1. Fix the 15-vs-20 contradiction in the deck. Pick one number.
2. Decide whether the POC is banking or legal. The deck says banking; the corpus
   is 41% court documents. Both cannot be the demo.
3. The 8 banking classes are now declared in `classes.yaml`. Roughly 22 fields
   are still missing from `fields.yaml`, and there is still no corpus to tune
   any of it against, so the classes cannot yet be claimed to work.
4. Either drop the 95% automation claim or produce the coverage-versus-accuracy
   curve that supports it. The harness can already generate it.
