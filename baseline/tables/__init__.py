"""Table and line item extraction.

A separate pass over a results file plus its bbox sidecar, exactly like the
knowledge graph. It never touches DocumentRecord, so the record contract stays
frozen and tables can be rebuilt from any run at any time.

Three strategies, run in config order, first hit wins per document:

  tab_delimited  Rows the extractor wrote tab separated, which is how .docx
             tables arrive. A tab is a declared cell boundary, so it runs
             first.
  geometry   Columns from word bounding boxes. The right signal for real PDFs
             and scans, where x positions are what the layout actually is.
  text_grid  Columns from runs of whitespace that are blank on every line of a
             band. The only signal available for .txt and .docx, which carry
             no geometry at all.

The v0 rule, borrowed from the graph: a wrong table is worse than a missing
one. Every structural check is a reason to abstain, and every abstention is
recorded with its reason rather than swallowed.

- `geometry.py`   the geometry strategy
- `text_grid.py`  the text grid strategy
- `tab_delimited.py` the tab delimited strategy
- `assemble.py`   a finished grid of cells into a Table, shared by all three
- `records.py`    one document through the strategies, and the whole pass
"""

from .geometry import detect_geometry
from .records import build, build_record, load_results, load_words, render_report
from .tab_delimited import detect_tab_delimited
from .text_grid import detect_text_grid

__all__ = [
    "build", "build_record", "detect_geometry", "detect_tab_delimited", "detect_text_grid",
    "load_results", "load_words", "render_report",
]
