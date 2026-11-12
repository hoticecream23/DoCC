"""A file over the page limit is read up to it and says so; a multi-page TIFF is one document."""

from dataclasses import replace

from PIL import Image

from baseline.extract import extract


def _limited(cfg, pages):
    return replace(cfg, pipeline={**cfg.pipeline, "extraction": {**cfg.extract_opts(), "max_pages": pages}})


def test_a_pdf_over_the_limit_is_cut_and_recorded(corpus, cfg):
    res = extract(corpus / "invoice_native.pdf", _limited(cfg, 1))
    assert res.page_count == 1 and any(e.startswith("page limit") for e in res.errors)


def test_every_tiff_frame_is_a_page(cfg, tmp_path):
    path = tmp_path / "two.tif"
    frames = [Image.new("RGB", (200, 200), "white") for _ in range(2)]
    frames[0].save(path, save_all=True, append_images=frames[1:])
    assert extract(path, cfg).page_count == 2
