"""Page limit and multi-frame images: off in normal mode, on in pack mode or when configured."""

from dataclasses import replace

import pytest
import yaml
from PIL import Image

from baseline.config import load_config
from baseline.errors import ConfigError
from baseline.extract import extract


def _with(cfg, **extraction):
    return replace(cfg, pipeline={**cfg.pipeline, "extraction": {**cfg.extract_opts(), **extraction}})


def _tiff(tmp_path, frames):
    path = tmp_path / "frames.tif"
    images = [Image.new("RGB", (200, 200), "white") for _ in range(frames)]
    images[0].save(path, save_all=True, append_images=images[1:])
    return path


def test_a_pdf_over_the_limit_is_cut_and_recorded(corpus, cfg):
    res = extract(corpus / "invoice_native.pdf", _with(cfg, max_pages=1))
    assert res.page_count == 1 and any(e.startswith("page limit") for e in res.errors)


def test_normal_mode_reads_every_pdf_page(corpus, cfg):
    res = extract(corpus / "invoice_native.pdf", cfg)
    assert res.page_count == 2 and not any(e.startswith("page limit") for e in res.errors)


def test_normal_mode_reads_frame_zero_of_a_multi_frame_tiff(cfg, tmp_path):
    assert extract(_tiff(tmp_path, 3), cfg).page_count == 1


def test_multi_frame_images_reads_every_tiff_frame(cfg, tmp_path):
    assert extract(_tiff(tmp_path, 3), _with(cfg, multi_frame_images=True)).page_count == 3


def test_multi_frame_images_stops_at_the_page_limit(cfg, tmp_path):
    assert extract(_tiff(tmp_path, 3), _with(cfg, multi_frame_images=True, max_pages=2)).page_count == 2


def _config_dir(tmp_path, root):
    for src in (root / "config").glob("*.yaml"):
        (tmp_path / src.name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize(
    "section, line",
    [
        ("extraction", "  max_page: 5"),
        ("extraction", "  max_pages: five"),
        ("extraction", "  multi_frame_images: maybe"),
        ("client_pack", "  cabnet: Acme"),
    ],
)
def test_an_unknown_or_mistyped_key_fails_the_load(section, line, tmp_path, request):
    root = request.config.rootpath
    text = (root / "config" / "pipeline.yaml").read_text(encoding="utf-8")
    text = text.replace(f"\n{section}:\n", f"\n{section}:\n{line}\n", 1)
    cdir = _config_dir(tmp_path, root)
    (cdir / "pipeline.yaml").write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match=section):
        load_config(cdir)


def test_normal_mode_notes_the_skipped_tiff_frames_without_an_error(cfg, tmp_path):
    res = extract(_tiff(tmp_path, 3), cfg)
    assert any("2 skipped (extraction.multi_frame_images)" in n for n in res.notes)
    assert not any("multi_frame_images" in e for e in res.errors)


def test_a_null_extraction_section_loads_as_the_defaults(tmp_path, request):
    cdir = _config_dir(tmp_path, request.config.rootpath)
    data = yaml.safe_load((cdir / "pipeline.yaml").read_text(encoding="utf-8"))
    (cdir / "pipeline.yaml").write_text(yaml.safe_dump({**data, "extraction": None}), encoding="utf-8")
    assert load_config(cdir).extract_opts() == {}
