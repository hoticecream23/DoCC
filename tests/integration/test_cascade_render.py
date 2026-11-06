import io

from PIL import Image


def _open(png: bytes) -> Image.Image:
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    return Image.open(io.BytesIO(png))


def test_pdf_renders_one_image_per_page_up_to_the_limit(corpus):
    from cascade.render import render_pages

    pages = render_pages(corpus / "invoice_native.pdf", max_pages=4, dpi=72, max_long_edge=5000)
    assert len(pages) == 2
    assert render_pages(corpus / "invoice_native.pdf", max_pages=1, dpi=72, max_long_edge=5000)[0] == pages[0]


def test_long_edge_is_capped(corpus):
    from cascade.render import render_pages

    img = _open(render_pages(corpus / "invoice_native.pdf", max_pages=1, dpi=300, max_long_edge=500)[0])
    assert max(img.size) == 500


def test_an_image_file_gives_one_rgb_png(tmp_path):
    from cascade.render import render_pages

    path = tmp_path / "scan.jpg"
    Image.new("CMYK", (120, 80)).save(path)
    pages = render_pages(path, max_pages=4, dpi=150, max_long_edge=1600)
    assert len(pages) == 1
    img = _open(pages[0])
    assert img.mode == "RGB" and img.size == (120, 80)


def test_formats_without_a_page_image_give_none(corpus):
    from cascade.render import render_pages

    assert render_pages(corpus / "salary_slip.docx", 4, 150, 1600) == []
    assert render_pages(corpus / "contract.txt", 4, 150, 1600) == []
