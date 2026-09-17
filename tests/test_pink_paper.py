"""Pink paper is read from the red channel with its own psm. Other paper is untouched."""

from PIL import Image

from baseline.extract import _prepare_pink_paper

RULE = {"pink_paper": {"min_red_over_green": 34, "max_green_blue_gap": 10, "psm": 6}, "psm": 3}


def test_pink_paper_is_read_from_the_red_channel():
    yaumi_pink = Image.new("RGB", (400, 500), (240, 200, 205))
    image, opts = _prepare_pink_paper(yaumi_pink, RULE)
    assert image.mode == "L"
    assert image.getpixel((0, 0)) == 240
    assert opts["psm"] == 6
    assert RULE["psm"] == 3  # the shared options are not changed


def test_white_and_yellow_paper_are_left_alone():
    for paper in [(250, 250, 250), (250, 244, 185)]:
        original = Image.new("RGB", (400, 500), paper)
        image, opts = _prepare_pink_paper(original, RULE)
        assert image is original
        assert opts is RULE


def test_no_rule_in_config_turns_it_off():
    pink = Image.new("RGB", (400, 500), (240, 200, 205))
    image, opts = _prepare_pink_paper(pink, {"psm": 3})
    assert image is pink
