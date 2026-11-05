from baseline.metadata import NormCtx, normalize

CTX = NormCtx(full="", start=0, end=0, opts={})


def test_name_norm_ignores_case():
    assert normalize("name_norm", "John doe", CTX)[0] == normalize("name_norm", "John Doe", CTX)[0]


def test_name_norm_collapses_spaces_and_end_punctuation():
    assert normalize("name_norm", "  Kedai   Abc Sdn. Bhd., ", CTX)[0] == "KEDAI ABC SDN. BHD."


def test_name_norm_keeps_inner_punctuation():
    assert normalize("name_norm", "(TAN S/B)", CTX)[0] == "TAN S/B"
