# tests/test_mutgen.py

from esm_mutation_score import generate_mutants

def test_single_mutation():
    wt = "ACDE"
    mutants = generate_mutants(wt, [1, 2], mut_order=1, allowed_aas=["Y"], output_format="list")
    assert any(m[0] == "AYDE" for m in mutants)
    assert any(m[0] == "ACYE" for m in mutants)
