from models.mutation_generator import generate_mutants

def test_double_mutants():
    wt = "AAGGTT"
    positions = [0, 2, 4]
    muts = generate_mutants(wt, positions, mut_order=2, output_format="dict")
    assert all(len(m['mutations']) == 2 for m in muts)
    assert all(len(m['mutant_seq']) == len(wt) for m in muts)
