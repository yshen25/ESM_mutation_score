# models/mutation_generator.py

from itertools import combinations, product
from typing import List, Tuple

DEFAULT_AAS = [aa for aa in "ACDEFGHIKLMNPQRSTVWY" if aa != "C"]

def generate_mutants(
    wt_seq: str,
    positions: List[int],
    allowed_aas: List[str] = None,
    mut_order: int = 2,
    exclude_consecutive: bool = True,
    skip_wt: bool = True,
    output_format: str = "list"  # "list", "dict", or "fasta"
):
    allowed_aas = allowed_aas or DEFAULT_AAS
    mutants = []

    for pos_combo in combinations(positions, mut_order):
        if exclude_consecutive and any(abs(p1 - p2) == 1 for p1, p2 in zip(pos_combo, pos_combo[1:])):
            continue

        from_residues = [wt_seq[p] for p in pos_combo]
        for to_aas in product(allowed_aas, repeat=mut_order):
            if all(f == t for f, t in zip(from_residues, to_aas)) and skip_wt:
                continue
            mutant = list(wt_seq)
            mut_info = []
            for p, to_aa in zip(pos_combo, to_aas):
                mut_info.append((p, mutant[p], to_aa))
                mutant[p] = to_aa
            mut_seq = "".join(mutant)

            if output_format == "list":
                mutants.append((mut_seq, mut_info))
            elif output_format == "dict":
                mutants.append({
                    "mutant_seq": mut_seq,
                    "mutations": mut_info,
                    "mut_key": ",".join([f"{f}{p}{t}" for (p, f, t) in mut_info])
                })
            elif output_format == "fasta":
                header = "|".join([f"{f}{p}{t}" for (p, f, t) in mut_info])
                mutants.append((header, mut_seq))
            else:
                raise ValueError(f"Unsupported output format: {output_format}")

    return mutants
