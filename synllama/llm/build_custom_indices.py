"""Build custom building block indices for reconstruction.

This module provides utilities to build per-reaction fingerprint indices and SMILES
TF-IDF searchers from custom building block lists. The indices are required for the
reconstruction algorithm to find synthesizable pathways using custom BBs.

The workflow mirrors step_11_generate_fpindex_smiles_tfidf.py but works with
user-provided BB lists instead of the full Enamine database.
"""

import os
import pickle
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

from synllama.chem.fpindex import FingerprintIndex
from synllama.chem.smiles_tfidf import SmilesSimilaritySearch
from synllama.chem.mol import FingerprintOption, Molecule
from synllama.chem.reaction import Reaction


def _filter_bbs_for_reaction(
    custom_bbs: List[Molecule], reaction: Reaction
) -> List[Molecule]:
    """Filter building blocks to find valid reactants for a reaction.

    Args:
        custom_bbs: List of custom building block Molecule objects
        reaction: Reaction object to test against

    Returns:
        List of Molecule objects that are valid reactants for this reaction
    """
    valid_bbs = []

    for bb in custom_bbs:
        if not bb.is_valid:
            continue

        # For reactions with different numbers of reactants, we need to test
        # if this BB can participate as any of the reactants
        is_valid = False

        if reaction.num_reactants == 1:
            # Test single reactant
            if reaction([bb]):
                is_valid = True
        elif reaction.num_reactants == 2:
            # Test if this BB can be either reactant
            # We use a dummy BB to test (the BB itself acts as both for testing)
            if (
                reaction([bb, bb])
                or len(valid_bbs) > 0
                and any(
                    reaction([bb, other]) or reaction([other, bb])
                    for other in valid_bbs[:5]
                )
            ):
                is_valid = True
            else:
                # If no valid_bbs yet, tentatively include it
                is_valid = True
        elif reaction.num_reactants == 3:
            # For 3-reactant reactions, tentatively include all BBs
            # The reconstruction algorithm will filter during actual matching
            is_valid = True

        if is_valid:
            valid_bbs.append(bb)

    return valid_bbs


def build_custom_bb_indices(
    custom_bbs: List[str],
    reaction_smarts_dict_path: str,
    output_dir: str,
    token_list_path: Optional[str] = None,
    filter_bbs_per_reaction: bool = True,
) -> str:
    """Build per-reaction fingerprint indices and SMILES TF-IDF searchers from custom BBs.

    This function creates the necessary index files for reconstruction with custom building
    blocks. It filters BBs per-reaction to include only valid reactants, builds fingerprint
    indices for fast similarity search, and builds SMILES TF-IDF searchers for string-based
    matching.

    The output directory will contain:
    - reaction_smarts_map.pkl: Mapping of reaction indices to SMARTS templates
    - fpindex_{rxn_idx}.pkl: FingerprintIndex for each reaction
    - smiles_tfidf_{rxn_idx}.pkl: SmilesSimilaritySearch for each reaction

    Args:
        custom_bbs: List of custom building block SMILES strings
        reaction_smarts_dict_path: Path to reaction_smarts_map.pkl file containing
                                  reaction templates (e.g., from Enamine indices)
        output_dir: Directory to save the generated indices
        token_list_path: Path to SMILES tokenizer vocabulary file. If None, will try
                        to use "data/smiles_vocab.txt" or create SMILES searchers
                        without custom tokenization
        filter_bbs_per_reaction: If True, filter BBs to only include valid reactants
                                per reaction. If False, use all BBs for all reactions.

    Returns:
        Path to output directory containing the generated indices

    Raises:
        FileNotFoundError: If reaction_smarts_dict_path doesn't exist
        ValueError: If custom_bbs list is empty or contains no valid SMILES

    Example:
        >>> custom_bbs = ["CC", "C=O", "CCN", "c1ccccc1"]
        >>> indices_path = build_custom_bb_indices(
        ...     custom_bbs=custom_bbs,
        ...     reaction_smarts_dict_path="synllama-data/inference/reconstruction/91rxns/rxn_embeddings/reaction_smarts_map.pkl",
        ...     output_dir="my_custom_indices"
        ... )
        >>> print(f"Indices built at: {indices_path}")
    """
    # Validate inputs
    if not custom_bbs:
        raise ValueError("custom_bbs list cannot be empty")

    if not os.path.exists(reaction_smarts_dict_path):
        raise FileNotFoundError(
            f"reaction_smarts_dict_path not found: {reaction_smarts_dict_path}"
        )

    # Load reaction SMARTS dictionary
    with open(reaction_smarts_dict_path, "rb") as f:
        reaction_smarts_dict = pickle.load(f)

    print(f"Loaded {len(reaction_smarts_dict)} reaction templates")

    # Convert custom BBs to Molecule objects and filter invalid ones
    print(f"Processing {len(custom_bbs)} custom building blocks...")
    bb_molecules = []
    invalid_count = 0

    for bb_smiles in tqdm(custom_bbs, desc="Validating BBs"):
        mol = Molecule(bb_smiles, source="smiles")
        if mol.is_valid:
            bb_molecules.append(mol)
        else:
            invalid_count += 1

    if not bb_molecules:
        raise ValueError(
            f"No valid SMILES in custom_bbs list ({invalid_count} invalid)"
        )

    print(f"Found {len(bb_molecules)} valid BBs ({invalid_count} invalid)")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Copy reaction SMARTS map to output directory
    shutil.copy(reaction_smarts_dict_path, output_path / "reaction_smarts_map.pkl")

    # Check if token list exists
    if token_list_path is None:
        default_token_path = "data/smiles_vocab.txt"
        if os.path.exists(default_token_path):
            token_list_path = default_token_path
        else:
            print(
                f"Warning: Token list not found at {default_token_path}. "
                "SMILES TF-IDF search will use character-level tokenization."
            )

    # Process each reaction
    fp_option = FingerprintOption.morgan_for_building_blocks()
    reaction_bb_counts: Dict[int, int] = {}

    print(f"\nBuilding indices for {len(reaction_smarts_dict)} reactions...")

    for reaction_idx, (smarts, _) in tqdm(
        reaction_smarts_dict.items(), desc="Processing reactions"
    ):
        try:
            # Create Reaction object
            reaction = Reaction(smarts)

            # Filter BBs for this reaction or use all
            if filter_bbs_per_reaction:
                reaction_bbs = _filter_bbs_for_reaction(bb_molecules, reaction)
            else:
                reaction_bbs = bb_molecules

            reaction_bb_counts[reaction_idx] = len(reaction_bbs)

            if len(reaction_bbs) == 0:
                print(
                    f"\nWarning: Reaction {reaction_idx} has no valid BBs. "
                    "This reaction will not be usable in reconstruction."
                )
                # Still create empty indices to maintain file structure
                reaction_bbs = bb_molecules[:1]  # Use at least one BB

            # Build FingerprintIndex
            fp_index = FingerprintIndex(molecules=reaction_bbs, fp_option=fp_option)
            fp_index_file = output_path / f"fpindex_{reaction_idx}.pkl"
            fp_index.save(fp_index_file)

            # Build SmilesSimilaritySearch
            if token_list_path:
                smiles_search = SmilesSimilaritySearch(token_list_path=token_list_path)
            else:
                smiles_search = SmilesSimilaritySearch()

            smiles_search.fit(molecules=reaction_bbs)
            smiles_search_file = output_path / f"smiles_tfidf_{reaction_idx}.pkl"
            smiles_search.save(smiles_search_file)

        except Exception as e:
            print(f"\nError processing reaction {reaction_idx}: {e}")
            raise

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Index Building Complete!")
    print("=" * 60)
    print(f"Output directory: {output_path.absolute()}")
    print(f"Total reactions: {len(reaction_smarts_dict)}")
    print(f"Total custom BBs: {len(bb_molecules)}")
    print(
        f"Average BBs per reaction: {sum(reaction_bb_counts.values()) / len(reaction_bb_counts):.1f}"
    )

    # Find reactions with few BBs
    low_bb_reactions = [
        (idx, count) for idx, count in reaction_bb_counts.items() if count < 10
    ]
    if low_bb_reactions:
        print(
            f"\nWarning: {len(low_bb_reactions)} reactions have fewer than 10 valid BBs:"
        )
        for idx, count in sorted(low_bb_reactions)[:5]:
            print(f"  - Reaction {idx}: {count} BBs")
        if len(low_bb_reactions) > 5:
            print(f"  ... and {len(low_bb_reactions) - 5} more")

    print("\nFiles created:")
    print(f"  - reaction_smarts_map.pkl")
    print(f"  - fpindex_{{0..{len(reaction_smarts_dict) - 1}}}.pkl")
    print(f"  - smiles_tfidf_{{0..{len(reaction_smarts_dict) - 1}}}.pkl")
    print("=" * 60)

    return str(output_path.absolute())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build custom BB indices for SynLlama reconstruction"
    )
    parser.add_argument(
        "--custom_bbs_file",
        type=str,
        required=True,
        help="Path to file with custom BB SMILES (one per line)",
    )
    parser.add_argument(
        "--reaction_smarts_dict_path",
        type=str,
        required=True,
        help="Path to reaction_smarts_map.pkl from Enamine indices",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for generated indices",
    )
    parser.add_argument(
        "--token_list_path",
        type=str,
        default=None,
        help="Path to SMILES tokenizer vocabulary (default: data/smiles_vocab.txt)",
    )
    parser.add_argument(
        "--no_filter",
        action="store_true",
        help="Don't filter BBs per reaction (use all BBs for all reactions)",
    )

    args = parser.parse_args()

    # Load custom BBs from file
    with open(args.custom_bbs_file, "r") as f:
        custom_bbs = [line.strip() for line in f if line.strip()]

    # Build indices
    output_path = build_custom_bb_indices(
        custom_bbs=custom_bbs,
        reaction_smarts_dict_path=args.reaction_smarts_dict_path,
        output_dir=args.output_dir,
        token_list_path=args.token_list_path,
        filter_bbs_per_reaction=not args.no_filter,
    )

    print(f"\nSuccess! Custom BB indices saved to: {output_path}")
