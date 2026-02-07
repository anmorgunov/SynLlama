"""BB reconstruction module for custom datasets.

This module provides functions to reconstruct synthesis pathways using custom
building blocks (BBs). It adapts the reconstruction algorithm from step_31_enamine_reconstruct.py
to work with user-provided BB lists instead of the Enamine database.

It also provides raw output filtering functions adapted from step_30_0_benchmark_filter_raw_output.py
for validating generated BBs against custom BB lists.
"""

import copy
import pickle
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import numpy as np
import pandas as pd
from rdkit import Chem

from synllama.chem.smiles_tfidf import (
    SmilesSimilaritySearch,
    find_closest_match,
    string_similarity,
)
from synllama.chem.fpindex import FingerprintIndex, compute_fingerprints
from synllama.chem.mol import FingerprintOption, Molecule
from synllama.chem.reaction import Reaction
from synllama.chem.stack import Stack


def similarity_score(product_template: str, stack_prod_smiles: str) -> float:
    """Calculate similarity score between product template and stack product.

    Adapted from step_31_enamine_reconstruct.py:similarity_score

    Args:
        product_template: Expected product SMILES
        stack_prod_smiles: Actual product SMILES from stack

    Returns:
        Similarity score (0-1)
    """
    if not Chem.MolFromSmiles(product_template):
        return string_similarity(product_template, stack_prod_smiles)
    else:
        return Molecule(product_template).sim(
            Molecule(stack_prod_smiles),
            FingerprintOption.morgan_for_tanimoto_similarity(),
        )


def get_top_k_smiles(
    input_smiles: str,
    smiles_searcher: SmilesSimilaritySearch,
    fp_searcher: FingerprintIndex,
    k: int = 10,
) -> List[Molecule]:
    """Get the top k similar SMILES from searchers.

    Adapted from step_31_enamine_reconstruct.py:get_top_k_smiles

    Args:
        input_smiles: Input SMILES to search for
        smiles_searcher: SMILES similarity searcher
        fp_searcher: Fingerprint index searcher
        k: Number of top similar SMILES to return

    Returns:
        List of top k similar Molecule objects
    """
    input_mol = Chem.MolFromSmiles(input_smiles)
    if input_mol is None:
        searched_smiles = smiles_searcher.query(input_smiles, k=k * 2)
        results = [result.molecule.smiles for result in searched_smiles]
        result_mols = [Molecule(s, source="smiles") for s in results]
    else:
        searched_smiles = smiles_searcher.query(input_smiles, k=k)
        results = [result.molecule.smiles for result in searched_smiles]
        result_mols = [Molecule(s, source="smiles") for s in results]
        fingerprints = Molecule(input_smiles).get_fingerprint(
            FingerprintOption.morgan_for_building_blocks(), as_bitvec=False
        )
        fp_searched_results = fp_searcher.query_single(np.array([fingerprints]), k=k)
        results.extend([result.molecule.smiles for result in fp_searched_results])
        result_mols.extend([Molecule(s, source="fp") for s in results])
    return list(set(result_mols))


def match_two_reactants(
    reactant1_list: List[Molecule],
    reactant2_list: List[Molecule],
    rxn: Reaction,
    continue_rxn: bool = False,
) -> List:
    """Match two reactants with a reaction.

    Adapted from step_31_enamine_reconstruct.py:match_two_reactants

    Args:
        reactant1_list: List of potential first reactants
        reactant2_list: List of potential second reactants
        rxn: Reaction object
        continue_rxn: If True, return only second reactant; if False, return combo

    Returns:
        List of valid reactant combinations or reactants
    """
    valid_combinations = []
    for reactant1 in reactant1_list:
        for reactant2 in reactant2_list:
            reactant_combo1 = [reactant1, reactant2]
            reactant_combo2 = [reactant2, reactant1]
            if rxn(reactant_combo1) or rxn(reactant_combo2):
                if continue_rxn:
                    valid_combinations.append(reactant2)
                else:
                    valid_combinations.append(reactant_combo1)
    return valid_combinations


def match_three_reactants(
    reactant1_list: List[Molecule],
    reactant2_list: List[Molecule],
    reactant3_list: List[Molecule],
    rxn: Reaction,
    continue_rxn: bool = False,
) -> List:
    """Match three reactants with a reaction.

    Adapted from step_31_enamine_reconstruct.py:match_three_reactants

    Args:
        reactant1_list: List of potential first reactants
        reactant2_list: List of potential second reactants
        reactant3_list: List of potential third reactants
        rxn: Reaction object
        continue_rxn: If True, return [reactant2, reactant3]; if False, return full combo

    Returns:
        List of valid reactant combinations or pairs
    """
    valid_combinations = []
    for reactant1 in reactant1_list:
        for reactant2 in reactant2_list:
            for reactant3 in reactant3_list:
                reactant_combo1 = [reactant1, reactant2, reactant3]
                reactant_combo2 = [reactant1, reactant3, reactant2]
                reactant_combo3 = [reactant2, reactant1, reactant3]
                reactant_combo4 = [reactant2, reactant3, reactant1]
                reactant_combo5 = [reactant3, reactant1, reactant2]
                reactant_combo6 = [reactant3, reactant2, reactant1]
                if (
                    rxn(reactant_combo1)
                    or rxn(reactant_combo2)
                    or rxn(reactant_combo3)
                    or rxn(reactant_combo4)
                    or rxn(reactant_combo5)
                    or rxn(reactant_combo6)
                ):
                    if continue_rxn:
                        valid_combinations.append([reactant2, reactant3])
                    else:
                        valid_combinations.append(reactant_combo1)
    return valid_combinations


def reconstruct_single_rxn(
    smiles_to_search: List[str],
    product_template: str,
    smiles_searcher: SmilesSimilaritySearch,
    fp_searcher: FingerprintIndex,
    template: str,
    rxn_idx: int,
    stacks: Optional[List[Stack]] = None,
    k: int = 5,
    n_stacks: int = 25,
    product_limit: int = 3,
) -> Optional[List[Stack]]:
    """Reconstruct a single reaction from building blocks and reactants.

    Adapted from step_31_enamine_reconstruct.py:reconstruct_single_rxn

    Args:
        smiles_to_search: List of SMILES to search for similar BBs
        product_template: Expected product SMILES
        smiles_searcher: SMILES similarity searcher
        fp_searcher: Fingerprint index searcher
        template: Reaction SMARTS template
        rxn_idx: Reaction index
        stacks: List of Stack objects to continue from (if None, start fresh)
        k: Number of similar BBs to search
        n_stacks: Maximum number of stacks to keep
        product_limit: Limit on number of products per reaction

    Returns:
        List of Stack objects with reconstructed pathways, or None if failed
    """
    rxn = Reaction(template)
    new_stacks = []

    if stacks is None:
        stacks = []

    if len(stacks) > 0 and len(stacks[0]) > 0:
        scores = []
        for stack in stacks:
            prev_mol = list(stack.get_top())

            if rxn.num_reactants == 1:
                assert len(smiles_to_search) == 0
                success = stack.push_rxn(
                    rxn,
                    rxn_idx,
                    product_template=product_template,
                    product_limit=product_limit,
                )
                if success:
                    new_stacks.append(stack)

            elif rxn.num_reactants == 2:
                assert len(smiles_to_search) == 1
                top_bbs_reactants = get_top_k_smiles(
                    smiles_to_search[0], smiles_searcher, fp_searcher, k
                )
                valid_mols = match_two_reactants(
                    prev_mol, top_bbs_reactants, rxn, continue_rxn=True
                )
                for mol in valid_mols:
                    new_stack = copy.deepcopy(stack)
                    new_stack.push_mol(mol, 0)
                    success = new_stack.push_rxn(
                        rxn,
                        rxn_idx,
                        product_template=product_template,
                        product_limit=product_limit,
                    )
                    if success:
                        scores.append(
                            similarity_score(product_template, new_stack[-1].smiles)
                        )
                        new_stacks.append(new_stack)

            elif rxn.num_reactants == 3:
                assert len(smiles_to_search) == 2
                top_bbs_reactants1 = get_top_k_smiles(
                    smiles_to_search[0], smiles_searcher, fp_searcher, k
                )
                top_bbs_reactants2 = get_top_k_smiles(
                    smiles_to_search[1], smiles_searcher, fp_searcher, k
                )
                valid_mols = match_three_reactants(
                    prev_mol,
                    top_bbs_reactants1,
                    top_bbs_reactants2,
                    rxn,
                    continue_rxn=True,
                )
                for mol1, mol2 in valid_mols:
                    new_stack = copy.deepcopy(stack)
                    new_stack.push_mol(mol1, 0)
                    new_stack.push_mol(mol2, 0)
                    success = new_stack.push_rxn(
                        rxn,
                        rxn_idx,
                        product_template=product_template,
                        product_limit=product_limit,
                    )
                    if success:
                        scores.append(
                            similarity_score(product_template, new_stack[-1].smiles)
                        )
                        new_stacks.append(new_stack)
    else:
        scores = []
        if rxn.num_reactants == 3:
            assert len(smiles_to_search) == 3
            top_bbs_reactants1 = get_top_k_smiles(
                smiles_to_search[0], smiles_searcher, fp_searcher, k // 2 + 1
            )
            top_bbs_reactants2 = get_top_k_smiles(
                smiles_to_search[1], smiles_searcher, fp_searcher, k // 2 + 1
            )
            top_bbs_reactants3 = get_top_k_smiles(
                smiles_to_search[2], smiles_searcher, fp_searcher, k // 2 + 1
            )
            valid_mols = match_three_reactants(
                top_bbs_reactants1,
                top_bbs_reactants2,
                top_bbs_reactants3,
                rxn,
                continue_rxn=False,
            )
            for mol1, mol2, mol3 in valid_mols:
                new_stack = Stack()
                new_stack.push_mol(mol1, 0)
                new_stack.push_mol(mol2, 0)
                new_stack.push_mol(mol3, 0)
                success = new_stack.push_rxn(
                    rxn,
                    rxn_idx,
                    product_template=product_template,
                    product_limit=product_limit,
                )
                if success:
                    scores.append(
                        similarity_score(product_template, new_stack[-1].smiles)
                    )
                    new_stacks.append(new_stack)

        elif rxn.num_reactants == 2:
            assert len(smiles_to_search) == 2
            top_bbs_reactants1 = get_top_k_smiles(
                smiles_to_search[0], smiles_searcher, fp_searcher, k
            )
            top_bbs_reactants2 = get_top_k_smiles(
                smiles_to_search[1], smiles_searcher, fp_searcher, k
            )
            valid_mols = match_two_reactants(
                top_bbs_reactants1, top_bbs_reactants2, rxn, continue_rxn=False
            )
            for mol1, mol2 in valid_mols:
                new_stack = Stack()
                new_stack.push_mol(mol1, 0)
                new_stack.push_mol(mol2, 0)
                success = new_stack.push_rxn(
                    rxn,
                    rxn_idx,
                    product_template=product_template,
                    product_limit=product_limit,
                )
                if success:
                    scores.append(
                        similarity_score(product_template, new_stack[-1].smiles)
                    )
                    new_stacks.append(new_stack)

        elif rxn.num_reactants == 1:
            assert len(smiles_to_search) == 1
            top_bbs_reactants = get_top_k_smiles(
                smiles_to_search[0], smiles_searcher, fp_searcher, k
            )
            for mol in top_bbs_reactants:
                new_stack = Stack()
                new_stack.push_mol(mol, 0)
                success = new_stack.push_rxn(
                    rxn,
                    rxn_idx,
                    product_template=product_template,
                    product_limit=product_limit,
                )
                if success:
                    scores.append(
                        similarity_score(product_template, new_stack[-1].smiles)
                    )
                    new_stacks.append(new_stack)

    new_stacks = [stack for stack in new_stacks if stack is not None and len(stack) > 0]
    if len(new_stacks) == 0:
        return None
    if len(new_stacks) > n_stacks:
        new_stacks = sorted(
            new_stacks, key=lambda x: scores[new_stacks.index(x)], reverse=True
        )[:n_stacks]
    return new_stacks


def reconstruct_all_rxns(
    output: Dict[str, Any],
    reaction_idx_map: Dict[str, int],
    embedding_path: str,
    k: int,
    n_stacks: int,
) -> Optional[List[Stack]]:
    """Reconstruct all reactions from LLM output.

    Adapted from step_31_enamine_reconstruct.py:reconstruct_all_rxns

    Args:
        output: LLM output dict with 'reactions' and 'building_blocks' keys
        reaction_idx_map: Mapping from reaction SMARTS to reaction index
        embedding_path: Path to reaction embeddings directory
        k: Number of similar BBs to search per reactant
        n_stacks: Maximum number of stacks to keep

    Returns:
        List of Stack objects with reconstructed pathways, or None if failed
    """
    if "reactions" not in output or "building_blocks" not in output:
        return None

    building_blocks = [
        bb.split("<bb>")[-1].split("</bb>")[0] for bb in output["building_blocks"]
    ]
    reactions = output["reactions"]
    stacks = [Stack()]

    for i, reaction in enumerate(reactions[::-1]):
        if (
            "reaction_template" not in reaction
            or "reactants" not in reaction
            or "product" not in reaction
        ):
            continue

        template = reaction["reaction_template"].split("<rxn>")[1].split("</rxn>")[0]
        if template not in reaction_idx_map:
            template = find_closest_match(template, list(reaction_idx_map.keys()))

        rxn_idx = reaction_idx_map[template]
        reactants = reaction["reactants"]
        product_template = reaction["product"]
        smiles_to_search = [s for s in reactants if s in building_blocks]

        smiles_searcher = SmilesSimilaritySearch.load(
            f"{embedding_path}/smiles_tfidf_{rxn_idx}.pkl"
        )
        fp_searcher = FingerprintIndex.load(f"{embedding_path}/fpindex_{rxn_idx}.pkl")

        stacks = reconstruct_single_rxn(
            smiles_to_search,
            product_template,
            smiles_searcher,
            fp_searcher,
            template,
            rxn_idx,
            stacks,
            k,
            n_stacks,
        )

        if stacks is None:
            return None

    return stacks


def score_pathways(
    stacks: List[Stack], target_mol: Molecule, num_calc_extra_metrics: int = 10
) -> pd.DataFrame:
    """Score pathways by their similarity to target molecule.

    Adapted from step_31_enamine_reconstruct.py:reaction_scorer

    Args:
        stacks: List of Stack objects with reconstructed pathways
        target_mol: Target molecule
        num_calc_extra_metrics: Number of top pathways to calculate extra metrics for

    Returns:
        DataFrame with scored pathways
    """
    rows = []
    smiles_to_mol = {}

    if not stacks:
        return pd.DataFrame()

    for stack in stacks:
        product_mol = stack[-1]
        rows.append(
            {
                "target": target_mol.smiles,
                "smiles": product_mol.smiles,
                "score": target_mol.sim(
                    product_mol, FingerprintOption.morgan_for_tanimoto_similarity()
                ),
                "synthesis": stack.get_action_string(),
                "num_steps": stack.count_reactions(),
            }
        )
        smiles_to_mol[product_mol.smiles] = product_mol

    rows.sort(key=lambda r: r["score"], reverse=True)

    for row in rows[:num_calc_extra_metrics]:
        mol = smiles_to_mol[str(row["smiles"])]
        row["scf_sim"] = target_mol.scaffold.tanimoto_similarity(
            mol.scaffold,
            fp_option=FingerprintOption.morgan_for_tanimoto_similarity(),
        )
        row["pharm2d_sim"] = target_mol.dice_similarity(
            mol, fp_option=FingerprintOption.gobbi_pharm2d()
        )
        row["rdkit_sim"] = target_mol.tanimoto_similarity(
            mol, fp_option=FingerprintOption.rdkit()
        )

    df = pd.DataFrame(rows)
    return df


def reconstruct_pathways(
    smiles: str,
    llama_outputs: List[Dict[str, Any]],
    reaction_smarts_dict_path: str,
    embedding_path: str,
    k: int = 5,
    n_stacks: int = 25,
    num_calc_extra_metrics: int = 10,
) -> Optional[pd.DataFrame]:
    """Reconstruct pathways for a target molecule from LLM outputs.

    Adapted from step_31_enamine_reconstruct.py:result_generator

    Args:
        smiles: Target molecule SMILES
        llama_outputs: List of LLM output dicts
        reaction_smarts_dict_path: Path to reaction SMARTS mapping file
        embedding_path: Path to reaction embeddings directory
        k: Number of similar BBs to search per reactant
        n_stacks: Maximum number of stacks to keep
        num_calc_extra_metrics: Number of top pathways to calculate extra metrics for

    Returns:
        DataFrame with reconstructed and scored pathways, or None if failed
    """
    reaction_smarts_dict = pickle.load(open(reaction_smarts_dict_path, "rb"))
    reaction_idx_map = {v[0]: k for k, v in reaction_smarts_dict.items()}

    product_mol = Molecule(smiles)
    df_product = pd.DataFrame()

    for i, output in enumerate(llama_outputs):
        try:
            stacks = reconstruct_all_rxns(
                output, reaction_idx_map, embedding_path, k, n_stacks
            )
            if stacks is None:
                continue
            df = score_pathways(stacks, product_mol, num_calc_extra_metrics)
            df_product = pd.concat([df_product, df])
        except Exception as e:
            continue

    if len(df_product) == 0:
        return None

    return (
        df_product.sort_values(
            by=["score", "rdkit_sim", "scf_sim", "pharm2d_sim"],
            ascending=[False, False, False, False],
        )
        .reset_index(drop=True)
        .iloc[:n_stacks]
    )


# ============================================================================
# Raw Output Filtering Functions
# Adapted from step_30_0_benchmark_filter_raw_output.py
# ============================================================================


def arrange_reactants_and_react(
    template: str, reactant_mols: List[Molecule]
) -> Tuple[Optional[List], bool]:
    """Try all reactant permutations with a reaction template.

    Adapted from step_30_0_benchmark_filter_raw_output.py:arrange_reactants_and_react_synllama

    Args:
        template: Reaction SMARTS template
        reactant_mols: List of reactant Molecule objects

    Returns:
        Tuple of (product_list, matched_bool)
    """
    rxn = Reaction(template)
    if len(reactant_mols) != rxn.num_reactants:
        return None, False

    if len(reactant_mols) == 1:
        product = rxn(reactant_mols)
        if len(product) == 0:
            return None, False
    elif len(reactant_mols) == 2:
        product = []
        product.extend(rxn([reactant_mols[0], reactant_mols[1]]))
        product.extend(rxn([reactant_mols[1], reactant_mols[0]]))
        if len(product) == 0:
            return None, False
    elif len(reactant_mols) == 3:
        product = []
        product.extend(rxn([reactant_mols[0], reactant_mols[1], reactant_mols[2]]))
        product.extend(rxn([reactant_mols[0], reactant_mols[2], reactant_mols[1]]))
        product.extend(rxn([reactant_mols[1], reactant_mols[0], reactant_mols[2]]))
        product.extend(rxn([reactant_mols[1], reactant_mols[2], reactant_mols[0]]))
        product.extend(rxn([reactant_mols[2], reactant_mols[0], reactant_mols[1]]))
        product.extend(rxn([reactant_mols[2], reactant_mols[1], reactant_mols[0]]))
        if len(product) == 0:
            return None, False
    else:
        return None, False

    return product, True


def filter_raw_output(
    llama_output: Dict[str, Any], reaction_idx_map: Dict[str, int]
) -> Dict[str, List[Dict[str, Any]]]:
    """Filter raw LLM output to extract valid synthesis pathways.

    Adapted from step_30_0_benchmark_filter_raw_output.py:filter_raw_output

    Args:
        llama_output: Dict mapping product SMILES to list of LLM outputs
        reaction_idx_map: Mapping from reaction SMARTS to reaction index

    Returns:
        Dict mapping product SMILES to list of valid pathway dicts with keys:
            - reaction_strings: Semicolon-separated synthesis string
            - bbs: List of building block SMILES
    """
    successful_synthesis = defaultdict(list)

    for product_smiles, example_data in llama_output.items():
        if type(example_data) == str:
            continue

        for output in example_data:
            if type(output) == str:
                continue

            try:
                assert "reactions" in output and "building_blocks" in output
                reactions = output["reactions"]
                building_blocks = output["building_blocks"]
                reactant_stack = []
                reaction_strings = []
                reactant_stack.append(product_smiles)
                reaction_strings.append(product_smiles)

                for reaction in reactions:
                    assert (
                        "reaction_template" in reaction
                        and "reactants" in reaction
                        and "product" in reaction
                    )
                    product = reaction["product"]
                    assert product in reactant_stack
                    reactant_stack.remove(product)
                    reaction_strings.remove(product)
                    reaction_strings.append(product)
                    template = (
                        reaction["reaction_template"]
                        .split("<rxn>")[1]
                        .split("</rxn>")[0]
                    )
                    assert template in reaction_idx_map
                    reaction_strings.append(f"R{reaction_idx_map[template]}")
                    reactants = reaction["reactants"]
                    reactants = [
                        reactant.split("<bb>")[-1].split("</bb>")[0]
                        if "<bb>" in reactant
                        else reactant
                        for reactant in reactants
                    ]
                    reactant_stack.extend(reactants)
                    reactant_mols = []
                    for reactant in reactants:
                        if reactant == "":
                            continue
                        mol = Molecule(reactant, source="smiles")
                        if not mol.is_valid:
                            raise ValueError(f"Invalid molecule {reactant}")
                        reactant_mols.append(mol)
                    product_mol = Molecule(product, source="smiles")
                    if not product_mol.is_valid:
                        raise ValueError(f"Invalid molecule {product}")
                    product_from_rxn, matched = arrange_reactants_and_react(
                        template, reactant_mols
                    )
                    assert matched
                    product_from_rxn = [
                        prod.csmiles for prod in product_from_rxn if prod is not None
                    ]
                    assert product_mol.csmiles in product_from_rxn

                bbs = []
                for bb in building_blocks:
                    bb_clean = bb.split("<bb>")[-1].split("</bb>")[0]
                    assert bb_clean in reactant_stack
                    reactant_stack.remove(bb_clean)
                    bbs.append(bb_clean)

                successful_synthesis[product_smiles].append(
                    {
                        "reaction_strings": ";".join(reaction_strings[::-1]),
                        "bbs": bbs,
                    }
                )

            except Exception as e:
                continue

    return successful_synthesis


def check_bbs_in_custom_list(
    bbs: List[str], custom_bb_index: FingerprintIndex, similarity_threshold: float = 1.0
) -> Dict[str, float]:
    """Check which BBs are in custom BB list using fingerprint similarity.

    Adapted from step_30_0_benchmark_filter_raw_output.py:check_bbs_in_enamine_parallel

    Args:
        bbs: List of BB SMILES to check
        custom_bb_index: FingerprintIndex of custom BBs
        similarity_threshold: Threshold for considering a BB as "in list" (default: 1.0 = exact match)

    Returns:
        Dict mapping BB SMILES to max similarity score with custom BB list
    """
    bb_mols = [Molecule(bb, source="smiles") for bb in bbs]
    fingerprints = compute_fingerprints(
        bb_mols, FingerprintOption.morgan_for_building_blocks(), batch_size=1024
    )
    fp_searched_results = custom_bb_index.query(fingerprints, k=10)
    bbs_similarity = []

    for bb, result in zip(bbs, fp_searched_results):
        bbs_similarity.append(
            np.max(
                [
                    Molecule(bb).sim(
                        r.molecule, FingerprintOption.morgan_for_tanimoto_similarity()
                    )
                    for r in result
                ]
            )
        )

    bb_similarity_dict = {bb: similarity for bb, similarity in zip(bbs, bbs_similarity)}
    return bb_similarity_dict


def analyze_bb_coverage(
    successful_synthesis: Dict[str, List[Dict[str, Any]]],
    custom_bb_index: FingerprintIndex,
    similarity_threshold: float = 1.0,
) -> Dict[str, List[Dict[str, Any]]]:
    """Analyze BB coverage against custom BB list.

    Args:
        successful_synthesis: Output from filter_raw_output
        custom_bb_index: FingerprintIndex of custom BBs
        similarity_threshold: Threshold for considering a BB as "in list"

    Returns:
        Updated successful_synthesis dict with added fields:
            - bbs_similarity: List of similarity scores for each BB
            - bbs_not_in_list: BBs below similarity threshold
            - bbs_in_list: BBs at or above similarity threshold
    """
    # Collect all BBs
    all_bbs = []
    for _, value in successful_synthesis.items():
        for v in value:
            all_bbs.extend(v["bbs"])
    all_bbs = list(set(all_bbs))

    # Check similarity
    bb_similarity_dict = check_bbs_in_custom_list(
        all_bbs, custom_bb_index, similarity_threshold
    )

    # Update with coverage info
    for _, value in successful_synthesis.items():
        for v in value:
            v["bbs_similarity"] = [bb_similarity_dict[bb] for bb in v["bbs"]]
            v["bbs_not_in_list"] = [
                bb
                for bb, sim in zip(v["bbs"], v["bbs_similarity"])
                if sim < similarity_threshold
            ]
            v["bbs_in_list"] = [
                bb
                for bb, sim in zip(v["bbs"], v["bbs_similarity"])
                if sim >= similarity_threshold
            ]

    return successful_synthesis
