"""SynLlama API v1 - Raw synthesis planning from LLM output.

This module provides two planning modes:
1. Raw generation (plan_synthesis) - Direct LLM output without BB reconstruction
2. BB reconstruction (plan_synthesis_with_reconstruction) - Reconstruct pathways using custom BBs

Examples:
    # Mode 1: Raw generation
    result = plan_synthesis(
        smiles="CCO",
        model_path="path/to/model",
        sample_mode="greedy"
    )

    # Mode 2: BB Reconstruction
    result = plan_synthesis_with_reconstruction(
        smiles="CCO",
        model_path="path/to/model",
        custom_bbs=["CC", "C=O", "CCN"],
        rxn_embedding_path="path/to/reconstruction/91rxns/rxn_embeddings",
        top_n=5
    )
"""

from typing import Dict, Any, Optional, List
import os
import pandas as pd

from synllama.llm.inference import (
    load_model,
    generate_single_smiles,
    get_sample_params,
)
from synllama.llm.reconstruction import (
    reconstruct_pathways,
)
from synllama.chem.mol import Molecule


def _pathway_to_synthesis(pathway: Dict[str, Any]) -> str:
    """Convert a pathway dict to synthesis string format.

    Format: BB1;BB2;R{num};Intermediate;BB3;R{num};Final

    Args:
        pathway: Dict with 'reactions' and 'building_blocks' keys

    Returns:
        Synthesis string
    """
    if "reactions" not in pathway or "building_blocks" not in pathway:
        return ""

    reactions = pathway["reactions"]
    building_blocks = pathway["building_blocks"]

    if not reactions:
        return ""

    # Extract clean BBs from tags
    bb_set = set()
    for bb in building_blocks:
        if "<bb>" in bb and "</bb>" in bb:
            bb_clean = bb.split("<bb>")[-1].split("</bb>")[0]
            bb_set.add(bb_clean)

    # Build synthesis string
    parts = []

    # Reactions are in reverse order (last step first), so reverse them
    for i, reaction in enumerate(reversed(reactions)):
        if "reactants" not in reaction or "product" not in reaction:
            continue

        reactants = reaction["reactants"]
        product = reaction["product"]

        # Add reactants that are building blocks
        for reactant in reactants:
            if reactant in bb_set:
                parts.append(reactant)

        # Add reaction template
        parts.append(f"R{i}")

        # Add product
        parts.append(product)

    return ";".join(parts)


def _count_steps(pathway: Dict[str, Any]) -> int:
    """Count number of synthesis steps in a pathway.

    Args:
        pathway: Dict with 'reactions' key

    Returns:
        Number of reaction steps
    """
    if "reactions" not in pathway:
        return 0
    return len(pathway["reactions"])


def plan_synthesis(
    smiles: str,
    model_path: str,
    sample_mode: str = "greedy",
    max_length: int = 1600,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Plan synthesis for a target molecule using SynLlama.

    Args:
        smiles: Target molecule SMILES string
        model_path: Path to the SynLlama model directory
        sample_mode: Sampling mode - "greedy", "frugal", "frozen_only",
                     "low_only", "medium_only", or "high_only"
        max_length: Maximum number of tokens to generate
        device: Device to use ("cuda", "cpu", or None for auto)

    Returns:
        Dictionary with keys:
            - target: Input SMILES
            - synthesis: Synthesis string (BB1;BB2;R1;Intermediate;...)
            - num_steps: Number of reaction steps
            - success: Whether a valid pathway was found
            - error: Error message if failed
    """
    result = {
        "target": smiles,
        "synthesis": None,
        "num_steps": 0,
        "success": False,
        "error": None,
    }

    try:
        # Load model
        tokenizer, model = load_model(model_path, device=device)

        # Get sampling parameters
        sample_params = get_sample_params(sample_mode)

        # Generate pathways
        pathways = generate_single_smiles(
            smiles=smiles,
            tokenizer=tokenizer,
            model=model,
            sample_params=sample_params,
            max_length=max_length,
        )

        if not pathways:
            result["error"] = "No valid synthesis pathways generated"
            return result

        # Use first valid pathway
        pathway = pathways[0]

        # Convert to synthesis string
        synthesis = _pathway_to_synthesis(pathway)

        if not synthesis:
            result["error"] = "Could not convert pathway to synthesis string"
            return result

        # Count steps
        num_steps = _count_steps(pathway)

        result["synthesis"] = synthesis
        result["num_steps"] = num_steps
        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


def plan_synthesis_with_reconstruction(
    smiles: str,
    model_path: str,
    rxn_embedding_path: str,
    sample_mode: str = "greedy",
    top_n: int = 5,
    k: int = 5,
    n_stacks: int = 25,
    max_length: int = 1600,
    device: Optional[str] = None,
    custom_bb_index_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Plan synthesis with BB reconstruction using Enamine or custom building blocks.

    This function combines LLM generation with the reconstruction algorithm to find
    synthesizable pathways. It follows the workflow from the inference guide:

    1. Generate raw pathways using LLM (like plan_synthesis)
    2. Reconstruct pathways using BBs (like step_31_enamine_reconstruct.py)
    3. Score and rank pathways by similarity to target

    Args:
        smiles: Target molecule SMILES string
        model_path: Path to the SynLlama model directory
        rxn_embedding_path: Path to reaction embeddings directory with Enamine BBs
                           (e.g., "synllama-data/inference/reconstruction/91rxns/rxn_embeddings")
        sample_mode: Sampling mode - "greedy", "frugal", "frozen_only",
                     "low_only", "medium_only", or "high_only"
        top_n: Number of top pathways to return (default: 5)
        k: Number of similar BBs to search per reactant (default: 5)
        n_stacks: Maximum number of stacks to keep during reconstruction (default: 25)
        max_length: Maximum number of tokens to generate
        device: Device to use ("cuda", "cpu", or None for auto)
        custom_bb_index_path: Optional path to custom BB indices directory. If provided,
                             uses custom BBs instead of Enamine. Must contain files built
                             by build_custom_bb_indices() from synllama.llm.build_custom_indices

    Returns:
        Dictionary with keys:
            - target: Input SMILES
            - mode: "reconstruction"
            - pathways: List of pathway dicts with keys:
                - synthesis: Synthesis string (BB1;BB2;R1;Product;...)
                - num_steps: Number of reaction steps
                - similarity_score: Morgan Tanimoto similarity to target (0-1)
                - product_smiles: Actual synthesized molecule SMILES
                - scf_sim: Scaffold similarity (calculated for top pathways)
                - pharm2d_sim: Pharmacophore similarity (calculated for top pathways)
                - rdkit_sim: RDKit fingerprint similarity (calculated for top pathways)
            - success: Whether any valid pathway was found
            - error: Error message if failed

    Example with Enamine BBs:
        >>> result = plan_synthesis_with_reconstruction(
        ...     smiles="CCO",
        ...     model_path="path/to/SynLlama-1B-2M-91rxns",
        ...     rxn_embedding_path="path/to/reconstruction/91rxns/rxn_embeddings",
        ...     top_n=5
        ... )

    Example with custom BBs (requires pre-built indices):
        >>> from synllama.llm.build_custom_indices import build_custom_bb_indices
        >>> custom_bbs = ["CC", "C=O", "CCN", "c1ccccc1"]
        >>> custom_index_path = build_custom_bb_indices(
        ...     custom_bbs=custom_bbs,
        ...     reaction_smarts_dict_path="path/to/rxn_embeddings/reaction_smarts_map.pkl",
        ...     output_dir="my_custom_indices"
        ... )
        >>> result = plan_synthesis_with_reconstruction(
        ...     smiles="CCO",
        ...     model_path="path/to/SynLlama-1B-2M-91rxns",
        ...     rxn_embedding_path="path/to/reconstruction/91rxns/rxn_embeddings",
        ...     custom_bb_index_path=custom_index_path,
        ...     top_n=5
        ... )
    """
    result = {
        "target": smiles,
        "mode": "reconstruction",
        "pathways": [],
        "success": False,
        "error": None,
    }

    try:
        # Determine which embedding path to use
        if custom_bb_index_path is not None:
            # Use custom BB indices
            embedding_path_to_use = custom_bb_index_path

            if not os.path.exists(custom_bb_index_path):
                result["error"] = (
                    f"custom_bb_index_path does not exist: {custom_bb_index_path}"
                )
                return result

            # Validate custom index structure
            reaction_smarts_path = os.path.join(
                custom_bb_index_path, "reaction_smarts_map.pkl"
            )
            if not os.path.exists(reaction_smarts_path):
                result["error"] = (
                    f"Invalid custom BB index: reaction_smarts_map.pkl not found in {custom_bb_index_path}. "
                    "Use build_custom_bb_indices() to create valid indices."
                )
                return result
        else:
            # Use Enamine BB indices (default)
            embedding_path_to_use = rxn_embedding_path

            if not os.path.exists(rxn_embedding_path):
                result["error"] = (
                    f"rxn_embedding_path does not exist: {rxn_embedding_path}"
                )
                return result

            # Check for required files
            reaction_smarts_path = os.path.join(
                rxn_embedding_path, "reaction_smarts_map.pkl"
            )
            if not os.path.exists(reaction_smarts_path):
                result["error"] = (
                    f"reaction_smarts_map.pkl not found in {rxn_embedding_path}"
                )
                return result

        # Validate target molecule
        target_mol = Molecule(smiles, source="smiles")
        if not target_mol.is_valid:
            result["error"] = f"Invalid target SMILES: {smiles}"
            return result

        # Load model
        tokenizer, model = load_model(model_path, device=device)

        # Get sampling parameters
        sample_params = get_sample_params(sample_mode)

        # Generate raw pathways from LLM
        llama_outputs = generate_single_smiles(
            smiles=smiles,
            tokenizer=tokenizer,
            model=model,
            sample_params=sample_params,
            max_length=max_length,
        )

        if not llama_outputs:
            result["error"] = "No valid pathways generated by LLM"
            return result

        # Reconstruct pathways using BBs (Enamine or custom)
        df_pathways = reconstruct_pathways(
            smiles=smiles,
            llama_outputs=llama_outputs,
            reaction_smarts_dict_path=reaction_smarts_path,
            embedding_path=embedding_path_to_use,
            k=k,
            n_stacks=n_stacks,
            num_calc_extra_metrics=min(
                top_n, 10
            ),  # Calculate extra metrics for top pathways
        )

        if df_pathways is None or len(df_pathways) == 0:
            result["error"] = "No valid pathways found during reconstruction"
            return result

        # Convert DataFrame to list of pathway dicts
        pathways = []
        for idx, row in df_pathways.head(top_n).iterrows():
            pathway = {
                "synthesis": row["synthesis"],
                "num_steps": int(row["num_steps"]),
                "similarity_score": float(row["score"]),
                "product_smiles": row["smiles"],
            }

            # Add extra metrics if available
            if "scf_sim" in row and not pd.isna(row["scf_sim"]):
                pathway["scf_sim"] = float(row["scf_sim"])
            if "pharm2d_sim" in row and not pd.isna(row["pharm2d_sim"]):
                pathway["pharm2d_sim"] = float(row["pharm2d_sim"])
            if "rdkit_sim" in row and not pd.isna(row["rdkit_sim"]):
                pathway["rdkit_sim"] = float(row["rdkit_sim"])

            pathways.append(pathway)

        result["pathways"] = pathways
        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    return result
