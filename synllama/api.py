"""SynLlama API v1 - Raw synthesis planning from LLM output."""

from typing import Dict, Any, Optional
from synllama.llm.inference import (
    load_model,
    generate_single_smiles,
    get_sample_params,
)


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
