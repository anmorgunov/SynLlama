"""Clean LLM inference API without CLI dependencies."""

import torch
import json
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from synllama.llm.vars import (
    TEMPLATE,
    sampling_params_greedy,
    sampling_params_frugal,
    sampling_params_frozen_only,
    sampling_params_low_only,
    sampling_params_medium_only,
    sampling_params_high_only,
)


def generate_single_smiles(
    smiles: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    sample_params: List[Dict[str, Any]],
    max_length: int = 1600,
) -> List[Dict[str, Any]]:
    """Generate synthesis pathways for a single SMILES string.

    Args:
        smiles: Target molecule SMILES
        tokenizer: Loaded tokenizer
        model: Loaded model
        sample_params: List of sampling parameter dicts with keys:
            - temp: temperature
            - top_p: top-p value
            - repeat: number of times to repeat
        max_length: Maximum number of new tokens to generate

    Returns:
        List of parsed JSON pathways
    """
    instruction = TEMPLATE["instruction"]
    input_template = TEMPLATE["input"]

    input_str = input_template.replace("SMILES_STRING", smiles)
    prompt_complete = (
        "### Instruction:\n"
        + instruction
        + "\n\n### Input:\n"
        + input_str
        + "\n\n### Response: \n"
    )

    inputs = tokenizer(prompt_complete, return_tensors="pt").to(model.device)
    prompt_length = inputs.input_ids.shape[1]

    stopping_ids = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    generated_texts = []

    for params in sample_params:
        temp = params["temp"]
        top_p = params["top_p"]
        repeat = params["repeat"]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=temp,
                top_p=top_p,
                num_return_sequences=repeat,
                eos_token_id=stopping_ids,
                pad_token_id=tokenizer.eos_token_id,
            )

        for output in outputs:
            generated_text = tokenizer.decode(
                output[prompt_length:], skip_special_tokens=True
            )
            generated_texts.append(generated_text.strip())

    # Parse JSON responses
    pathways = []
    for text in generated_texts:
        try:
            pathway = json.loads(text)
            if "reactions" in pathway and "building_blocks" in pathway:
                pathways.append(pathway)
        except json.JSONDecodeError:
            continue

    return pathways


def load_model(model_path: str, device: str = None):
    """Load model and tokenizer.

    Args:
        model_path: Path to model directory
        device: Device to load model on (auto-detected if None)

    Returns:
        Tuple of (tokenizer, model)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map={"": device} if device == "cuda" else None,
    )

    if device == "cpu":
        model = model.to(device)

    model.eval()
    return tokenizer, model


def get_sample_params(mode: str) -> List[Dict[str, Any]]:
    """Get sampling parameters for a given mode.

    Args:
        mode: Sampling mode ("greedy", "frugal", etc.)

    Returns:
        List of sampling parameter dictionaries
    """
    mode_map = {
        "greedy": sampling_params_greedy,
        "frugal": sampling_params_frugal,
        "frozen_only": sampling_params_frozen_only,
        "low_only": sampling_params_low_only,
        "medium_only": sampling_params_medium_only,
        "high_only": sampling_params_high_only,
    }

    if mode not in mode_map:
        raise ValueError(
            f"Unknown sample mode: {mode}. Choose from: {list(mode_map.keys())}"
        )

    return mode_map[mode]
