## 🔧 API Guide

This guide describes the Python API for SynLlama synthesis planning. The API provides a programmatic interface to the functionality described in the [inference_guide.md](inference_guide.md), with two planning modes:

1. **Raw Generation** - Direct LLM output (corresponds to parallel_inference.py)
2. **BB Reconstruction** - Reconstruction with custom building blocks (corresponds to step_31_enamine_reconstruct.py)

### Mode 1: Raw Generation

This mode provides direct access to LLM generation, equivalent to running `synllama/llm/parallel_inference.py` but for single molecules programmatically.

```python
from synllama.api import plan_synthesis

result = plan_synthesis(
    smiles="CCO",
    model_path="synllama-data/inference/model/SynLlama-1B-2M-91rxns",
    sample_mode="greedy",
    max_length=1600,
    device="cuda"
)

if result['success']:
    print(f"Synthesis: {result['synthesis']}")
    print(f"Steps: {result['num_steps']}")
else:
    print(f"Error: {result['error']}")
```

**Parameters:**
- `smiles` (str): Target molecule SMILES
- `model_path` (str): Path to SynLlama model directory
- `sample_mode` (str): Sampling mode - "greedy", "frugal", "frozen_only", "low_only", "medium_only", or "high_only"
- `max_length` (int): Maximum tokens to generate (default: 1600)
- `device` (str, optional): "cuda", "cpu", or None for auto-detect

**Returns:**
- `target` (str): Input SMILES
- `synthesis` (str): Synthesis string in format `BB1;BB2;R{idx};Intermediate;...`
- `num_steps` (int): Number of reaction steps
- `success` (bool): Whether valid pathway found
- `error` (str | None): Error message if failed

**Key Differences from parallel_inference.py:**
- Single molecule (not batch processing)
- Returns dict (not .pkl file)
- In-memory processing (no file I/O)

### Mode 2: BB Reconstruction with Enamine or Custom Building Blocks

This mode provides the reconstruction algorithm from `steps/step_31_enamine_reconstruct.py`. By default it uses the Enamine BB database, but you can also provide custom building blocks.

#### Option A: Using Enamine Building Blocks (Default)

```python
from synllama.api import plan_synthesis_with_reconstruction

result = plan_synthesis_with_reconstruction(
    smiles="CCO",
    model_path="synllama-data/inference/model/SynLlama-1B-2M-91rxns",
    rxn_embedding_path="synllama-data/inference/reconstruction/91rxns/rxn_embeddings",
    sample_mode="greedy",
    top_n=5,
    k=5,
    n_stacks=25,
    max_length=1600,
    device="cuda"
)

if result['success']:
    for pathway in result['pathways']:
        print(f"Product: {pathway['product_smiles']}")
        print(f"Similarity: {pathway['similarity_score']:.3f}")
        print(f"Synthesis: {pathway['synthesis']}")
```

#### Option B: Using Custom Building Blocks

Custom building blocks require a **two-step workflow**:

**Step 1: Build Custom BB Indices (One-Time Setup)**

```python
from synllama.llm.build_custom_indices import build_custom_bb_indices

# Load your custom building blocks (one SMILES per line)
with open("my_building_blocks.smi", "r") as f:
    custom_bbs = [line.strip() for line in f if line.strip()]

# Or define them directly
custom_bbs = ["CC", "C=O", "CCN", "c1ccccc1", "CCO", "c1cccnc1"]

# Build indices for your custom BBs (this takes a few minutes)
custom_index_path = build_custom_bb_indices(
    custom_bbs=custom_bbs,
    reaction_smarts_dict_path="synllama-data/inference/reconstruction/91rxns/rxn_embeddings/reaction_smarts_map.pkl",
    output_dir="my_custom_bb_indices",
    token_list_path="data/smiles_vocab.txt"  # Optional, uses default if not provided
)

print(f"Custom BB indices built at: {custom_index_path}")
```

**Step 2: Run Reconstruction with Custom BBs**

```python
from synllama.api import plan_synthesis_with_reconstruction

# Use the custom BB indices you just built
result = plan_synthesis_with_reconstruction(
    smiles="CCO",
    model_path="synllama-data/inference/model/SynLlama-1B-2M-91rxns",
    rxn_embedding_path="synllama-data/inference/reconstruction/91rxns/rxn_embeddings",
    custom_bb_index_path="my_custom_bb_indices",  # Point to your custom indices
    sample_mode="greedy",
    top_n=5,
    k=5,
    n_stacks=25,
    device="cuda"
)

if result['success']:
    for pathway in result['pathways']:
        print(f"Product: {pathway['product_smiles']}")
        print(f"Similarity: {pathway['similarity_score']:.3f}")
        print(f"Synthesis: {pathway['synthesis']}")
```

**Parameters:**
- `smiles` (str): Target molecule SMILES
- `model_path` (str): Path to SynLlama model directory
- `rxn_embedding_path` (str): Path to Enamine reaction embeddings directory (used for reaction templates even with custom BBs)
- `sample_mode` (str): Sampling mode (default: "greedy")
- `top_n` (int): Number of top pathways to return (default: 5)
- `k` (int): Number of similar BBs to search per reactant (default: 5)
- `n_stacks` (int): Maximum stacks during reconstruction (default: 25)
- `max_length` (int): Maximum tokens to generate (default: 1600)
- `device` (str, optional): "cuda", "cpu", or None
- `custom_bb_index_path` (str, optional): Path to custom BB indices directory (built with `build_custom_bb_indices`)

**Returns:**
- `target` (str): Input SMILES
- `mode` (str): "reconstruction"
- `pathways` (List[Dict]): List of pathway dicts, each containing:
  - `synthesis` (str): Synthesis string
  - `num_steps` (int): Number of reaction steps
  - `similarity_score` (float): Morgan Tanimoto similarity to target (0-1)
  - `product_smiles` (str): Actual synthesized molecule SMILES
  - `scf_sim` (float, optional): Scaffold similarity
  - `pharm2d_sim` (float, optional): Pharmacophore similarity
  - `rdkit_sim` (float, optional): RDKit fingerprint similarity
- `success` (bool): Whether any valid pathway found
- `error` (str | None): Error message if failed

**Important Notes for Custom BBs:**
- Index building is a **one-time operation** per BB set - the indices can be reused for multiple molecules
- Index building can take several minutes for large BB sets (progress is shown)
- The custom BB set must match the **reaction set** (91rxns vs 115rxns) of your model
- Some reactions may have few or no valid BBs from your custom set - this is normal
- Custom BB indices require ~100-500MB disk space depending on BB set size
- The builder **automatically filters** BBs per-reaction to include only valid reactants for each reaction template

**How BB Filtering Works:**

The `build_custom_bb_indices` function tests each custom BB against each reaction template to determine which BBs are valid reactants. This mirrors the Enamine workflow where only relevant BBs are indexed per reaction. For example:
- A BB like "CC" (ethane) might be valid for alkylation reactions but not for aromatic substitutions
- The indices for each reaction only contain BBs that can actually participate in that reaction
- During reconstruction, the algorithm searches only the pre-filtered BB set for each reaction step

### Parameter Tuning

As mentioned in inference_guide.md, for **synthesis planning**, use default parameters:
```python
k=5, n_stacks=25
```

For **analog generation** or **hit expansion**, increase exploration:
```python
k=10, n_stacks=50
```

### Reaction Set Selection

Match the model and embeddings (both must use same reaction set):

**91 Reactions:**
```python
model_path = "synllama-data/inference/model/SynLlama-1B-2M-91rxns"
rxn_embedding_path = "synllama-data/inference/reconstruction/91rxns/rxn_embeddings"
```

**115 Reactions:**
```python
model_path = "synllama-data/inference/model/SynLlama-1B-2M-115rxns"
rxn_embedding_path = "synllama-data/inference/reconstruction/115rxns/rxn_embeddings"
```

### Building Custom BB Indices from Command Line

You can also build custom BB indices from the command line:

```bash
python -m synllama.llm.build_custom_indices \
    --custom_bbs_file my_building_blocks.smi \
    --reaction_smarts_dict_path synllama-data/inference/reconstruction/91rxns/rxn_embeddings/reaction_smarts_map.pkl \
    --output_dir my_custom_bb_indices \
    --token_list_path data/smiles_vocab.txt
```

The output directory can then be used with `custom_bb_index_path` parameter in the API.

### Batch Processing with Custom BBs

Once you've built custom BB indices, you can reuse them across multiple molecules:

```python
from synllama.llm.build_custom_indices import build_custom_bb_indices
from synllama.api import plan_synthesis_with_reconstruction

# Step 1: Build indices once
custom_bbs = ["CC", "C=O", "CCN", "c1ccccc1", "CCO"]
custom_index_path = build_custom_bb_indices(
    custom_bbs=custom_bbs,
    reaction_smarts_dict_path="synllama-data/inference/reconstruction/91rxns/rxn_embeddings/reaction_smarts_map.pkl",
    output_dir="my_custom_indices"
)

# Step 2: Process multiple targets using the same indices
targets = ["CCO", "c1ccccc1O", "CCN"]
results = []

for smiles in targets:
    result = plan_synthesis_with_reconstruction(
        smiles=smiles,
        model_path="synllama-data/inference/model/SynLlama-1B-2M-91rxns",
        rxn_embedding_path="synllama-data/inference/reconstruction/91rxns/rxn_embeddings",
        custom_bb_index_path=custom_index_path,  # Reuse the same indices
        top_n=5
    )
    results.append(result)
```

### Understanding Synthesis String Format

Both modes return synthesis strings in the format:
```
BB1;BB2;R{idx};Intermediate;BB3;R{idx};Final
```

Example:
```
CC;C=O;R5;CC=O;CCN;R12;CC(=O)N(C)C
```

This represents:
1. React BB1 (`CC`) + BB2 (`C=O`) using reaction R5 → intermediate `CC=O`
2. React intermediate + BB3 (`CCN`) using reaction R12 → final product
