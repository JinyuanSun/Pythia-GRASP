# Protein Backbone Energy Minimization

This module implements protein backbone structure energy minimization using Pythia's likelihood scoring mechanism.

## Overview

The energy minimization algorithm works as follows:

1. **Load Initial Structure**: Read the input PDB file and extract backbone atom coordinates
2. **Apply Perturbations**: Apply small random translations to backbone atoms of each residue independently
3. **Score Structures**: Use Pythia's likelihood scoring to evaluate the quality of perturbed structures
4. **Iterative Optimization**: Accept perturbations that improve the score (or occasionally accept worse solutions to escape local minima)
5. **Convergence**: Continue until convergence or maximum iterations reached
6. **RMSD Constraint**: Ensure the optimized structure doesn't deviate too much from the initial structure

## Usage

### Basic Usage

```bash
cd /path/to/Pythia-GRASP
PYTHONPATH=. python3 pythia/energy_minimize.py --pdb_file examples/1pga.pdb --device cpu
```

### Advanced Options

```bash
python3 pythia/energy_minimize.py \
  --pdb_file examples/1pga.pdb \
  --output examples/1pga_optimized.pdb \
  --device cpu \
  --max_iterations 200 \
  --translation_std 0.05 \
  --max_rmsd 1.5 \
  --convergence_threshold 0.0001
```

### Parameters

- `--pdb_file`: Path to input PDB file (required)
- `--output`: Path to output PDB file (default: `<input>_minimized.pdb`)
- `--device`: Device for computation, 'cpu' or 'cuda' (default: 'cpu')
- `--max_iterations`: Maximum number of optimization iterations (default: 100)
- `--translation_std`: Standard deviation for translation noise in Angstroms (default: 0.1)
- `--rotation_std`: Standard deviation for rotation noise in radians (default: 0.05)
- `--max_rmsd`: Maximum allowed RMSD from initial structure in Angstroms (default: 2.0)
- `--convergence_threshold`: Threshold for score improvement to declare convergence (default: 1e-4)

## Algorithm Details

### Perturbation Strategy

The algorithm uses **per-residue perturbations** rather than global transformations. This is crucial because:

- Pythia's score is based on local geometry and sequence context
- Global rotations/translations don't change the internal structure
- Per-residue perturbations can improve local conformations

### Acceptance Criterion

The algorithm uses a modified Metropolis criterion:

1. Always accept if the score improves (greedy acceptance)
2. Occasionally accept worse solutions (10% probability) to escape local minima
3. Track the best solution seen so far

### Scoring

The score is computed as the negative log-likelihood of the native sequence given the structure:

```
score = -mean(log P(native_seq | structure))
```

Lower scores indicate better structures. The scoring uses two Pythia models (pythia-c.pt and pythia-p.pt) and averages their predictions.

## Example Results

For the example protein 1pga.pdb with 200 iterations:

```
Initial score: 1.8042
Final score: 1.6144
Score improvement: 0.1899 (10.5% improvement)
Final RMSD: 0.586 Å
Accepted moves: 71/200
Improved moves: 53/200
```

## Tips for Best Results

1. **Start with moderate perturbations**: `translation_std=0.05` works well
2. **Use sufficient iterations**: At least 100-200 iterations for good results
3. **Adjust RMSD constraint**: Based on how much structural deviation is acceptable
4. **Monitor convergence**: The algorithm will report when it converges
5. **Use GPU if available**: Set `--device cuda` for faster scoring

## Limitations

- The algorithm performs local optimization; it won't fix large-scale structural errors
- Quality of results depends on the initial structure
- Pythia's scoring may not capture all aspects of protein stability
- The RMSD constraint limits how much the structure can be improved

## Integration with Other Tools

The minimized structures can be used as input to:

- Structure prediction validation
- Protein design workflows  
- Molecular dynamics simulations
- Docking studies

## Implementation Notes

The implementation is located in `pythia/energy_minimize.py` and uses:

- `pythia.pdb_utils` for PDB I/O and structure representation
- `pythia.masked_ddg_scan` for loading Pythia models
- BioPython for PDB manipulation
- PyTorch for neural network inference
