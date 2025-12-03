"""
Energy minimization of protein backbone structures using Pythia likelihood scoring.

This module implements an iterative optimization procedure that:
1. Applies small translation and rotation perturbations to backbone atoms
2. Scores perturbed structures using Pythia's likelihood scoring
3. Iterates until convergence while maintaining RMSD constraint
"""

import os
import argparse
import numpy as np
import torch
from typing import List, Tuple, Optional
from copy import deepcopy

from pythia.masked_ddg_scan import get_torch_model, pythia_root_dpath
from pythia.pdb_utils import get_neighbor, read_pdb_to_protbb, ProtBB
from Bio.PDB import PDBIO


def compute_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """
    Compute RMSD between two sets of coordinates.
    
    Args:
        coords1: First set of coordinates (N, 3)
        coords2: Second set of coordinates (N, 3)
    
    Returns:
        RMSD value
    """
    diff = coords1 - coords2
    return np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))


def apply_perturbation(
    protbb: ProtBB,
    translation_std: float = 0.1,
    rotation_std: float = 0.05,
    per_residue: bool = True,
) -> ProtBB:
    """
    Apply small random translation and rotation perturbations to backbone atoms.
    
    Args:
        protbb: Protein backbone structure
        translation_std: Standard deviation for translation noise (Angstroms)
        rotation_std: Standard deviation for rotation noise (radians)
        per_residue: If True, apply perturbations per-residue; otherwise globally
    
    Returns:
        Perturbed protein backbone structure
    """
    # Create a copy of the structure
    perturbed_protbb = deepcopy(protbb)
    
    if per_residue:
        # Apply per-residue perturbations to each residue independently
        num_residues = len(protbb.ca)
        
        for i in range(num_residues):
            # Generate random translation for this residue
            translation = np.random.normal(0, translation_std, size=3)
            
            # Apply translation to all atoms of this residue
            for attr in ['ca', 'cb', 'c', 'n', 'o']:
                coords = getattr(perturbed_protbb, attr)
                coords_np = coords[i].squeeze().numpy()
                coords_np = coords_np + translation
                coords[i] = torch.tensor(coords_np).unsqueeze(0).float()
    else:
        # Apply global perturbation (original implementation)
        # Generate random translation
        translation = np.random.normal(0, translation_std, size=3)
        
        # Generate random rotation (small angle approximation)
        rotation_angles = np.random.normal(0, rotation_std, size=3)
        
        # Create rotation matrix using Rodrigues' formula for small angles
        # For small angles, we can use the approximation R ≈ I + [w]_x
        # where [w]_x is the skew-symmetric matrix
        wx, wy, wz = rotation_angles
        rotation_matrix = np.array([
            [1, -wz, wy],
            [wz, 1, -wx],
            [-wy, wx, 1]
        ])
        
        # Apply perturbation to all backbone atoms
        for attr in ['ca', 'cb', 'c', 'n', 'o']:
            coords = getattr(perturbed_protbb, attr)
            # Convert to numpy for manipulation
            coords_np = coords.squeeze(1).numpy()
            
            # Apply rotation and translation
            coords_np = coords_np @ rotation_matrix.T + translation
            
            # Convert back to tensor
            setattr(perturbed_protbb, attr, torch.tensor(coords_np).unsqueeze(1).float())
    
    return perturbed_protbb


def score_structure(models: List, protbb: ProtBB, device: str) -> float:
    """
    Score a protein structure using Pythia likelihood.
    
    Args:
        models: List of Pythia models
        protbb: Protein backbone structure
        device: Device for computation ('cpu' or 'cuda')
    
    Returns:
        Negative log-likelihood score (lower is better)
    """
    node, edge, native_seq = get_neighbor(protbb, noise_level=0.00)
    score = 0.0
    
    for model in models:
        with torch.no_grad():
            y_hat, _ = model(node.to(device).float(), edge.to(device).float())
            y_hat = y_hat.cpu()
            score += -torch.mean(
                torch.log(torch.gather(y_hat.softmax(-1), 1, native_seq.unsqueeze(-1)))
            )
    
    score /= len(models)
    return score.item()


def minimize_energy(
    pdb_file: str,
    models: List,
    device: str,
    max_iterations: int = 100,
    translation_std: float = 0.1,
    rotation_std: float = 0.05,
    max_rmsd: float = 2.0,
    convergence_threshold: float = 1e-4,
    output_file: Optional[str] = None,
) -> Tuple[ProtBB, float, int]:
    """
    Minimize protein backbone energy using Pythia likelihood scoring.
    
    Args:
        pdb_file: Path to input PDB file
        models: List of Pythia models for scoring
        device: Device for computation ('cpu' or 'cuda')
        max_iterations: Maximum number of optimization iterations
        translation_std: Standard deviation for translation noise (Angstroms)
        rotation_std: Standard deviation for rotation noise (radians)
        max_rmsd: Maximum allowed RMSD from initial structure
        convergence_threshold: Threshold for score change to declare convergence
        output_file: Path to save optimized structure (optional)
    
    Returns:
        Tuple of (optimized_protbb, final_score, num_iterations)
    """
    # Load initial structure
    initial_protbb = read_pdb_to_protbb(pdb_file)
    initial_ca = initial_protbb.ca.squeeze(1).numpy()
    
    # Score initial structure
    current_protbb = deepcopy(initial_protbb)
    current_score = score_structure(models, current_protbb, device)
    best_protbb = deepcopy(current_protbb)
    best_score = current_score
    initial_score = current_score
    
    print(f"Initial score: {current_score:.4f}")
    
    accepted_count = 0
    improved_count = 0
    
    # Optimization loop
    for iteration in range(max_iterations):
        # Generate perturbed structure
        perturbed_protbb = apply_perturbation(
            current_protbb,
            translation_std=translation_std,
            rotation_std=rotation_std
        )
        
        # Check RMSD constraint
        perturbed_ca = perturbed_protbb.ca.squeeze(1).numpy()
        rmsd = compute_rmsd(initial_ca, perturbed_ca)
        
        if rmsd > max_rmsd:
            # Reject perturbation if it violates RMSD constraint
            continue
        
        # Score perturbed structure
        perturbed_score = score_structure(models, perturbed_protbb, device)
        
        # Metropolis criterion: accept if score improves or with small probability for worse solutions
        score_diff = perturbed_score - current_score
        accept = False
        
        if score_diff < 0:
            # Accept improvement
            accept = True
            improved_count += 1
        elif np.random.rand() < 0.1:
            # Occasionally accept worse solutions to escape local minima
            accept = True
        
        if accept:
            current_protbb = perturbed_protbb
            current_score = perturbed_score
            accepted_count += 1
            
            if current_score < best_score:
                improvement = best_score - current_score
                best_protbb = deepcopy(current_protbb)
                best_score = current_score
                
                print(f"Iteration {iteration + 1}: Score = {best_score:.4f}, "
                      f"RMSD = {rmsd:.3f} Å, Improvement = {improvement:.6f}, "
                      f"Accepted = {accepted_count}/{iteration+1}")
                
                # Check convergence
                if improvement < convergence_threshold and iteration > 10:
                    print(f"Converged after {iteration + 1} iterations")
                    break
    
    print(f"\nFinal score: {best_score:.4f}")
    print(f"Score improvement: {initial_score - best_score:.4f}")
    print(f"Total accepted moves: {accepted_count}/{max_iterations}")
    print(f"Improved moves: {improved_count}/{max_iterations}")
    
    # Save optimized structure if requested
    if output_file:
        save_protbb_to_pdb(best_protbb, initial_protbb, pdb_file, output_file)
        print(f"Saved optimized structure to {output_file}")
    
    return best_protbb, best_score, iteration + 1


def save_protbb_to_pdb(
    protbb: ProtBB,
    reference_protbb: ProtBB,
    reference_pdb: str,
    output_file: str
) -> None:
    """
    Save ProtBB structure to PDB file.
    
    Args:
        protbb: ProtBB structure to save
        reference_protbb: Reference ProtBB for sequence information
        reference_pdb: Path to reference PDB file
        output_file: Path to output PDB file
    """
    from Bio.PDB import PDBParser, PDBIO
    
    # Read reference structure to get residue information
    parser = PDBParser(QUIET=True)
    ref_structure = parser.get_structure("ref", reference_pdb)
    
    # Create new structure
    io = PDBIO()
    
    # Update coordinates in the reference structure
    residue_idx = 0
    for chain in ref_structure[0].get_chains():
        for residue in chain.get_residues():
            if residue.id[0] == ' ' and residue_idx < len(protbb.ca):
                # Update backbone atom coordinates
                if 'CA' in residue:
                    residue['CA'].coord = protbb.ca[residue_idx].squeeze().numpy()
                if 'CB' in residue:
                    residue['CB'].coord = protbb.cb[residue_idx].squeeze().numpy()
                if 'C' in residue:
                    residue['C'].coord = protbb.c[residue_idx].squeeze().numpy()
                if 'N' in residue:
                    residue['N'].coord = protbb.n[residue_idx].squeeze().numpy()
                if 'O' in residue:
                    residue['O'].coord = protbb.o[residue_idx].squeeze().numpy()
                
                residue_idx += 1
    
    # Save structure
    io.set_structure(ref_structure)
    io.save(output_file)


def main():
    parser = argparse.ArgumentParser(
        description="Minimize protein backbone energy using Pythia likelihood scoring"
    )
    parser.add_argument(
        "--pdb_file",
        type=str,
        required=True,
        help="Path to input PDB file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output PDB file (default: <input>_minimized.pdb)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for computation (cpu or cuda)"
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=100,
        help="Maximum number of optimization iterations"
    )
    parser.add_argument(
        "--translation_std",
        type=float,
        default=0.1,
        help="Standard deviation for translation noise (Angstroms)"
    )
    parser.add_argument(
        "--rotation_std",
        type=float,
        default=0.05,
        help="Standard deviation for rotation noise (radians)"
    )
    parser.add_argument(
        "--max_rmsd",
        type=float,
        default=2.0,
        help="Maximum allowed RMSD from initial structure (Angstroms)"
    )
    parser.add_argument(
        "--convergence_threshold",
        type=float,
        default=1e-4,
        help="Threshold for score improvement to declare convergence"
    )
    
    args = parser.parse_args()
    
    # Set output file
    if args.output is None:
        base_name = args.pdb_file.replace('.pdb', '')
        args.output = f"{base_name}_minimized.pdb"
    
    # Load Pythia models
    print("Loading Pythia models...")
    torch_model_c = get_torch_model(
        os.path.join(pythia_root_dpath, "pythia-c.pt"),
        args.device
    )
    torch_model_p = get_torch_model(
        os.path.join(pythia_root_dpath, "pythia-p.pt"),
        args.device
    )
    models = [torch_model_c, torch_model_p]
    
    # Run energy minimization
    print(f"Starting energy minimization for {args.pdb_file}")
    print(f"Parameters:")
    print(f"  Max iterations: {args.max_iterations}")
    print(f"  Translation std: {args.translation_std} Å")
    print(f"  Rotation std: {args.rotation_std} rad")
    print(f"  Max RMSD: {args.max_rmsd} Å")
    print(f"  Convergence threshold: {args.convergence_threshold}")
    print()
    
    optimized_protbb, final_score, num_iterations = minimize_energy(
        pdb_file=args.pdb_file,
        models=models,
        device=args.device,
        max_iterations=args.max_iterations,
        translation_std=args.translation_std,
        rotation_std=args.rotation_std,
        max_rmsd=args.max_rmsd,
        convergence_threshold=args.convergence_threshold,
        output_file=args.output
    )
    
    print(f"\nOptimization completed in {num_iterations} iterations")
    print(f"Final score: {final_score:.4f}")


if __name__ == "__main__":
    main()
