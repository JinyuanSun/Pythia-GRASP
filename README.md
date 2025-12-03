# Pythia
A series of models for protein related tasks, including zero-shot [protein stability prediction](pythia/), [ligand binding pocket prediction](pythia-pocket/), and [protein backbone energy minimization](docs/energy_minimization.md).

## Installation

```bash
git clone https://github.com/Wublab/Pythia.git
cd Pythia
pip install -e .
```

## Zero-shot Stability Prediction of Mutations

```bash
pythia --pdb_filename examples/1pga.pdb 
```

## Zero-shot stability prediction of proteins
Lower the score is better.
```bash
python3 pythia/score.py
```

## Protein Backbone Energy Minimization
Optimize protein backbone structures using Pythia's likelihood scoring.
```bash
PYTHONPATH=. python3 pythia/energy_minimize.py --pdb_file examples/1pga.pdb --device cpu
```

See [detailed documentation](docs/energy_minimization.md) for more options and usage examples.

## Citation
```text
@article{sun2025structure,
  title={Structure-based self-supervised learning enables ultrafast protein stability prediction upon mutation},
  author={Sun, Jinyuan and Zhu, Tong and Cui, Yinglu and Wu, Bian},
  journal={The Innovation},
  volume={6},
  number={1},
  year={2025},
  publisher={Elsevier}
}
```