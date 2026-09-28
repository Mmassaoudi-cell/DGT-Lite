# DGT-Lite

DGT-Lite is a compact, deadline-aware scheduler for virtual-machine (VM) task allocation in edge
computing, distilled from a deadline-aware graph-attention teacher (DGT-Sched). 


## Repository structure

```
src/
  simulator/
    core.py          Discrete-time edge-computing simulator (hosts, VMs, tasks, energy model)
    schedulers.py     Round Robin, Greedy, Min-Min, Max-Min, MrLBA-approx, HIWIGOA-LB-approx,
                       GA, PSO heuristics, and the deadline-aware lookahead labeling oracle
  models/
    blocks.py          Shared attention-pool / GCN / GAT / TCN building blocks
    flexible.py         Configurable architecture used for the DCLD-net reproduction, its
                         architectural/loss ablations, and the MLP / LSTM / Transformer /
                         GCN-only baselines
    dqn.py               DQN baseline (online Q-learning against the simulator)
    candidates.py         The four screened hybrid candidates, including DGT-Sched (teacher)
                           and DGT-Lite (final distilled model)
    dataset.py             Imitation-learning dataset collection (oracle-labeled)
    train_supervised.py     Generic supervised/imitation training loop
  utils/
    metrics.py         Composite Scheduling Quality Score used for model selection
  experiments/
    benchmark.py                Unified benchmark runner (all methods, all scales, all seeds)
    run_main_grid.py             Main benchmark grid (all methods, small/medium scale)
    run_screening.py             Validation-only candidate screening
    select_final.py               Final-model selection from screening results
    tune_final.py                  Hyperparameter tuning of the selected model (Optuna)
    verify_selection_stable.py      Post-hoc check that the selection is not a training-budget
                                      artifact
    run_ablation_repro.py            Architectural and loss ablation of the reproduced baseline
    run_oversmoothing_diagnostic.py   Direct GCN representation-similarity measurement
    run_robustness_final.py           Distribution-shift robustness evaluation
    run_reproduction_table2.py         Reproduction-fidelity check against the source paper's
                                        own configuration
    run_large_scale.py                  Large-scale scalability check
    stats_analysis.py                    Paired statistical testing (Wilcoxon, Holm correction)
    figures.py / tables.py                Figure and LaTeX table generation from result files
FINAL_MODEL_CONFIG.yaml   Frozen hyperparameters and protocol for the final model
requirements.txt
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.11+ and a CPU is sufficient; no GPU is required.

## Data

No external dataset is used or required. All experiments run against a self-contained,
seed-reproducible discrete-time simulator (`src/simulator/core.py`) that generates the
host/VM/task workload procedurally. Running any experiment script regenerates the exact
workloads used in the paper from the fixed random seeds baked into the code, so there is nothing
to download.

## Training

Train an individual model through the shared benchmark harness, for example:

```bash
python -m src.experiments.benchmark --scale medium --methods "Distilled-DGT" --n_seeds 1
```

`--methods` accepts any comma-separated combination of the implemented schedulers (see
`src/experiments/benchmark.py` for the full registry); `--scale` selects the VM-pool size
(`small`, `medium`, `large`, or `table2`).

## Evaluation and reproducing the paper's experiments

Run the following in order:

```bash
# Validation-only screening of the candidate hybrid models
python -m src.experiments.run_screening
python -m src.experiments.select_final

# Light hyperparameter tuning of the selected model on validation data
python -m src.experiments.tune_final

# Main benchmark grid (all methods, small/medium scale, paired seeds)
python -m src.experiments.run_main_grid

# Architectural/loss ablation and the GCN over-smoothing diagnostic
python -m src.experiments.run_ablation_repro
python -m src.experiments.run_oversmoothing_diagnostic

# Distribution-shift robustness evaluation
python -m src.experiments.run_robustness_final

# Reproduction-fidelity check against the source paper's own configuration
python -m src.experiments.run_reproduction_table2

# Statistical testing, figures, and IEEE-ready tables from the collected results
python -m src.experiments.stats_analysis results/raw/main_benchmark.csv Distilled-DGT
python -m src.experiments.figures Distilled-DGT
python -m src.experiments.tables Distilled-DGT
```

Each script writes its raw, per-seed results to `results/raw/` and aggregate outputs to
`results/aggregate/` (both created on first run). `FINAL_MODEL_CONFIG.yaml` records the exact,
frozen hyperparameters and protocol used to produce the results reported in the paper.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{massaoudi2026dgtlite,
  title   = {DGT-Lite: A Distilled Deadline-Aware Graph-Attention Scheduler for Load Balancing in Edge Computing},
  author  = {Massaoudi, Mohamed and Ez Eddin, Maymouna},
  year    = {2026}
}
```

## License

MIT License. See `LICENSE`.

## Contact

Mohamed Massaoudi -- mohamed.massaoudi@tamu.edu
