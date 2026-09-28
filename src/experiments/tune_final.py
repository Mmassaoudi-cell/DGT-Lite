"""Stage 3 (master protocol section 9): light Optuna TPE tuning of the selected final
candidate (Distilled-DGT, chosen by results/aggregate/final_selection.json from VALIDATION-only
screening) and its teacher (DGT-Sched, runner-up). Tunes a small, meaningful hyperparameter set
-- never a large grid -- and never touches TEST seeds."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
import optuna

from src.simulator.core import SPLIT_SEEDS
from src.models.dataset import collect_dataset
from src.models import candidates as C
from src.experiments.benchmark import run_episode, TRAIN_SEEDS
from src.utils.metrics import sqs

SCALE = "small"
N_VAL_SEEDS = 3


def build_teacher(samples, adj_by_ep, hidden=32, lr=8e-4, lam=0.5, epochs=14, seed=0):
    teacher = C.DGTSched(hidden=hidden)
    C.train_dgt_supervised(teacher, samples, adj_by_ep, epochs=epochs, lr=lr, lam=lam, seed=seed)
    return teacher


def eval_scores(sched, scale, n_seeds=N_VAL_SEEDS):
    scores = []
    for i in range(n_seeds):
        val_seed = SPLIT_SEEDS["val"][i]
        summ = run_episode(scale, val_seed, sched)
        scores.append(sqs(summ["deadline_compliance"], summ["load_balance_fairness"],
                           summ["energy"], energy_ref=summ["energy"] * 1.3))
    return float(np.mean(scores))


def objective(trial, samples, adj_by_ep, teacher):
    hidden = trial.suggest_categorical("hidden", [8, 16, 24, 32])
    lr = trial.suggest_float("lr", 3e-4, 3e-3, log=True)
    alpha = trial.suggest_float("alpha", 0.2, 0.8)
    T = trial.suggest_float("T", 1.0, 4.0)
    epochs = trial.suggest_categorical("epochs", [8, 12, 16])

    scores = []
    for i in range(N_VAL_SEEDS):
        student = C.DistilledDGT(hidden=hidden)
        C.distill_train(student, teacher, samples, adj_by_ep, epochs=epochs, lr=lr,
                         alpha=alpha, T=T, seed=i)
        sched = C.LogitScheduler(student, "Distilled-DGT")
        val_seed = SPLIT_SEEDS["val"][i]
        summ = run_episode(SCALE, val_seed, sched)
        scores.append(sqs(summ["deadline_compliance"], summ["load_balance_fairness"],
                           summ["energy"], energy_ref=summ["energy"] * 1.3))
    return float(np.mean(scores))


if __name__ == "__main__":
    os.makedirs("results/aggregate", exist_ok=True)
    samples, adj_by_ep = collect_dataset(SCALE, TRAIN_SEEDS[SCALE], eps=0.3)

    print("Training reference teacher (DGT-Sched) with default hyperparameters ...")
    teacher = build_teacher(samples, adj_by_ep)
    teacher_sched = C.LogitScheduler(teacher, "DGT-Sched")
    teacher_score = eval_scores(teacher_sched, SCALE)
    print("Teacher (DGT-Sched) validation SQS:", teacher_score)

    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(lambda t: objective(t, samples, adj_by_ep, teacher), n_trials=15,
                    show_progress_bar=False)
    print("Best params:", study.best_params)
    print("Best value:", study.best_value)

    with open("results/aggregate/tuning_distilled_dgt.json", "w") as f:
        json.dump(dict(teacher_score=teacher_score, best_params=study.best_params,
                        best_value=study.best_value,
                        trials=[dict(params=t.params, value=t.value) for t in study.trials]),
                  f, indent=2)
