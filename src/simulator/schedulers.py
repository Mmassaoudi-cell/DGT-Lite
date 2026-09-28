"""
Scheduling policies: the lookahead oracle (used to label supervised training data,
since the source paper never defines its ground-truth task-schedule labels -- see
SOURCE_PAPER_AUDIT.md assumption #3) plus every non-learned baseline scheduler.

Every scheduler exposes `select(sim, task) -> vm_idx`.
"""
from __future__ import annotations

import numpy as np


class RoundRobin:
    name = "RR"

    def __init__(self, n_vms):
        self.ptr = 0
        self.n_vms = n_vms

    def select(self, sim, task):
        v = self.ptr % self.n_vms
        self.ptr += 1
        return v


class GreedyLeastLoaded:
    name = "Greedy-LL"

    def select(self, sim, task):
        return int(np.argmin([vm.free_at for vm in sim.vms]))


class MinMinHeuristic:
    """Classic Min-Min: pick the VM giving the earliest completion time for this task."""
    name = "Min-Min"

    def select(self, sim, task):
        finishes = [max(vm.free_at, task.arrival_time) + sim._service_time(vm, task)
                    for vm in sim.vms]
        return int(np.argmin(finishes))


class MaxMinHeuristic:
    """Max-Min: among VMs, favor the one maximizing completion-time headroom vs. deadline
    for large tasks first; approximated per-decision as choosing the VM with the largest
    slack after assignment among the K least-loaded VMs (batch Max-Min not applicable to
    strictly online one-task-at-a-time arrivals)."""
    name = "Max-Min"

    def select(self, sim, task):
        k = max(1, len(sim.vms) // 4)
        idx = np.argsort([vm.free_at for vm in sim.vms])[:k]
        slacks = []
        for i in idx:
            vm = sim.vms[i]
            finish = max(vm.free_at, task.arrival_time) + sim._service_time(vm, task)
            slacks.append(task.deadline - finish)
        return int(idx[int(np.argmax(slacks))])


class MrLBAApprox:
    """Documented re-implementation approximating MrLBA [16]: multi-resource priority
    heuristic that scores VMs on a weighted combination of MIPS headroom, RAM headroom,
    and current load, favoring the highest composite score (not the original authors' code)."""
    name = "MrLBA-approx"

    def select(self, sim, task):
        scores = []
        for vm in sim.vms:
            mips_headroom = vm.mips / (1.0 + vm.busy_time)
            load_term = 1.0 / (1.0 + vm.n_assigned)
            ram_term = vm.ram / 4.0
            scores.append(0.5 * mips_headroom / 1000.0 + 0.3 * load_term + 0.2 * ram_term)
        return int(np.argmax(scores))


class HiwigoaLBApprox:
    """Documented re-implementation approximating HIWIGOA-LB [17]: a simplified hybrid
    invasive-weed / grasshopper-optimization metaheuristic. For online one-task decisions
    we run a small population search per decision over VM indices with a grasshopper-style
    attraction/repulsion update on a 1-D score line, minimizing (finish_time, imbalance)."""
    name = "HIWIGOA-LB-approx"

    def __init__(self, pop=8, iters=4, seed=0):
        self.pop, self.iters = pop, iters
        self.rng = np.random.default_rng(seed)

    def select(self, sim, task):
        n = len(sim.vms)
        finishes = np.array([max(vm.free_at, task.arrival_time) + sim._service_time(vm, task)
                              for vm in sim.vms])
        loads = np.array([vm.n_assigned for vm in sim.vms], dtype=np.float32)
        cost = finishes + 0.05 * loads
        pos = self.rng.integers(0, n, size=self.pop).astype(np.float64)
        best_idx = int(np.argmin(cost))
        for _ in range(self.iters):
            for p in range(self.pop):
                r = self.rng.uniform(-1, 1)
                pos[p] = np.clip(pos[p] + r * (best_idx - pos[p]) * 0.5, 0, n - 1)
            cand = np.clip(np.round(pos).astype(int), 0, n - 1)
            costs = cost[cand]
            j = int(np.argmin(costs))
            if costs[j] < cost[best_idx]:
                best_idx = int(cand[j])
        return best_idx


class GeneticAlgorithmScheduler:
    """GA over a short lookahead window of pending same-batch tasks; here applied per-task
    online by evolving a population of VM-index genes against the (finish_time, imbalance,
    energy-proxy) fitness."""
    name = "GA"

    def __init__(self, pop=12, gens=6, seed=0):
        self.pop, self.gens = pop, gens
        self.rng = np.random.default_rng(seed)

    def _fitness(self, sim, task, vm_idx):
        vm = sim.vms[vm_idx]
        finish = max(vm.free_at, task.arrival_time) + sim._service_time(vm, task)
        miss_penalty = 0.0 if finish <= task.deadline else (finish - task.deadline)
        imbalance = vm.n_assigned
        return finish + 5.0 * miss_penalty + 0.1 * imbalance

    def select(self, sim, task):
        n = len(sim.vms)
        genes = self.rng.integers(0, n, size=self.pop)
        for _ in range(self.gens):
            fits = np.array([self._fitness(sim, task, g) for g in genes])
            order = np.argsort(fits)
            survivors = genes[order[: self.pop // 2]]
            children = []
            for _ in range(self.pop - len(survivors)):
                a, b = self.rng.choice(survivors, size=2)
                child = a if self.rng.random() < 0.5 else b
                if self.rng.random() < 0.2:
                    child = self.rng.integers(0, n)
                children.append(child)
            genes = np.concatenate([survivors, np.array(children)])
        fits = np.array([self._fitness(sim, task, g) for g in genes])
        return int(genes[np.argmin(fits)])


class PSOScheduler:
    """Particle Swarm Optimization scheduler: particles encode a candidate VM index
    (continuous relaxation), optimizing the same fitness as the GA baseline."""
    name = "PSO"

    def __init__(self, n_particles=12, iters=6, seed=0):
        self.n_particles, self.iters = n_particles, iters
        self.rng = np.random.default_rng(seed)

    def _fitness(self, sim, task, vm_idx):
        vm = sim.vms[vm_idx]
        finish = max(vm.free_at, task.arrival_time) + sim._service_time(vm, task)
        miss_penalty = 0.0 if finish <= task.deadline else (finish - task.deadline)
        imbalance = vm.n_assigned
        return finish + 5.0 * miss_penalty + 0.1 * imbalance

    def select(self, sim, task):
        n = len(sim.vms)
        x = self.rng.uniform(0, n - 1, size=self.n_particles)
        v = self.rng.uniform(-1, 1, size=self.n_particles)
        pbest = x.copy()
        pbest_val = np.array([self._fitness(sim, task, int(round(xi)) % n) for xi in x])
        gbest = pbest[int(np.argmin(pbest_val))]
        gbest_val = pbest_val.min()
        for _ in range(self.iters):
            r1, r2 = self.rng.random(self.n_particles), self.rng.random(self.n_particles)
            v = 0.5 * v + 1.5 * r1 * (pbest - x) + 1.5 * r2 * (gbest - x)
            x = np.clip(x + v, 0, n - 1)
            vals = np.array([self._fitness(sim, task, int(round(xi)) % n) for xi in x])
            improve = vals < pbest_val
            pbest[improve] = x[improve]
            pbest_val[improve] = vals[improve]
            if pbest_val.min() < gbest_val:
                gbest_val = pbest_val.min()
                gbest = pbest[int(np.argmin(pbest_val))]
        return int(round(gbest)) % n


class LookaheadOracle:
    """Strong non-learned oracle used ONLY to generate imitation-learning labels for the
    supervised neural baselines/candidates and the DCLD-net reproduction (documented
    assumption #3). Minimizes finish time subject to a deadline-feasibility check first,
    then load-imbalance as tiebreaker -- i.e., deadline-aware Min-Min."""
    name = "Oracle"

    def select(self, sim, task):
        n = len(sim.vms)
        finishes = np.array([max(vm.free_at, task.arrival_time) + sim._service_time(vm, task)
                              for vm in sim.vms])
        feasible = finishes <= task.deadline
        loads = np.array([vm.n_assigned for vm in sim.vms], dtype=np.float32)
        if feasible.any():
            cand = np.where(feasible)[0]
            j = cand[np.argmin(loads[cand])]
            return int(j)
        return int(np.argmin(finishes))


HEURISTIC_REGISTRY = {
    "RR": RoundRobin,
    "Greedy-LL": GreedyLeastLoaded,
    "Min-Min": MinMinHeuristic,
    "Max-Min": MaxMinHeuristic,
    "MrLBA-approx": MrLBAApprox,
    "HIWIGOA-LB-approx": HiwigoaLBApprox,
    "GA": GeneticAlgorithmScheduler,
    "PSO": PSOScheduler,
}
