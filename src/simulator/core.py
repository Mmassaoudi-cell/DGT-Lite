"""
Discrete-time edge-computing simulator reproducing the environment described in
Ali et al., "Attention-Driven Graph Convolutional Networks for Deadline-Constrained
Virtual Machine Task Allocation in Edge Computing," IEEE TCE, 2025.

CloudSim (Java) is unavailable in this environment, so this module is a documented
Python substitution implementing the paper's own semantics: physical hosts host VMs,
tasks (cloudlets) with MIPS length, type (CPU/IO-bound) and a deadline arrive over
time, a scheduler assigns each task to a VM, and PA / NPA / DVFS energy accounting
follows Eqs. (19)-(22) of the source paper (see SOURCE_PAPER_AUDIT.md, section 15).

All randomness is seed-controlled so that TRAIN / VALIDATION / TEST correspond to
disjoint *scenario* seeds (never disjoint rows of one workload) -- see
DATA_SPLIT_MANIFEST.csv and SOURCE_PAPER_AUDIT.md section 15, assumption 8.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

# ----------------------------------------------------------------------------
# Scale configurations (Section VI of the source paper).
# ----------------------------------------------------------------------------
SCALE_CONFIGS = {
    "small": dict(n_hosts=10, n_vms=20, n_tasks=20),
    "medium": dict(n_hosts=100, n_vms=200, n_tasks=200),
    "large": dict(n_hosts=1000, n_vms=2000, n_tasks=2000),
    # Table II configuration, used only for the direct reproduction-fidelity check.
    "table2": dict(n_hosts=100, n_vms=50, n_tasks=1000),
}

HIST_WINDOW = 8  # temporal window length fed to sequence models


@dataclass
class Task:
    task_id: int
    arrival_time: float
    length_mi: float          # million instructions
    task_type: int            # 0 = CPU-bound, 1 = IO-bound
    cpu_req: float            # fraction of a VM's MIPS capacity requested
    io_overhead: float        # fixed extra seconds for IO-bound tasks
    deadline: float           # absolute time by which task must finish


@dataclass
class VM:
    vm_id: int
    host_id: int
    mips: float
    ram: float
    storage: float
    free_at: float = 0.0          # time the VM's execution queue drains
    busy_time: float = 0.0        # cumulative execution time assigned
    sticky_client: int = -1       # last client / session routed here (sticky sessions)
    n_assigned: int = 0
    util_history: List[float] = field(default_factory=list)  # utilization per completed decision


@dataclass
class Host:
    host_id: int
    mips_capacity: float
    idle_power_frac: float = 0.70   # Eq. (19)-(21): idle machine draws 70% of full power
    on: bool = True


class EdgeSimulator:
    """One workload scenario / episode of the deadline-constrained VM allocation task."""

    def __init__(self, scale: str, seed: int, power_mode: str = "dvfs",
                 arrival_scale: float = 0.55, slack_range=(1.0, 3.2),
                 host_capacity_noise_std: float = 0.0):
        assert scale in SCALE_CONFIGS
        assert power_mode in ("npa", "pa", "dvfs")
        self.scale = scale
        self.power_mode = power_mode
        self.cfg = SCALE_CONFIGS[scale]
        self.rng = np.random.default_rng(seed)
        self.arrival_scale = arrival_scale
        self.slack_range = slack_range

        n_hosts, n_vms, n_tasks = self.cfg["n_hosts"], self.cfg["n_vms"], self.cfg["n_tasks"]
        self.n_hosts, self.n_vms, self.n_tasks = n_hosts, n_vms, n_tasks

        # Heterogeneous hosts (documented robustness improvement over the source
        # paper's implicit homogeneity assumption; homogeneous mode also supported).
        host_mips = self.rng.choice([2000.0, 4000.0, 8000.0], size=n_hosts,
                                     p=[0.5, 0.35, 0.15])
        if host_capacity_noise_std > 0:
            noise = self.rng.normal(1.0, host_capacity_noise_std, size=n_hosts).clip(0.3, 1.7)
            host_mips = host_mips * noise
        self.hosts = [Host(i, float(host_mips[i])) for i in range(n_hosts)]

        # VMs: 2 vCPU / 4GB / 100GB per Table II, MIPS derived from host capacity / colocated VMs.
        vms_per_host = max(1, n_vms // n_hosts)
        self.vms: List[VM] = []
        for v in range(n_vms):
            h = v % n_hosts
            host_mips_share = self.hosts[h].mips_capacity / max(1, vms_per_host)
            self.vms.append(VM(v, h, mips=host_mips_share, ram=4.0, storage=100.0))

        # VM adjacency: co-location on same host UNION communication-affinity edges
        # (documented assumption #2 in SOURCE_PAPER_AUDIT.md).
        self.adjacency = self._build_adjacency()

        # Tasks: Poisson-ish arrivals, CPU/IO-bound mix, deadline slack.
        self.tasks = self._generate_tasks()
        self.t_idx = 0  # index of next task to schedule

        # Bookkeeping
        self.n_met = 0
        self.n_missed = 0
        self.response_times: List[float] = []
        self.energy = 0.0
        self.last_energy_t = 0.0
        self.client_pool_size = max(4, n_vms // 4)

    # ------------------------------------------------------------------
    def _build_adjacency(self) -> np.ndarray:
        n = self.n_vms
        A = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                if i != j and self.vms[i].host_id == self.vms[j].host_id:
                    A[i, j] = 1.0
        # sparse random communication-affinity edges (task-chain co-invocation)
        n_extra = max(1, n // 5)
        for _ in range(n_extra):
            i, j = self.rng.integers(0, n, size=2)
            if i != j:
                A[i, j] = 1.0
                A[j, i] = 1.0
        return A

    def _generate_tasks(self) -> List[Task]:
        n = self.n_tasks
        # Poisson process arrivals -- bursty relative to service time to create genuine
        # deadline pressure and VM contention (avoids the ceiling effect where every
        # scheduler trivially meets every deadline).
        inter_arrival = self.rng.exponential(scale=self.arrival_scale, size=n)
        arrival_times = np.cumsum(inter_arrival)
        types = self.rng.binomial(1, 0.35, size=n)  # 35% IO-bound
        lengths = self.rng.lognormal(mean=8.3, sigma=0.7, size=n)  # MI
        cpu_reqs = self.rng.uniform(0.08, 0.45, size=n)
        io_overheads = np.where(types == 1, self.rng.uniform(0.05, 0.4, size=n), 0.0)
        # deadline slack: tighter, per "deadline-constrained" framing
        slack_factor = self.rng.uniform(*self.slack_range, size=n)
        tasks = []
        for i in range(n):
            base_service = lengths[i] / 3000.0  # rough reference service time
            deadline = arrival_times[i] + slack_factor[i] * base_service + io_overheads[i]
            tasks.append(Task(i, float(arrival_times[i]), float(lengths[i]), int(types[i]),
                               float(cpu_reqs[i]), float(io_overheads[i]), float(deadline)))
        return tasks

    # ------------------------------------------------------------------
    def done(self) -> bool:
        return self.t_idx >= self.n_tasks

    def current_task(self) -> Task:
        return self.tasks[self.t_idx]

    def vm_snapshot_features(self, vm: VM, task: Task) -> np.ndarray:
        """Instantaneous per-VM feature vector (documented assumption #1)."""
        host = self.hosts[vm.host_id]
        queue_wait = max(0.0, vm.free_at - task.arrival_time)
        proj_finish = max(vm.free_at, task.arrival_time) + self._service_time(vm, task)
        slack = task.deadline - proj_finish
        sticky = 1.0 if vm.sticky_client == (task.task_id % self.client_pool_size) else 0.0
        util = min(1.0, vm.busy_time / max(1e-6, task.arrival_time + 1e-6))
        health_ok = 1.0 if util < 0.95 else 0.0
        return np.array([
            queue_wait, proj_finish - task.arrival_time, slack, sticky,
            util, health_ok, vm.mips / 8000.0, host.mips_capacity / 8000.0,
            float(task.task_type), task.cpu_req,
        ], dtype=np.float32)

    def all_vm_features(self, task: Task) -> np.ndarray:
        return np.stack([self.vm_snapshot_features(vm, task) for vm in self.vms], axis=0)

    def push_util_history(self):
        t_now = self.tasks[self.t_idx - 1].arrival_time if self.t_idx > 0 else 0.0
        for vm in self.vms:
            util = min(1.0, vm.busy_time / max(1e-6, t_now + 1e-6))
            vm.util_history.append(util)
            if len(vm.util_history) > HIST_WINDOW:
                vm.util_history.pop(0)

    def history_tensor(self) -> np.ndarray:
        """(n_vms, HIST_WINDOW) padded utilization history for temporal models."""
        out = np.zeros((self.n_vms, HIST_WINDOW), dtype=np.float32)
        for i, vm in enumerate(self.vms):
            h = vm.util_history
            if h:
                out[i, -len(h):] = h
        return out

    def _service_time(self, vm: VM, task: Task) -> float:
        compute_time = task.length_mi / max(1e-6, vm.mips * task.cpu_req * 10.0)
        return compute_time + task.io_overhead

    # ------------------------------------------------------------------
    def step(self, vm_idx: int) -> dict:
        """Assign current task to vm_idx, advance simulation, return per-step outcome."""
        task = self.tasks[self.t_idx]
        vm = self.vms[vm_idx]
        start = max(vm.free_at, task.arrival_time)
        service = self._service_time(vm, task)
        finish = start + service
        vm.free_at = finish
        vm.busy_time += service
        vm.n_assigned += 1
        vm.sticky_client = task.task_id % self.client_pool_size
        met = finish <= task.deadline
        self.n_met += int(met)
        self.n_missed += int(not met)
        self.response_times.append(finish - task.arrival_time)

        self._accumulate_energy(task.arrival_time)
        self.push_util_history()
        self.t_idx += 1
        return dict(met=met, finish=finish, response_time=finish - task.arrival_time)

    def _accumulate_energy(self, t_now: float):
        dt = max(0.0, t_now - self.last_energy_t)
        if dt <= 0:
            self.last_energy_t = t_now
            return
        for host in self.hosts:
            host_vms = [vm for vm in self.vms if vm.host_id == host.host_id]
            active = any(vm.free_at > self.last_energy_t for vm in host_vms)
            if not host_vms:
                continue
            util = np.mean([min(1.0, vm.busy_time / max(1e-6, t_now)) for vm in host_vms])
            if self.power_mode == "npa":
                p = 1.0  # C * Vmax^2 * fmax, constant, host never sleeps
            elif self.power_mode == "pa":
                p = 0.0 if not active else (host.idle_power_frac +
                                             (1 - host.idle_power_frac) * min(1.0, util * 2))
            else:  # dvfs: continuous linear scaling with utilization
                p = host.idle_power_frac + (1 - host.idle_power_frac) * util
                if not active:
                    p *= 0.5
            self.energy += p * dt
        self.last_energy_t = t_now

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        utils = np.array([vm.busy_time / max(1e-6, self.tasks[-1].arrival_time)
                           for vm in self.vms])
        utils = np.clip(utils, 0, None)
        fairness = (utils.sum() ** 2) / (len(utils) * (utils ** 2).sum() + 1e-9)  # Jain's index
        total = self.n_met + self.n_missed
        return dict(
            deadline_compliance=self.n_met / max(1, total),
            load_balance_fairness=float(fairness),
            energy=self.energy,
            mean_response_time=float(np.mean(self.response_times)) if self.response_times else 0.0,
            makespan=self.tasks[-1].arrival_time if self.tasks else 0.0,
            util_std=float(np.std(utils)),
        )


def make_scenario(scale: str, seed: int, power_mode: str = "dvfs", **kwargs) -> EdgeSimulator:
    return EdgeSimulator(scale=scale, seed=seed, power_mode=power_mode, **kwargs)


# Seed ranges enforce a *scenario* split: disjoint workload realizations per fold,
# never a random row split of one workload (SOURCE_PAPER_AUDIT.md leakage policy).
SPLIT_SEEDS = {
    "train": list(range(0, 60)),
    "val": list(range(1000, 1015)),
    "test": list(range(2000, 2020)),
}
