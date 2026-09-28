import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.simulator.core import make_scenario, SPLIT_SEEDS
from src.simulator.schedulers import HEURISTIC_REGISTRY, LookaheadOracle, RoundRobin


def run_episode(scale, seed, sched_factory, power_mode="dvfs"):
    sim = make_scenario(scale, seed, power_mode)
    sched = sched_factory(sim)
    while not sim.done():
        task = sim.current_task()
        vm_idx = sched.select(sim, task)
        sim.step(vm_idx)
    return sim.summary()


def factory_for(name, n_vms):
    if name == "RR":
        return lambda sim: RoundRobin(sim.n_vms)
    if name == "Oracle":
        return lambda sim: LookaheadOracle()
    return lambda sim: HEURISTIC_REGISTRY[name]()


if __name__ == "__main__":
    for scale in ["small", "medium"]:
        for name in ["RR", "Greedy-LL", "Min-Min", "Max-Min", "MrLBA-approx",
                     "HIWIGOA-LB-approx", "GA", "PSO", "Oracle"]:
            t0 = time.time()
            summ = run_episode(scale, seed=SPLIT_SEEDS["test"][0], sched_factory=factory_for(name, None))
            dt = time.time() - t0
            print(f"{scale:8s} {name:20s} compliance={summ['deadline_compliance']:.3f} "
                  f"fairness={summ['load_balance_fairness']:.3f} energy={summ['energy']:.2f} "
                  f"resp={summ['mean_response_time']:.3f}  ({dt*1000:.0f} ms)")
