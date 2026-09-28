import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
from src.simulator.core import make_scenario, SPLIT_SEEDS
from src.models.dataset import collect_dataset
from src.models import candidates as C
from src.models.dqn import train_dqn, DQNScheduler


def run_episode(scale, seed, sched):
    sim = make_scenario(scale, seed)
    while not sim.done():
        task = sim.current_task()
        v = sched.select(sim, task)
        sim.step(v)
    return sim.summary()


print("== C1 DA-STGAT-RL (actor-critic) ==")
t0 = time.time()
m1 = C.DA_STGAT_RL()
C.train_actor_critic(m1, "small", seeds=list(range(0, 5)), epochs=2, log_every=1)
print("train time", time.time() - t0)
s1 = C.ACScheduler(m1, "DA-STGAT-RL")
print(run_episode("small", SPLIT_SEEDS["val"][0], s1))

print("== C3 MoE-GAT-AC (actor-critic) ==")
t0 = time.time()
m3 = C.MoE_GAT_AC()
C.train_actor_critic(m3, "small", seeds=list(range(0, 5)), epochs=2, log_every=1)
print("train time", time.time() - t0)
s3 = C.ACScheduler(m3, "MoE-GAT-AC")
print(run_episode("small", SPLIT_SEEDS["val"][0], s3))

print("== dataset for supervised C2/C4 ==")
samples, adj_by_ep = collect_dataset("small", seeds=list(range(0, 10)), eps=0.3)
print(len(samples), "samples")

print("== C2 DGT-Sched (supervised, deadline-aware loss) ==")
t0 = time.time()
m2 = C.DGTSched()
C.train_dgt_supervised(m2, samples, adj_by_ep, epochs=5, log_every=1)
print("train time", time.time() - t0)
s2 = C.LogitScheduler(m2, "DGT-Sched")
print(run_episode("small", SPLIT_SEEDS["val"][0], s2))

print("== C4 Distilled-DGT (distilled from C2) ==")
t0 = time.time()
m4 = C.DistilledDGT()
C.distill_train(m4, m2, samples, adj_by_ep, epochs=5, log_every=1)
print("train time", time.time() - t0)
s4 = C.LogitScheduler(m4, "Distilled-DGT")
print(run_episode("small", SPLIT_SEEDS["val"][0], s4))

print("== DQN ==")
t0 = time.time()
qnet = train_dqn("small", seeds=list(range(0, 5)), epochs=1, log_every=1)
print("train time", time.time() - t0)
sdqn = DQNScheduler(qnet)
print(run_episode("small", SPLIT_SEEDS["val"][0], sdqn))
