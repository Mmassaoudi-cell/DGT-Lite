import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
from src.models.dataset import collect_dataset
from src.models import flexible
from src.models.train_supervised import train_model, NeuralScheduler
from src.simulator.core import make_scenario, SPLIT_SEEDS

t0 = time.time()
samples, adj_by_ep = collect_dataset("small", seeds=list(range(0, 15)), eps=0.3)
print(f"collected {len(samples)} samples in {time.time()-t0:.1f}s")

model = flexible.build("DCLD-net-repro")
t0 = time.time()
train_model(model, samples, adj_by_ep, epochs=8, batch_size=32, loss_type="mse_softmax",
            log_every=1)
print(f"trained in {time.time()-t0:.1f}s")

sched = NeuralScheduler(model, "DCLD-net-repro")


def run_episode(scale, seed, sched):
    sim = make_scenario(scale, seed)
    while not sim.done():
        task = sim.current_task()
        v = sched.select(sim, task)
        sim.step(v)
    return sim.summary()


for seed in SPLIT_SEEDS["val"][:3]:
    summ = run_episode("small", seed, sched)
    print(seed, summ)
