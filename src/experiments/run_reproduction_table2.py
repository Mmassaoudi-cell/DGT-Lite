import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.experiments.benchmark import run_all

if __name__ == "__main__":
    os.makedirs("results/raw", exist_ok=True)
    methods = ["RR", "MrLBA-approx", "HIWIGOA-LB-approx", "DCLD-net-repro"]
    cache = {}  # share the collected supervised dataset across power modes
    # Source paper's own numeric tables cover only PA (Table III) and DVFS (Table IV);
    # NPA is discussed qualitatively only, so we skip it here for compute efficiency.
    for power_mode in ["pa", "dvfs"]:
        print(f"=== table2 scale, power_mode={power_mode} ===", flush=True)
        df = run_all("table2", methods, split="test", n_seeds=3, power_mode=power_mode,
                      cache=cache, verbose=True)
        df["power_mode"] = power_mode
        df.to_csv(f"results/raw/reproduction_table2_{power_mode}.csv", index=False)
    print("DONE", flush=True)
