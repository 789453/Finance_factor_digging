import sys, inspect, importlib, pkgutil

print("Python:", sys.executable)

import gfn
print("gfn:", gfn.__file__)

from gfn.env import DiscreteEnv
print("DiscreteEnv.__init__:", inspect.signature(DiscreteEnv.__init__))

from gfn.modules import DiscretePolicyEstimator
print("DiscretePolicyEstimator.__init__:", inspect.signature(DiscretePolicyEstimator.__init__))

for path in ["gfn.utils", "gfn.utils.modules", "gfn.gflownet", "gfn.gflownet.trajectory_balance"]:
    try:
        mod = importlib.import_module(path)
        print(f"\n[{path}]")
        print([x for x in dir(mod) if "Neural" in x or "Flow" in x or "Estimator" in x])
    except Exception as e:
        print(f"\n[{path}] import failed: {e}")

print("\nSearching for NeuralNet...")
for m in pkgutil.walk_packages(gfn.__path__, prefix="gfn."):
    try:
        mod = importlib.import_module(m.name)
        if hasattr(mod, "NeuralNet"):
            print("Found NeuralNet in", m.name)
    except Exception:
        pass
