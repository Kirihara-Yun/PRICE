# PRICE

**PRICE** — An Operator Recomputation and Fission Framework for Computation Graphs
via Attainable Arithmetic Intensity (AAI).

PRICE automatically explores the memory/latency trade-off space of deep-learning
computation graphs by combining:

- **Recomputation** (gradient checkpointing) scheduling
- **Operator fission** (intra-operator splitting) transformations
- A hardware-aware cost model built on attainable arithmetic intensity

## Installation

```bash
cd PRICE/python
pip install -e .
```

## Quick start

```python
import price.graph as OG
import price.optimizer as OPT
import price.scheduler as SCH
import price.runtime as BAK
from price.bench.nn.mlp import simple_mlp_net
from price.bench import setup_training_graph
from price.bench.configs import get_configured_mutator

# 1) build a computation graph
G, y = simple_mlp_net(16, [784, 256, 256, 10], num_classes=10)
# 2) turn it into a training graph (loss + backward)
G = setup_training_graph(G, y)

# 3) search recomputation / fission schedules
mutator = get_configured_mutator()
scheduler = SCH.RefineMemRematSchedule(adjust_load_op=False)
with BAK.CudaTorchBackend(dtype="float32") as bknd:
    opt = OPT.RecomputeOptimizer(scheduler, bknd, mutator)
    G_opt, sched = opt.run(G)
```

## Package layout

```
price/
├── graph.py                 # ComputeGraph: core computation-graph IR
├── dimension_graph.py       # dimension / memory graph
├── scheduler.py             # schedules (BaseSchedule, RematSchedule, ...)
├── optimizer.py             # RecomputeOptimizer: transform + schedule search
├── estimator.py             # DefaultEstimator: performance / memory simulation
├── ci_calculator.py         # attainable-arithmetic-intensity calculator
├── dhg_ir.py                # dynamic hardware-aware graph IR
├── optimization_algorithms.py  # recomputation / fission decision algorithms
├── nodes/                   # operator (graph-node) definitions
├── passes/                  # graph transformation passes & rewrite rules
├── runtime/                 # backend measurement (CudaTorchBackend)
├── common/                  # shared utilities (digest, profiling, logging)
└── bench/                   # benchmarks, reference networks, driver
```

## Self-contained analysis toolkit

`real_ai_optimization_analysis/` is an independent toolkit that empirically measures
the latency / memory trade-offs of recomputation and operator fission on real models.
Run it with:

```bash
cd PRICE/python/real_ai_optimization_analysis
python main.py
```

## License

See [LICENSE](../LICENSE).
