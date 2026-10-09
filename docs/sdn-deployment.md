# Running it on a real SDN: Mininet + Ryu

> **Status: written but never executed.** The machine this repository was developed on runs Windows, without WSL, Mininet or Open vSwitch. The Python behind the SDN scripts (flow tracking, scoring, mitigation policy, labelling) is unit-tested; the glue that talks to Mininet, Open vSwitch and Ryu is not. The tests only check that the scripts parse. Expect to fix small things on first use, and please open an issue or a pull request if you do.

The pieces in [`sdn/`](../sdn):

| File | What it does |
|---|---|
| `topology.py` | Mininet topology: one OpenFlow 1.3 switch, the server `10.0.0.1`, clients `10.0.0.10...`, bot hosts `10.0.1.1...` |
| `run_experiment.py` | Builds the network, runs normal traffic, then an attack (`syn_spoof`, `udp`, `icmp` or `http`), then normal traffic again; writes `experiment.json` with the legitimate hosts and the attack window |
| `ryu_ddos_controller.py` | Ryu app: learning switch + per-second flow-statistics polling + (optional) detection and mitigation |

## Requirements

A Linux machine (a VM is fine) with Mininet 2.3 and Open vSwitch, `hping3`, `curl` and Python 3, plus Ryu. Ryu's dependencies (eventlet) are picky about the Python version: use the Python version your Ryu release supports (the Ryu documentation recommends a virtual environment). This repository's own tests run on Python 3.10 and 3.12; the controller app additionally needs `ddos_ai` importable and `lightgbm`, `pandas`, `numpy` installed in that environment:

```bash
pip install -e .          # from the repository root: makes `ddos_ai` importable by ryu-manager
pip install ryu           # see the Ryu documentation if this fails on your Python version
```

## 1. Collect labelled data from your own network

Terminal A, from the repository root:

```bash
DDOS_MODE=collect DDOS_CAPTURE=capture.csv ryu-manager sdn/ryu_ddos_controller.py
```

Terminal B:

```bash
sudo python3 sdn/run_experiment.py --attack syn_spoof --out experiment.json
```

The app logs one line per flow record and poll window to `capture.csv` (columns `ts, t, src, dst, proto, pkts, bytes, flow_age, pair_flow`). The experiment script writes the legitimate hosts and the attack window to `experiment.json`. Label the capture:

```bash
python -m ddos_ai.labeling capture.csv experiment.json labeled_syn.csv --scenario-id 0
```

The labelling rule: during the attack window, any flow towards the victim from an address that is **not** a known legitimate host is attack traffic. That also labels spoofed floods, whose source addresses are random. Repeat for each attack type with a different `--scenario-id` (and a fresh `capture.csv`, since the window counter restarts with the controller), then concatenate the labelled files, keeping one header line:

```bash
python - <<'EOF'
import pandas as pd
files = ["labeled_syn.csv", "labeled_udp.csv", "labeled_icmp.csv", "labeled_http.csv"]
pd.concat([pd.read_csv(f) for f in files]).to_csv("labeled_all.csv", index=False)
EOF
```

## 2. Train on it

```bash
python -m ddos_ai.run_all --data labeled_all.csv --out results_capture --models-dir models_capture
```

This runs the same comparison, leakage check and mitigation replay on your capture, with each scenario id as one scenario of the split. Use at least a handful of scenarios per class, or the split will be meaningless. Leave-one-attack-out is skipped for captures.

## 3. Defend

```bash
DDOS_MODE=defend DDOS_MODEL_DIR=models_capture ryu-manager sdn/ryu_ddos_controller.py
sudo python3 sdn/run_experiment.py --attack syn_spoof --out experiment_defended.json
```

The repository's committed [`models/lightgbm.txt`](../models) was trained on **simulated** data. It lets the app start, but do not expect it to separate real Mininet traffic well: retrain on your own capture as above.

What the app does in `defend` mode:

* Every poll it turns the switch's flow counters into records (`FlowTracker`), scores them (`OnlineDetector`, the LightGBM model) and feeds the scores to `MitigationController`.
* A blocked source gets high-priority drop entries with the same match as its flow entries, so the switch keeps counting the dropped packets.
* In protect mode the packet-ins of unknown sources are ignored: no flow entry, no forwarding.

## Known limitations

* One switch. With several, the per-source strike counts would be mixed across switches.
* A spoofed flood still generates packet-in messages that reach the controller, even when they are ignored. An OpenFlow meter on the table-miss entry would limit that and is not implemented.
* Flows that already have an entry when protect mode starts keep forwarding until their 2 s idle timeout.
* Packet-ins are matched to flow records only through the tracker; the first packet of every flow is forwarded by the controller and is not counted in the flow entry.
* The poll interval must equal `ddos_ai.schema.POLL_INTERVAL_S` (1 s) for the features to mean what they meant in training.
