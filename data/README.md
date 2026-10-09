# Data

There is nothing to download. The data set is **simulated**, deterministically, by [`ddos_ai/simulate.py`](../ddos_ai/simulate.py)
the first time you run `python -m ddos_ai.run_all` (about 10 seconds) and cached in `cache/`, which is not committed.

`sample_flows.csv` is a small example (4 scenarios of 40 s: normal, SYN flood, UDP flood, HTTP flood; 4,228 rows) so the layout can be seen
without running anything. Regenerate it with `python -m ddos_ai.simulate`.

## Layout of a flow record

One row is what the SDN controller learns about **one flow entry in one poll window** (1 second):

| Column | Meaning |
|---|---|
| `scenario_id` | the simulated run the record belongs to (train / validation / test are split by this) |
| `t` | poll window number within the scenario (seconds) |
| `src`, `dst` | IPv4 addresses (dotted quads in the CSV, integers in memory). The protected server is `10.0.0.1` |
| `proto` | 0 = TCP, 1 = UDP, 2 = ICMP |
| `pkts`, `bytes` | packets and bytes of this flow during the window |
| `flow_age` | consecutive windows the flow has carried traffic (1 = new flow; a silent window resets it) |
| `pair_flow` | 1 if the reverse flow (`dst` to `src`) is active in the same window |
| `label` | 0 = normal, 1 = attack |
| `attack_type` | `none`, `syn_spoof`, `udp`, `icmp`, `http` or `pulse` |

The model features (see [`ddos_ai/features.py`](../ddos_ai/features.py)) are computed from these columns: per-flow rates and sizes plus
switch-level context of the same window (number of active flows, new flows, total rate, entropy). IP addresses are never model inputs.

## Using your own capture

Any CSV with the columns `t, src, dst, proto, pkts, bytes, flow_age, pair_flow, label` (and optionally `scenario_id`, `attack_type`) works:

```bash
python -m ddos_ai.run_all --data my_capture.csv --out results_capture --models-dir models_capture
```

The Ryu app in [`sdn/`](../sdn) writes exactly this format; see [`docs/sdn-deployment.md`](../docs/sdn-deployment.md).
