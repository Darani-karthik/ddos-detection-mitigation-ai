# Methodology

This document describes exactly what the code does, so that the numbers in the README can be judged. Every parameter below can be found in the module named next to it.

## 1. The task

An SDN controller polls its switch once per second. For every flow entry that carried traffic it gets one **flow record** (source, destination, protocol, packets, bytes, age, whether the reverse flow is active; see [`data/README.md`](../data/README.md)). The detector scores each record: *is this flow part of a DDoS attack?* The mitigation policy then decides what to drop.

This follows the original course project (flow statistics collected in Mininet, normal = 0 / attack = 1) and adds the step the project listed as future work, an automated mitigation.

## 2. The simulated data ([`simulate.py`](../ddos_ai/simulate.py))

The original Mininet data set is not available, and Mininet does not run on every machine, so the data is produced by a flow-level simulator. It draws flow-windows from simple stochastic models; it does not simulate packets, queues or TCP. Everything is deterministic given the seed. A data set is 120 scenarios of 90 seconds (16 normal, 8 flash crowd, 18 each of SYN, UDP, ICMP and HTTP flood, 12 pulsing, 12 mixed), about 489,000 flow records.

**Normal traffic.** 10 to 25 clients talk to one server (10.0.0.1) in sessions. A session is one or two flows: web (small requests up, large replies down), bulk download, upload, one-way UDP media, DNS-style request / response, ping. Per-client rate is log-normal (median 18 packets/s, clipped to 3-120); 8% of clients are "heavy hitters" with 3-8 times that rate (up to 400 packets/s), which is as fast as a modest flood. A new session starts with probability 0.06-0.25 per window and lasts 14 windows on average. Packets per window are Poisson with a log-normal rate jitter.

**Flash crowd** (legitimate!). From window 25-44, for 20-35 windows, ramping up over 4-8 windows: web and bulk rates times 2.5-6, session-start probability times 2-4, plus 15-70 new short-lived visitors per second. In aggregate it looks like an attack; about 30% of the attack scenarios also contain one.

**Attacks** start in window 25-40, last 35-50 windows and ramp up over 3-8 windows:

| Kind | Sources | Rate | Packets | Pair flow |
|---|---|---|---|---|
| `syn_spoof` | a new random spoofed address for every flow | 30-150 new flows/s | TCP, 1-3 packets of 40-66 B | no |
| `udp` | 3-10 bots with fixed addresses | 60-1500 packets/s per bot | 300-1400 B | no |
| `icmp` | 3-10 bots | 40-1000 packets/s per bot | 64-1024 B | replies seen in half the scenarios |
| `http` | 3-10 bots | 20-200 packets/s per bot | TCP, 150-500 B | yes (server replies) |
| `pulse` | 3-10 bots | 100-1200 packets/s while on; on 2-5 s, off 4-10 s, synchronised or not | UDP (70%) or TCP | no |
| `mixed` | two different attacks at the same time | | | |

The HTTP flood is deliberately close to normal traffic and to a flash crowd. The rate ranges of floods and legitimate heavy hitters overlap.

**Not modelled:** packet-level timing, congestion and loss, TCP behaviour, reactive bots, application semantics, multi-switch topologies, real traffic mixes. Real floods are usually also *harder* to separate than these.

## 3. Features ([`features.py`](../ddos_ai/features.py))

15 inputs per record: packets, bytes, bytes per packet, packets/s, kbit/s, flow age, pair flow, protocol (one-hot), and five switch-level features of the same window: number of active flows, number of new flows (the packet-in rate), total packets/s, total kbit/s, and the normalised entropy of the packet share per record. IP addresses are not features. Switch-level features use only the record's own window, never the future.

GNB and the neural networks see the heavy-tailed features on a log1p scale, standardised with statistics from the training rows only. LightGBM uses the raw values.

## 4. Models ([`models.py`](../ddos_ai/models.py))

| Model | Input | Settings |
|---|---|---|
| GNB | 15 scaled features | scikit-learn `GaussianNB`, default priors |
| LightGBM | 15 raw features | gradient-boosted trees, learning rate 0.05, 31 leaves, min 50 samples per leaf, bagging and feature fraction 0.8, L2 1.0, up to 600 rounds with early stopping (30 rounds) on the validation scenarios |
| FNN | 15 scaled features | Dense 64 (ReLU), dropout 0.2, Dense 32 (ReLU), sigmoid; 3,137 parameters |
| LSTM | the last 5 windows of the same flow (a flow without history is padded; a mask channel marks real steps), 16 channels | LSTM 32, Dense 16 (ReLU), sigmoid; 6,817 parameters |

The neural networks use Adam (1e-3), batch size 512, class weights balanced to the training scenarios, at most 30 epochs with early stopping (patience 4) on validation loss, and TensorFlow's deterministic ops. A spoofed flood has no flow history (every spoofed flow is seen exactly once), so for it the LSTM sees the same information as the FNN. It can only use history for flows that persist (bots with fixed addresses).

## 5. Evaluation protocol ([`experiments.py`](../ddos_ai/experiments.py), [`evaluation.py`](../ddos_ai/evaluation.py))

* **Scenario-wise split.** Whole scenarios are assigned to train / validation / test (about 60 / 15 / 25%), stratified by scenario kind, so that no run, and no bot, is on both sides of the split. A unit test asserts the parts are disjoint.
* **Three splits.** The main comparison is run on three different random splits; the model seed equals the split seed. "Mean ± std" in the README is over these three runs and therefore includes both initialisation and split variation. Split 0 is the one used for the confusion matrices, precision-recall curves and the mitigation replay.
* **Threshold.** The decision threshold is the one that maximises balanced accuracy on the validation scenarios (an attack missed and a normal flow flagged weigh the same, whatever the class ratio). Metrics at the natural 0.5 threshold are in [`results/summary.md`](../results/summary.md).
* **Pooled metrics are dominated by the SYN flood**, which produces about 60% of all attack records (one record per spoofed source). The README therefore also reports **recall per attack type** and its mean (**macro recall**), and the false-alarm rate on normal and on flash-crowd traffic.
* **Leave-one-attack-out.** For each attack type, train on scenarios that contain no trace of it, tune the threshold on validation scenarios without it either, and test on scenarios of exactly that attack. One run per cell (seed 0).
* **Leakage comparison.** Same LightGBM, once on the scenario-wise split and once on a random split of the flow records with the same part sizes.

## 6. Mitigation ([`mitigation.py`](../ddos_ai/mitigation.py))

The controller feeds each window's scores to `MitigationController`:

1. **Source blocking with hysteresis.** A source flagged in 2 consecutive windows is blocked for 30 windows; the block is renewed while the source keeps being flagged. The protected server is on an allowlist and is never blocked. When a client is blocked, the replies to it stop as well.
2. **Protect mode** against spoofed floods. If one window has at least 20 flagged sources, then for the next 8 windows only *trusted* sources (seen unflagged in at least 3 windows) are let through and everything unknown is dropped. Blocking addresses cannot stop a spoofed flood, since every address appears once.

A decision after window ``t`` acts from window ``t + 1``.

**Replay.** `replay()` runs this policy over the recorded test scenarios (one fresh controller per scenario) with the detector scores of the first split and the thresholds tuned on its validation scenarios, and counts which records would have been dropped. Metrics: share of attack / legitimate packets dropped, legitimate packets dropped during flash crowds, peak packets/s reaching the server (before vs after), and windows until at least 90% of a window's attack packets are dropped. An **oracle** detector (the true labels as scores) shows what the policy itself costs when detection is perfect.

**Simplifications.** The replay does not feed the mitigation back into the traffic: attackers do not adapt, TCP does not back off, and dropped traffic keeps appearing in the statistics (as it does with OpenFlow drop rules, whose counters keep counting). The numbers say how the policy behaves on this data, not what a real network would do.

## 7. Relation to the original report

The course report ("DDoS Detection using ML", 23MAT204 Mathematics for Intelligent Systems 3, Amrita School of Artificial Intelligence, November 2024) trained the same four model families on Mininet-generated flow data (labelled and unlabelled sets, normal 0 / attack 1) and reported:

| Model | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|
| GNB | 91.8% | 0.8902 | 0.9011 | 0.8807 |
| LSTM | 98.8% | 0.9842 | 0.9868 | 0.9818 |
| FNN | 97.93% | 0.9728 | 0.9765 | 0.9692 |
| LightGBM | 92.5% | 0.8864 | 0.9868 | 0.8540 |

These numbers **are not reproduced here and are not comparable** to this repository's results: they come from a different, unavailable data set and an unknown evaluation protocol. One internal inconsistency is worth knowing about when reading the report: LightGBM's reported precision (0.9868) and recall (0.8540) give an F1 of about 0.916, not the 0.8864 in the same row, and its precision is identical to LSTM's. A transcription error is likely, but it cannot be checked.

The ranking in the report (LSTM first, then FNN, with GNB and LightGBM behind) is partly repeated here: LSTM leads and GNB is last, but LightGBM is on par with the FNN rather than behind it. This repository's contribution beyond the report: a documented simulator, a split that cannot leak, per-attack and unseen-attack evaluation, and the mitigation stage.
