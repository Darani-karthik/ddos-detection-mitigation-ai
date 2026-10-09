"""Label flow records captured from a real Mininet run, so that they can be used for training.

In an experiment the operator knows who the legitimate hosts are and when the attack ran, which is all
that is needed: during the attack window, any flow towards the victim from an address that is *not* a
legitimate host is attack traffic. This also labels spoofed floods, whose source addresses are random.
"""
from __future__ import annotations

import pandas as pd

from .schema import ATTACK_TYPE, LABEL, VICTIM_IP, ip_to_int


def label_capture(records: pd.DataFrame, legit_hosts: list[str], attack_start: int, attack_end: int,
                  victim: int = VICTIM_IP, attack_name: str = "attack") -> pd.DataFrame:
    """Add `label` and `attack_type` columns. `attack_start` / `attack_end` are window numbers (end exclusive)."""
    legit = {ip_to_int(h) for h in legit_hosts} | {victim}
    during = records["t"].between(attack_start, attack_end - 1)
    is_attack = during & (records["dst"] == victim) & ~records["src"].isin(legit)
    out = records.copy()
    out[LABEL] = is_attack.astype("int8")
    out[ATTACK_TYPE] = is_attack.map({True: attack_name, False: "none"})
    return out


def main(argv=None) -> None:
    """Command line: label a capture written by the Ryu app with the metadata of `sdn/run_experiment.py`."""
    import argparse
    import json
    from pathlib import Path

    from .schema import int_to_ip

    ap = argparse.ArgumentParser(description="Label a Ryu capture: python -m ddos_ai.labeling capture.csv "
                                             "experiment.json labeled.csv")
    ap.add_argument("capture")
    ap.add_argument("experiment")
    ap.add_argument("out")
    ap.add_argument("--scenario-id", type=int, help="tag every row with this scenario id (useful when you "
                                                    "concatenate several experiments)")
    args = ap.parse_args(argv)

    meta = json.loads(Path(args.experiment).read_text())
    capture = pd.read_csv(args.capture)
    for column in ("src", "dst"):
        capture[column] = capture[column].map(ip_to_int)
    first_ts = capture.groupby("t")["ts"].min()
    start = int(first_ts[first_ts >= meta["attack_start"]].index.min())
    end = int(first_ts[first_ts <= meta["attack_end"]].index.max()) + 1
    labeled = label_capture(capture, meta["legit_hosts"], start, end, ip_to_int(meta.get("victim", "10.0.0.1")),
                            meta.get("attack", "attack"))
    if args.scenario_id is not None:
        labeled.insert(0, "scenario_id", args.scenario_id)
    for column in ("src", "dst"):
        labeled[column] = labeled[column].map(int_to_ip)
    labeled.drop(columns=["ts"]).to_csv(args.out, index=False)
    print(f"{args.out}: {len(labeled):,} records, {int(labeled['label'].sum()):,} labelled attack "
          f"(windows {start}-{end - 1})")


if __name__ == "__main__":
    main()
