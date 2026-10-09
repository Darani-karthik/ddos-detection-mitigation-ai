#!/usr/bin/env python3
"""Run one experiment in Mininet: normal traffic, then a DDoS attack, then normal traffic again.

    sudo python3 sdn/run_experiment.py --attack syn_spoof --out experiment.json

Start the Ryu controller first (see `ryu_ddos_controller.py`). The script writes `experiment.json` with
the legitimate hosts and the wall-clock start and end of the attack, which `python -m ddos_ai.labeling` uses
to label the controller's capture.

NOT EXECUTED while this repository was written (needs Linux, Mininet, hping3, curl and iperf).
Every traffic generator is wrapped in `timeout`, so nothing is left running after the experiment and no
process is killed by name.
"""
import argparse
import json
import time

from topology import build_network

ATTACKS = {
    # name: hping3 arguments, or None for the HTTP request loop
    "syn_spoof": ["-S", "-p", "80", "--flood", "--rand-source"],
    "udp": ["--udp", "-p", "80", "--flood"],
    "icmp": ["--icmp", "--flood"],
    "http": None,
}
WEB_CLIENT_LOOP = "while true; do curl -s -o /dev/null --max-time 2 http://10.0.0.1/; sleep 0.$((RANDOM % 9 + 1)); done"
HTTP_FLOOD_LOOP = "while true; do curl -s -o /dev/null --max-time 2 http://10.0.0.1/; done"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attack", choices=sorted(ATTACKS), default="syn_spoof")
    ap.add_argument("--clients", type=int, default=8)
    ap.add_argument("--bots", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=30, help="seconds of normal traffic before the attack")
    ap.add_argument("--attack-duration", type=int, default=40)
    ap.add_argument("--cooldown", type=int, default=20, help="seconds of normal traffic after the attack")
    ap.add_argument("--controller-ip", default="127.0.0.1")
    ap.add_argument("--out", default="experiment.json")
    args = ap.parse_args()

    net = build_network(args.clients, args.bots, args.controller_ip)
    net.start()
    total = args.warmup + args.attack_duration + args.cooldown
    try:
        server = net.get("server")
        clients = [net.get(f"c{i + 1}") for i in range(args.clients)]
        bots = [net.get(f"b{i + 1}") for i in range(args.bots)]
        server.popen(["timeout", str(total + 10), "python3", "-m", "http.server", "80"])
        time.sleep(2)
        net.ping([server] + clients, timeout="1")             # let the switch learn every MAC address

        for client in clients:                                 # legitimate traffic for the whole run
            client.popen(["timeout", str(total), "bash", "-c", WEB_CLIENT_LOOP])
            client.popen(["timeout", str(total), "ping", "-i", "1", "10.0.0.1"])

        time.sleep(args.warmup)
        attack_start = time.time()
        for bot in bots:
            if ATTACKS[args.attack] is None:
                bot.popen(["timeout", str(args.attack_duration), "bash", "-c", HTTP_FLOOD_LOOP])
            else:
                bot.popen(["timeout", str(args.attack_duration), "hping3"] + ATTACKS[args.attack] + ["10.0.0.1"])
        time.sleep(args.attack_duration)
        attack_end = time.time()
        time.sleep(args.cooldown)

        with open(args.out, "w") as f:
            json.dump({
                "attack": args.attack,
                "victim": "10.0.0.1",
                "legit_hosts": [client.IP() for client in clients],
                "bot_hosts": [bot.IP() for bot in bots],
                "attack_start": attack_start,
                "attack_end": attack_end,
            }, f, indent=2)
        print(f"wrote {args.out}")
    finally:
        net.stop()


if __name__ == "__main__":
    main()
