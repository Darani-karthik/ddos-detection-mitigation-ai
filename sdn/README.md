# SDN scripts (Mininet + Ryu)

Scripts to collect labelled flow data from a Mininet network and to run the detection and mitigation inside a Ryu controller.

**These scripts have never been run.** They need Linux, Mininet, Open vSwitch and Ryu; the repository was developed on Windows. The shared Python they call is unit-tested, the glue is not. Read [`docs/sdn-deployment.md`](../docs/sdn-deployment.md) before you use them.

| File | Purpose |
|---|---|
| [`topology.py`](topology.py) | one OpenFlow 1.3 switch, a web server, clients and bot hosts |
| [`run_experiment.py`](run_experiment.py) | normal traffic, then an attack, then normal traffic; writes `experiment.json` |
| [`ryu_ddos_controller.py`](ryu_ddos_controller.py) | learning switch + flow statistics + detection + mitigation (`DDOS_MODE=collect` or `defend`) |
