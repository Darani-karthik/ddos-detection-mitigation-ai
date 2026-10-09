#!/usr/bin/env python3
"""Mininet topology: one OpenFlow 1.3 switch, the protected web server, legitimate clients and bot hosts.

    server  10.0.0.1                (the victim; every scenario in the simulator uses the same address)
    c1..cN  10.0.0.10, 10.0.0.11, ...   legitimate clients
    b1..bM  10.0.1.1,  10.0.1.2,  ...   hosts that will play the bots

NOT EXECUTED while this repository was written (needs Linux, Mininet and Open vSwitch). Run it through
`run_experiment.py`, or start an interactive network with `sudo python3 sdn/topology.py`.
"""
from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import setLogLevel
from mininet.net import Mininet
from mininet.node import OVSKernelSwitch, RemoteController
from mininet.topo import Topo


class StarTopo(Topo):
    def build(self, n_clients=8, n_bots=4):
        switch = self.addSwitch("s1", protocols="OpenFlow13")
        self.addLink(self.addHost("server", ip="10.0.0.1/8"), switch, bw=100)
        for i in range(n_clients):
            self.addLink(self.addHost(f"c{i + 1}", ip=f"10.0.0.{10 + i}/8"), switch, bw=100)
        for i in range(n_bots):
            self.addLink(self.addHost(f"b{i + 1}", ip=f"10.0.1.{1 + i}/8"), switch, bw=100)


def build_network(n_clients=8, n_bots=4, controller_ip="127.0.0.1", controller_port=6653):
    """A Mininet network whose switch talks to a Ryu controller on `controller_ip:controller_port`."""
    return Mininet(topo=StarTopo(n_clients, n_bots), switch=OVSKernelSwitch, link=TCLink,
                   controller=lambda name: RemoteController(name, ip=controller_ip, port=controller_port),
                   autoSetMacs=True)


if __name__ == "__main__":
    setLogLevel("info")
    net = build_network()
    net.start()
    CLI(net)
    net.stop()
