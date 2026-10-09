"""Ryu (OpenFlow 1.3) controller app: learning switch + flow-statistics collection + DDoS defence.

NOT EXECUTED while this repository was written: the author's machine runs Windows, and this needs Linux,
Mininet / Open vSwitch and Ryu. It only reuses the tested pieces of `ddos_ai` (FlowTracker, OnlineDetector,
MitigationController); the glue below is the part that has never run, so expect to debug it on first use.
Written for one switch (the Mininet topology in `topology.py`).

Run from the repository root, with `ddos_ai` importable (`pip install -e .`):

    # 1. record labelled training data from your own Mininet traffic
    DDOS_MODE=collect DDOS_CAPTURE=capture.csv ryu-manager sdn/ryu_ddos_controller.py
    # 2. defend, with a model trained by `python -m ddos_ai.run_all`
    DDOS_MODE=defend DDOS_MODEL_DIR=models ryu-manager sdn/ryu_ddos_controller.py

Environment variables: DDOS_MODE (collect | defend | off, default defend), DDOS_MODEL_DIR (default models),
DDOS_CAPTURE (CSV to log every poll window to), DDOS_POLL (poll interval in seconds, default 1.0; keep it equal to
`ddos_ai.schema.POLL_INTERVAL_S`, which the models were trained with).

How it works
------------
* Forwarding: a MAC-learning switch. IPv4 flows get an entry matching (in_port, src, dst, protocol) with a 2 s idle
  timeout, so each entry is exactly one "flow record" of the data set and expires when the flow goes quiet.
* Every poll the app asks the switch for flow statistics, `FlowTracker` turns the cumulative counters into per-window
  records, the detector scores them and `MitigationController` decides what to drop.
* Blocking a source installs high-priority drop entries with the same match as the flow's own entry (so the switch
  keeps counting the dropped packets, as the replay in `ddos_ai.mitigation` assumes). New flows of a blocked or, in
  protect mode, unknown source are not given an entry: the app just ignores their packet-in (and counts it).
  Limitation: packet-ins of a spoofed flood still reach the controller. Rate-limiting the table-miss entry with an
  OpenFlow meter is the proper fix and is not implemented here.
"""
import csv
import os
import time

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, DEAD_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.lib import hub
from ryu.lib.packet import ether_types, ethernet, ipv4, packet
from ryu.ofproto import ofproto_v1_3

from ddos_ai.mitigation import MitigationConfig, MitigationController
from ddos_ai.runtime import FlowTracker, OnlineDetector
from ddos_ai.schema import PROTO_ICMP, PROTO_TCP, PROTO_UDP, int_to_ip, ip_to_int

IP_TO_MODEL_PROTO = {1: PROTO_ICMP, 6: PROTO_TCP, 17: PROTO_UDP}
MODEL_TO_IP_PROTO = {v: k for k, v in IP_TO_MODEL_PROTO.items()}
FORWARD_PRIORITY, DROP_PRIORITY, IDLE_TIMEOUT = 10, 100, 2


class DdosDefender(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = os.environ.get("DDOS_MODE", "defend")
        self.poll = float(os.environ.get("DDOS_POLL", "1.0"))
        self.datapaths = {}
        self.mac_to_port = {}
        self.trackers = {}                   # one FlowTracker per switch
        self.partial = {}                    # multipart flow-stats replies being assembled
        self.window = -1                     # index of the current poll window
        self.installed_until = {}            # (src, dst, proto) -> window until which a drop entry is installed
        self.detector = self.policy = None
        if self.mode == "defend":
            self.detector = OnlineDetector(os.environ.get("DDOS_MODEL_DIR", "models"))
            self.policy = MitigationController(MitigationConfig(threshold=self.detector.threshold))
            self.logger.info("defence on, threshold %.3f", self.detector.threshold)
        self.capture = None
        if os.environ.get("DDOS_CAPTURE"):
            path = os.environ["DDOS_CAPTURE"]
            fresh = not os.path.exists(path)
            self.capture = open(path, "a", newline="")
            self.capture_writer = csv.writer(self.capture)
            if fresh:
                self.capture_writer.writerow(["ts", "t", "src", "dst", "proto", "pkts", "bytes", "flow_age",
                                              "pair_flow"])
        self.monitor_thread = hub.spawn(self._monitor)

    # ------------------------------------------------------------------------- switch set-up
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def _switch_features(self, ev):
        dp = ev.msg.datapath
        ofp, parser = dp.ofproto, dp.ofproto_parser
        miss = [parser.OFPActionOutput(ofp.OFPP_CONTROLLER, ofp.OFPCML_NO_BUFFER)]
        self._add_flow(dp, 0, parser.OFPMatch(), miss)

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change(self, ev):
        dp = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            self.datapaths[dp.id] = dp
            self.trackers.setdefault(dp.id, FlowTracker())
        elif ev.state == DEAD_DISPATCHER:
            self.datapaths.pop(dp.id, None)

    def _add_flow(self, dp, priority, match, actions, idle=0, hard=0):
        ofp, parser = dp.ofproto, dp.ofproto_parser
        instructions = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)] if actions else []
        dp.send_msg(parser.OFPFlowMod(datapath=dp, priority=priority, match=match, instructions=instructions,
                                      idle_timeout=idle, hard_timeout=hard))

    # ----------------------------------------------------------------------------- forwarding
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in(self, ev):
        msg = ev.msg
        dp = msg.datapath
        ofp, parser = dp.ofproto, dp.ofproto_parser
        in_port = msg.match["in_port"]
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        self.mac_to_port.setdefault(dp.id, {})[eth.src] = in_port
        out_port = self.mac_to_port[dp.id].get(eth.dst, ofp.OFPP_FLOOD)
        actions = [parser.OFPActionOutput(out_port)]

        ip = pkt.get_protocol(ipv4.ipv4)
        if ip is not None and ip.proto in IP_TO_MODEL_PROTO:
            src, dst, proto = ip_to_int(ip.src), ip_to_int(ip.dst), IP_TO_MODEL_PROTO[ip.proto]
            if self.policy is not None and self.policy.is_dropped(src, self.window, dst):
                # no flow entry, no forwarding; the packet is still visible to the detector via the tracker
                self.trackers[dp.id].add_unmatched(src, dst, proto, len(msg.data))
                return
            if out_port != ofp.OFPP_FLOOD:
                match = parser.OFPMatch(in_port=in_port, eth_type=ether_types.ETH_TYPE_IP, ipv4_src=ip.src,
                                        ipv4_dst=ip.dst, ip_proto=ip.proto)
                self._add_flow(dp, FORWARD_PRIORITY, match, actions, idle=IDLE_TIMEOUT)

        data = msg.data if msg.buffer_id == ofp.OFP_NO_BUFFER else None
        dp.send_msg(parser.OFPPacketOut(datapath=dp, buffer_id=msg.buffer_id, in_port=in_port,
                                        actions=actions, data=data))

    # ------------------------------------------------------------------------------ monitoring
    def _monitor(self):
        while True:
            self.window += 1
            for dp in list(self.datapaths.values()):
                dp.send_msg(dp.ofproto_parser.OFPFlowStatsRequest(dp))
            hub.sleep(self.poll)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply(self, ev):
        msg = ev.msg
        dp = msg.datapath
        self.partial.setdefault(dp.id, []).extend(msg.body)
        if msg.flags & dp.ofproto.OFPMPF_REPLY_MORE:
            return
        body, self.partial[dp.id] = self.partial[dp.id], []

        counters = {}
        for stat in body:
            if stat.priority not in (FORWARD_PRIORITY, DROP_PRIORITY):
                continue
            try:
                key = (ip_to_int(stat.match["ipv4_src"]), ip_to_int(stat.match["ipv4_dst"]),
                       IP_TO_MODEL_PROTO[stat.match["ip_proto"]])
            except KeyError:
                continue
            packets, nbytes = counters.get(key, (0, 0))
            counters[key] = (packets + stat.packet_count, nbytes + stat.byte_count)

        records = self.trackers[dp.id].update(self.window, counters)
        if self.capture is not None and not records.empty:
            now = time.time()
            for r in records.itertuples(index=False):
                self.capture_writer.writerow([f"{now:.3f}", r.t, int_to_ip(r.src), int_to_ip(r.dst), r.proto,
                                              r.pkts, r.bytes, r.flow_age, r.pair_flow])
            self.capture.flush()
        if self.detector is None or records.empty:
            return

        scores = self.detector.score(records)
        self.policy.observe(self.window, records["src"].to_numpy(), scores)
        self._install_drop_entries(dp, records)

    # ------------------------------------------------------------------------------ mitigation
    def _install_drop_entries(self, dp, records):
        """Install or refresh drop entries for the flows of every currently blocked source."""
        parser = dp.ofproto_parser
        for src, dst, proto in zip(records["src"].tolist(), records["dst"].tolist(), records["proto"].tolist()):
            for endpoint in (src, dst):
                until = self.policy.blocked_until.get(endpoint, -1)
                key = (src, dst, proto)
                if until <= self.window or self.installed_until.get(key) == until:
                    continue
                match = parser.OFPMatch(eth_type=ether_types.ETH_TYPE_IP, ipv4_src=int_to_ip(src),
                                        ipv4_dst=int_to_ip(dst), ip_proto=MODEL_TO_IP_PROTO[proto])
                lifetime = max(1, int((until - self.window) * self.poll))
                self._add_flow(dp, DROP_PRIORITY, match, [], hard=lifetime)
                self.installed_until[key] = until
                self.logger.info("blocking %s -> %s (proto %d) for %d s", int_to_ip(src), int_to_ip(dst),
                                 MODEL_TO_IP_PROTO[proto], lifetime)
                break
