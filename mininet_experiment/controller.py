"""Ryu OpenFlow 1.3 application: explicit UDP routes, no learning/flooding."""
import json
import os
import time
from urllib.request import Request, urlopen

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, DEAD_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.app.wsgi import ControllerBase, WSGIApplication, route
from ryu.lib import hub
from webob import Response

from settings import WORKER_PORT, UDP_PORT

COOKIE = 0x474E4E01


class RoutingController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS = {"wsgi": WSGIApplication}

    def __init__(self, *args, **kwargs):
        super(RoutingController, self).__init__(*args, **kwargs)
        self.datapaths, self.waiters, self.errors = {}, {}, []
        self.busy = False
        kwargs["wsgi"].register(API, {"app": self})

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def state(self, event):
        dp = event.datapath
        if event.state == MAIN_DISPATCHER:
            self.datapaths[dp.id] = dp
        else:
            self.datapaths.pop(dp.id, None)

    @set_ev_cls(ofp_event.EventOFPErrorMsg, [CONFIG_DISPATCHER, MAIN_DISPATCHER])
    def error(self, event):
        self.errors.append({"dpid": event.msg.datapath.id, "type": event.msg.type,
                            "code": event.msg.code, "xid": event.msg.xid})

    @set_ev_cls(ofp_event.EventOFPBarrierReply, MAIN_DISPATCHER)
    def barrier_reply(self, event):
        waiter = self.waiters.pop((event.msg.datapath.id, event.msg.xid), None)
        if waiter is not None:
            waiter.set()

    def barriers(self):
        pending = []
        for dp in self.datapaths.values():
            msg = dp.ofproto_parser.OFPBarrierRequest(dp)
            dp.set_xid(msg)
            done = hub.Event()
            self.waiters[dp.id, msg.xid] = done
            pending.append((dp.id, msg.xid, done))
            dp.send_msg(msg)
        for dpid, xid, done in pending:
            if not done.wait(timeout=10):
                self.waiters.pop((dpid, xid), None)
                raise RuntimeError("OpenFlow barrier timed out")

    def install(self, routes):
        error_count = len(self.errors)
        started = time.perf_counter()
        for dp in self.datapaths.values():
            dp.send_msg(dp.ofproto_parser.OFPFlowMod(
                dp, cookie=COOKIE, cookie_mask=0xffffffffffffffff,
                command=dp.ofproto.OFPFC_DELETE, out_port=dp.ofproto.OFPP_ANY,
                out_group=dp.ofproto.OFPG_ANY))
        self.barriers()
        count = 0
        for flow in routes:
            for step in flow["outputs"]:
                dp = self.datapaths[int(step["dpid"])]
                p, ofp = dp.ofproto_parser, dp.ofproto
                match = p.OFPMatch(eth_type=0x0800, ip_proto=17,
                                   ipv4_src=flow["src_ip"], ipv4_dst=flow["dst_ip"],
                                   udp_dst=UDP_PORT)
                actions = [p.OFPActionOutput(int(step["port"]))]
                dp.send_msg(p.OFPFlowMod(dp, cookie=COOKIE, priority=100, match=match,
                                        instructions=[p.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]))
                count += 1
        self.barriers()
        if len(self.errors) != error_count:
            raise RuntimeError("Switch rejected flow operation: " + str(self.errors[error_count:]))
        return {"installation_ms": (time.perf_counter() - started) * 1000, "rules": count}


class API(ControllerBase):
    def __init__(self, req, link, data, **config):
        super(API, self).__init__(req, link, data, **config)
        self.app = data["app"]

    @route("status", "/status", methods=["GET"])
    def status(self, req, **kwargs):
        return self.reply({"pid": os.getpid(), "dpids": sorted(self.app.datapaths), "errors": self.app.errors})

    @staticmethod
    def reply(value, status=200):
        return Response(body=json.dumps(value, allow_nan=False).encode(), status=status,
                        content_type="application/json")

    @route("decision", "/decision", methods=["POST"])
    def decision(self, req, **kwargs):
        if self.app.busy:
            return self.reply({"error": "Another decision is in progress"}, 409)
        self.app.busy = True
        try:
            started = time.perf_counter()
            body = req.json
            if body["method"] == "shortest":
                # Candidate override measures a fixed route; it is not a model decision.
                selected = body.get("candidate", 0)
                if selected not in (0, 1, 2):
                    raise ValueError("Candidate must be 0, 1 or 2")
                result = {"selected": selected, "preparation_ms": 0, "inference_ms": 0}
            else:
                worker_request = Request("http://127.0.0.1:%d" % WORKER_PORT,
                                         data=json.dumps({"action": "select", "scenario": body["scenario"]}).encode(),
                                         headers={"Content-Type": "application/json"})
                with urlopen(worker_request, timeout=60) as response:
                    result = json.load(response)
                if result["method"] != body["method"]:
                    raise ValueError("Wrong model worker is running")
            selected = result["selected"]
            result["selection_and_worker_roundtrip_ms"] = (time.perf_counter() - started) * 1000
            result.update(self.app.install(body["candidate_rules"][selected]))
            result["controller_total_ms"] = (time.perf_counter() - started) * 1000
            return self.reply(result)
        except Exception as exc:
            self.app.logger.exception("Decision failed")
            return self.reply({"error": str(exc)}, 400)
        finally:
            self.app.busy = False
