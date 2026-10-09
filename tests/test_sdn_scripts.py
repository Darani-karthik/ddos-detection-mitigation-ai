"""The Mininet / Ryu scripts need Linux, Mininet and Ryu to run, so here we only check that they parse."""
import ast
from pathlib import Path

import pytest

SDN = Path(__file__).resolve().parent.parent / "sdn"


@pytest.mark.parametrize("script", sorted(SDN.glob("*.py")), ids=lambda p: p.name)
def test_sdn_script_is_valid_python(script):
    ast.parse(script.read_text(encoding="utf-8"), filename=str(script))


def test_the_controller_app_uses_the_shared_policy_and_runtime():
    source = (SDN / "ryu_ddos_controller.py").read_text(encoding="utf-8")
    for name in ("FlowTracker", "OnlineDetector", "MitigationController"):
        assert name in source
