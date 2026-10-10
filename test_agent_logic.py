"""
Tests the agent loop with a scripted fake model and a stand-in gate, so they
run in a second with no network and no models. They check OUR logic: that only
STORE is saved, that the source is the user's own message, that the loop is
bounded, and that the audit chain stays intact.
"""
import json
from types import SimpleNamespace as NS

import pytest

import agent as agent_mod
from agent import Agent, Memory
from minieval_pro.persistence import AuditLog


class StubGate:
    """REVIEW if the fact says 'lawyer' but the source says 'brother'."""
    def __init__(self):
        self.calls = []

    def check(self, source, fact):
        self.calls.append((source, fact))
        bad = "lawyer" in fact.lower() and "brother" in source.lower()
        d = NS(verdict="REVIEW" if bad else "STORE",
               reason="third-party attribution" if bad else "supported",
               evidence_span=source if bad else None)
        d.to_dict = lambda d=d, s=source, f=fact: {
            "verdict": d.verdict, "fact": f, "source": s, "reason": d.reason}
        return d


def tool_call(name, **args):
    return NS(content="", tool_calls=[NS(
        id="c1", function=NS(name=name, arguments=json.dumps(args)))])


def text(t):
    return NS(content=t, tool_calls=None)


def make(tmp_path, replies):
    it = iter(replies)
    gate = StubGate()
    a = Agent(lambda m, t: next(it), gate=gate,
              memory=Memory(tmp_path / "m.json"),
              audit=AuditLog(tmp_path / "a.jsonl"))
    return a, gate


def test_store_is_saved_and_audited(tmp_path):
    a, _ = make(tmp_path, [tool_call("save_memory", fact="The user is vegetarian."),
                           text("Noted.")])
    assert a.turn("I'm vegetarian.") == "Noted."
    assert [m["fact"] for m in a.memory.stored] == ["The user is vegetarian."]
    assert a.audit.count() == 1


def test_review_is_not_saved_and_model_is_told(tmp_path):
    a, _ = make(tmp_path, [tool_call("save_memory", fact="The user is a lawyer."),
                           text("I didn't save that.")])
    a.turn("My brother is a lawyer.")
    assert a.memory.stored == []
    assert len(a.memory.review) == 1
    tool_msg = [m for m in a.messages if m["role"] == "tool"][0]["content"]
    assert tool_msg.startswith("HELD FOR REVIEW (not saved)")


def test_source_is_the_users_message_not_model_text(tmp_path):
    a, gate = make(tmp_path, [tool_call("save_memory", fact="The user is a lawyer."),
                              text("ok")])
    a.turn("My brother is a lawyer.")
    assert gate.calls[0][0] == "My brother is a lawyer."


def test_recall_returns_only_stored(tmp_path):
    a, _ = make(tmp_path, [tool_call("save_memory", fact="The user is vegetarian."),
                           text("ok"),
                           tool_call("save_memory", fact="The user is a lawyer."),
                           text("ok"),
                           tool_call("recall"), text("done")])
    a.turn("I'm vegetarian.")
    a.turn("My brother is a lawyer.")
    a.turn("What do you know?")
    recall_msg = [m for m in a.messages if m["role"] == "tool"][-1]["content"]
    assert json.loads(recall_msg) == ["The user is vegetarian."]


def test_tool_loop_is_bounded(tmp_path):
    a, _ = make(tmp_path, [tool_call("recall")] * 50)
    assert a.turn("hi").startswith("(stopped")


def test_approve_moves_item_and_is_audited(tmp_path):
    a, _ = make(tmp_path, [tool_call("save_memory", fact="The user is a lawyer."),
                           text("ok")])
    a.turn("My brother is a lawyer.")
    a.handle("/approve 0")
    assert [m["fact"] for m in a.memory.stored] == ["The user is a lawyer."]
    assert a.memory.review == []
    assert [e["verdict"] for e in a.audit.entries()][-1] == "HUMAN_APPROVE"


def test_audit_chain_intact_after_a_session(tmp_path):
    a, _ = make(tmp_path, [tool_call("save_memory", fact="The user is vegetarian."),
                           text("ok")])
    a.turn("I'm vegetarian.")
    a.handle("/propose The user is a lawyer.")
    ok, rep = a.audit.verify_chain()
    assert ok and rep["total"] == 2


def test_propose_command_uses_last_user_message(tmp_path):
    a, gate = make(tmp_path, [text("hi")])
    a.turn("My brother is a lawyer.")
    out = a.handle("/propose The user is a lawyer.")
    assert "REVIEW" in out and gate.calls[-1][0] == "My brother is a lawyer."