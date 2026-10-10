"""
MiniEval memory agent: a small personal assistant whose memory writes are
checked by MiniEval before anything is saved.

How it works (this is the whole idea):
  1. You chat. An LLM decides, on its own, whether something you said is worth
     remembering, and if so calls the tool `save_memory(fact)`.
  2. The tool does NOT save straight away. It sends the fact to MiniEval's
     gate, together with the user's actual message as the source. The model
     never supplies the source, so it cannot invent evidence for its own fact.
  3. The gate answers STORE / REVIEW / REJECT. Only STORE is saved. REVIEW goes
     to a queue a human clears. Every decision is written to a hash-chained
     audit log.
  4. The tool result goes back to the model, so it can tell you honestly that
     it did not save something and why.

Run:
    pip install minieval-pro openai
    set LLM_BASE_URL / LLM_API_KEY / LLM_MODEL  (any OpenAI-compatible API)
    python agent.py            # interactive chat
    python agent.py --demo     # scripted walkthrough, Enter between steps
"""

import json
import os
import sys
from pathlib import Path

from minieval_pro.gate import MemoryGate
from minieval_pro.persistence import AuditLog

MEMORY_FILE = Path(os.environ.get("AGENT_MEMORY_FILE", "agent_memory.json"))
AUDIT_FILE = Path(os.environ.get("AGENT_AUDIT_FILE", "agent_audit.jsonl"))
MAX_STEPS = 4  # tool-call rounds per user message, so a loop can never run away

SYSTEM_PROMPT = (
    "You are a personal assistant with a long-term memory.\n"
    "- When the user tells you a lasting fact about THEMSELVES, call save_memory "
    "with one short third-person sentence, e.g. 'The user is vegetarian.'\n"
    "- Do not save facts about other people as if they were about the user.\n"
    "- Before answering questions about the user, call recall.\n"
    "- save_memory can refuse. Only say you remembered something if the tool "
    "result says STORED. If it says HELD FOR REVIEW or REJECTED, tell the user "
    "plainly that it was not saved and why."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Propose a lasting fact about the user for long-term memory. "
                           "It is checked before it is saved and may be refused.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": "One short third-person sentence about the user.",
                    }
                },
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Return everything currently remembered about the user.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


class Memory:
    """Stored facts plus the human review queue, kept in one small JSON file."""

    def __init__(self, path=MEMORY_FILE):
        self.path = Path(path)
        self.stored, self.review = [], []
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.stored = data.get("stored", [])
            self.review = data.get("review", [])

    def save(self):
        self.path.write_text(
            json.dumps({"stored": self.stored, "review": self.review},
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


class Agent:
    def __init__(self, chat_fn, gate=None, memory=None, audit=None):
        self.chat = chat_fn
        self.gate = gate or MemoryGate(quiet=True)
        self.memory = memory or Memory()
        self.audit = audit or AuditLog(AUDIT_FILE)
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.last_user_text = ""

    # -- the gate sits here ------------------------------------------------
    def propose(self, fact, source):
        decision = self.gate.check(source=source, fact=fact)
        self.audit.record(decision)
        entry = {
            "fact": fact,
            "source": source,
            "verdict": decision.verdict,
            "reason": getattr(decision, "reason", ""),
            "evidence": getattr(decision, "evidence_span", None),
        }
        if decision.verdict == "STORE":
            self.memory.stored.append(entry)
        elif decision.verdict == "REVIEW":
            self.memory.review.append(entry)
        self.memory.save()
        return entry

    # -- tools the model can call -------------------------------------------
    def run_tool(self, name, args):
        if name == "save_memory":
            fact = (args.get("fact") or "").strip()
            if not fact:
                return "error: empty fact"
            # The SOURCE is always the user's own message, never model-written.
            e = self.propose(fact, self.last_user_text)
            if e["verdict"] == "STORE":
                return "STORED: " + fact
            if e["verdict"] == "REVIEW":
                return "HELD FOR REVIEW (not saved): " + e["reason"]
            return "REJECTED (not saved): " + e["reason"]
        if name == "recall":
            facts = [m["fact"] for m in self.memory.stored]
            return json.dumps(facts) if facts else "no stored memories"
        return "error: unknown tool " + str(name)

    # -- one user message, possibly several tool rounds ---------------------
    def turn(self, user_text):
        self.last_user_text = user_text
        self.messages.append({"role": "user", "content": user_text})
        for _ in range(MAX_STEPS):
            msg = self.chat(self.messages, TOOLS)
            calls = getattr(msg, "tool_calls", None) or []
            assistant = {"role": "assistant", "content": msg.content or ""}
            if calls:
                assistant["tool_calls"] = [
                    {"id": c.id, "type": "function",
                     "function": {"name": c.function.name,
                                  "arguments": c.function.arguments}}
                    for c in calls
                ]
            self.messages.append(assistant)
            if not calls:
                return msg.content or ""
            for c in calls:
                try:
                    args = json.loads(c.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = self.run_tool(c.function.name, args)
                print("  [tool] %s(%s) -> %s" % (c.function.name, args, result))
                self.messages.append(
                    {"role": "tool", "tool_call_id": c.id, "content": result})
        return "(stopped: too many tool steps)"

    # -- slash commands (also what the demo uses) ---------------------------
    def handle(self, line):
        line = line.strip()
        if not line.startswith("/"):
            return self.turn(line)
        cmd, _, rest = line.partition(" ")
        if cmd == "/memory":
            return "\n".join("  - " + m["fact"] for m in self.memory.stored) or "  (empty)"
        if cmd == "/review":
            if not self.memory.review:
                return "  (review queue empty)"
            out = []
            for i, m in enumerate(self.memory.review):
                out.append("  [%d] %s\n      why: %s\n      source: \"%s\"" % (
                    i, m["fact"], m["reason"], m["source"]))
                if m.get("evidence"):
                    out.append("      evidence: \"%s\"" % m["evidence"])
            return "\n".join(out)
        if cmd in ("/approve", "/reject"):
            try:
                m = self.memory.review.pop(int(rest))
            except (ValueError, IndexError):
                return "  usage: %s <number from /review>" % cmd
            approved = cmd == "/approve"
            if approved:
                self.memory.stored.append(m)
            self.memory.save()
            self.audit.record({
                "verdict": "HUMAN_APPROVE" if approved else "HUMAN_REJECT",
                "fact": m["fact"], "source": m["source"],
                "reason": "Decided by a person from the review queue.",
            })
            return "  %s: %s" % ("approved" if approved else "rejected", m["fact"])
        if cmd == "/propose":
            # Simulates a faulty extraction: the fact is proposed directly,
            # judged against the user's last message.
            e = self.propose(rest.strip(), self.last_user_text)
            return "  gate says %s: %s" % (e["verdict"], e["reason"])
        if cmd == "/audit":
            ok, rep = self.audit.verify_chain()
            return "  chain %s: %d/%d verified, broken=%s, compromised=%s" % (
                "INTACT" if ok else "BROKEN", rep["verified"], rep["total"],
                rep["broken_at"], rep["compromised"])
        return "  commands: /memory /review /approve N /reject N /propose <fact> /audit /quit"


def make_chat():
    from openai import OpenAI  # imported here so tests need no openai package

    client = OpenAI(base_url=os.environ["LLM_BASE_URL"],
                    api_key=os.environ["LLM_API_KEY"])
    model = os.environ["LLM_MODEL"]

    def chat(messages, tools):
        r = client.chat.completions.create(
            model=model, messages=messages, tools=tools, temperature=0)
        return r.choices[0].message

    return chat


DEMO_STEPS = [
    "I'm vegetarian and I live in Pune.",
    "My brother is a lawyer and he helped me with a contract.",
    "/propose The user is a lawyer.",   # simulates a sloppy extractor
    "/review",
    "What do you know about me?",
    "/audit",
]


def main():
    demo = "--demo" in sys.argv
    agent = Agent(make_chat())
    print("MiniEval memory agent. Type /quit to exit, /help for commands.\n")
    steps = iter(DEMO_STEPS) if demo else None
    while True:
        try:
            if demo:
                line = next(steps, None)
                if line is None:
                    break
                input("[Enter] ")
                print("you> " + line)
            else:
                line = input("you> ")
        except (EOFError, KeyboardInterrupt):
            break
        if line.strip() in ("/quit", "/exit"):
            break
        if not line.strip():
            continue
        try:
            print(agent.handle(line) + "\n")
        except Exception as e:  # network/model errors must never crash the chat
            print("  (model call failed: %s)\n" % e)


if __name__ == "__main__":
    main()