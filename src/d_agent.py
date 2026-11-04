"""Part D tasks 5-8: tools, ReAct loop, safety limits, resumable runs."""
import datetime, json, os, shutil, sqlite3, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root (mock_llm.py)
import pandas as pd
import mock_llm
from d_rag import retrieve

DB, LEDGER, RUNS, STATE = "data/instamart.db", "outputs/refund_ledger.jsonl", "outputs/agent_runs.jsonl", "state"
CURRENT_RUN = {"id": ""}      # run id stamped on ledger lines

# ------------------------------------------------------------------ tools
def get_order(order_id):
    con = sqlite3.connect(DB)
    row = con.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?", (order_id,)).fetchone()
    con.close()
    if row is None:
        return {"error": "order not found"}
    dm = None if row[3] is None or row[3] != row[3] else row[3]
    return {"order_id": row[0], "status": row[1], "order_amount": row[2], "delivery_minutes": dm}

def search_policy(query):
    return {"results": [{"source": h["source"], "score": round(h["score"], 4), "text": h["text"]} for h in retrieve(query, 3)]}

def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is not None and delivery_minutes > 45:
        return {"refund_inr": round(float(order_amount), 2)}
    return {"refund_inr": 0}

def _ledger(tool, order_id, refund_inr):
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": datetime.datetime.now().isoformat(timespec="seconds"), "tool": tool,
                            "run_id": CURRENT_RUN["id"], "order_id": order_id, "refund_inr": refund_inr}) + "\n")
    return {"refund_inr": refund_inr}

def issue_refund(order_id, refund_inr):
    return _ledger("issue_refund", order_id, refund_inr)

def escalate_to_human(order_id, refund_inr):
    return _ledger("escalate_to_human", order_id, refund_inr)

def _spec(name, desc, props, req):
    return {"name": name, "description": desc, "parameters": {"type": "object", "properties": props, "required": req}}
TOOLS_SPEC = [
    _spec("get_order", "Look up an order's status, amount and delivery time in the Instamart database.", {"order_id": {"type": "string"}}, ["order_id"]),
    _spec("search_policy", "Search the policy documents and return the three most relevant passages.", {"query": {"type": "string"}}, ["query"]),
    _spec("calculate_refund", "Compute the refund in INR from the order amount and delivery minutes.",
          {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "number"}}, ["order_amount", "delivery_minutes"]),
    _spec("issue_refund", "Issue an automatic refund of up to INR 1000 and record it in the ledger.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, ["order_id", "refund_inr"]),
    _spec("escalate_to_human", "Send a refund above INR 1000 to a human support lead and record it in the ledger.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, ["order_id", "refund_inr"])]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}

SYSTEM = ("You are a refund AGENT for Instamart support. Use the tools to look up the order, check the policy, "
          "compute the refund and issue or escalate it. Finish with one line: STATUS: explanation.")

class SimulatedCrash(Exception):
    pass

# ------------------------------------------------------------------ agent loop
def _checkpoint(st):
    os.makedirs(STATE, exist_ok=True)
    with open(os.path.join(STATE, f"{st['run_id']}.json"), "w", encoding="utf-8") as f:
        json.dump(st, f, indent=1, default=str)

def _finish(st, status, answer):
    st["steps_summary"] = len(st["steps"])
    return {"status": status, "final_answer": answer, "steps": st["steps"], "llm_calls": st["llm_calls"],
            "tokens_used": st["tokens_used"], "run_id": st["run_id"], "scenario_id": st["scenario_id"],
            "tools_executed": list(st["tools_executed"])}

def _handoff(st, why):
    done = ", ".join(st["tools_executed"]) or "no tools"
    return f"{why}. Handing over to a human agent. Steps completed: {done}. Tokens used: {st['tokens_used']}."

def _loop(st, crash_after_tools=None):
    start = time.monotonic()
    while st["llm_calls"] < st["max_iterations"]:
        if time.monotonic() - start >= st["time_limit_seconds"]:
            return _finish(st, "FALLBACK_TIMEOUT", _handoff(st, "Time limit reached"))
        resp = mock_llm.chat(st["messages"], tools=TOOLS_SPEC)
        st["llm_calls"] += 1
        st["tokens_used"] += resp.usage.total_tokens
        if st["tokens_used"] > st["token_limit"]:       # stop WITHOUT executing the requested tool
            return _finish(st, "FALLBACK_TOKEN_LIMIT", _handoff(st, "Token limit exceeded"))
        msg = resp.choices[0].message
        if not msg.tool_calls:
            text = msg.content
            return _finish(st, text.split(":")[0].strip(), text)
        tc = msg.tool_calls[0]
        st["messages"].append({"role": "assistant", "content": None,
                               "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}]})
        CURRENT_RUN["id"] = st["run_id"]
        obs = TOOL_MAP[tc.name](**tc.arguments)
        st["messages"].append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": json.dumps(obs)})
        st["steps"].append({"tool": tc.name, "arguments": tc.arguments, "observation": obs})
        st["tools_executed"].append(tc.name)
        st["step"] = len(st["steps"])
        _checkpoint(st)
        if crash_after_tools is not None and len(st["tools_executed"]) == crash_after_tools:
            raise SimulatedCrash(f"simulated crash after tool #{crash_after_tools} ({tc.name})")
    return _finish(st, "FALLBACK_MAX_ITERATIONS", _handoff(st, f"No final answer after {st['max_iterations']} model calls"))

def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              run_id=None, crash_after_tools=None):
    st = {"run_id": run_id or scenario_id, "scenario_id": scenario_id,
          "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message}],
          "tokens_used": 0, "step": 0, "tools_executed": [], "steps": [], "llm_calls": 0,
          "max_iterations": max_iterations, "token_limit": token_limit, "time_limit_seconds": time_limit_seconds}
    return _loop(st, crash_after_tools)

def resume_agent(run_id):
    with open(os.path.join(STATE, f"{run_id}.json"), encoding="utf-8") as f:
        st = json.load(f)
    return _loop(st)

def refund_of(res):
    for s in res["steps"]:
        if s["tool"] in ("issue_refund", "escalate_to_human"):
            return s["observation"]["refund_inr"]
    return 0

def record(res, path):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"scenario_id": res["scenario_id"], "status": res["status"], "final_answer": res["final_answer"],
                            "llm_calls": res["llm_calls"], "tokens_used": res["tokens_used"], "steps": res["steps"]}) + "\n")

if __name__ == "__main__":
    for p in (LEDGER, RUNS, "outputs/agent_demo_runs.jsonl"):
        if os.path.exists(p):
            os.remove(p)
    shutil.rmtree(STATE, ignore_errors=True); os.makedirs(STATE)
    sc = pd.read_csv("data/raw/agent_scenarios.csv")
    rows = []
    for r in sc.itertuples():
        mock_llm.reset()
        res = run_agent(r.message, r.scenario_id)
        record(res, RUNS)
        amt = refund_of(res)
        exp = r.expected_outcome
        ok = (res["status"].startswith("FALLBACK") if exp == "FALLBACK" else res["status"] == exp) and abs(amt - r.expected_refund_inr) < 0.01
        rows.append((r.scenario_id, res["status"], f"{amt:.2f}", res["llm_calls"], res["tokens_used"], r.expected_refund_inr, ok))
    print(pd.DataFrame(rows, columns=["scenario", "status", "refund_inr", "llm_calls", "tokens", "expected_refund", "match"]).to_string(index=False))

    s1 = sc.iloc[0]
    print("\n-- S1 token_limit=200")
    d1 = run_agent(s1.message, "S1", token_limit=200, run_id="S1-token-limit-demo"); record(d1, "outputs/agent_demo_runs.jsonl")
    print(d1["status"], "| calls", d1["llm_calls"], "| tokens", d1["tokens_used"], "| tools run:", d1["tools_executed"]); print(d1["final_answer"])
    print("-- S1 time_limit_seconds=0")
    d2 = run_agent(s1.message, "S1", time_limit_seconds=0, run_id="S1-timeout-demo"); record(d2, "outputs/agent_demo_runs.jsonl")
    print(d2["status"], "| calls", d2["llm_calls"], "| tools run:", d2["tools_executed"]); print(d2["final_answer"])

    print("-- S1 crash after tool #2, then resume")
    mock_llm.reset()
    try:
        run_agent(s1.message, "S1", run_id="S1-crash-demo", crash_after_tools=2)
    except SimulatedCrash as e:
        print("crashed:", e)
    ck = json.load(open(os.path.join(STATE, "S1-crash-demo.json")))
    print("checkpoint tools so far:", ck["tools_executed"], "| tokens:", ck["tokens_used"], "| step:", ck["step"])
    d3 = resume_agent("S1-crash-demo"); record(d3, "outputs/agent_demo_runs.jsonl")
    print("final:", d3["status"], refund_of(d3), "| tools executed across crash+resume:", d3["tools_executed"], "| llm calls total:", d3["llm_calls"])
    assert d3["tools_executed"] == ["get_order", "search_policy", "calculate_refund", "issue_refund"] and d3["status"] == "REFUND" and refund_of(d3) == 128.25
    print("\nledger lines:", sum(1 for _ in open(LEDGER)), "| runs lines:", sum(1 for _ in open(RUNS)), "| state files:", sorted(os.listdir(STATE)))
