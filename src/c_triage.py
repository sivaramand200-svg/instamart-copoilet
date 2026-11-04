"""Part C: support-ticket triage with guardrails, redaction, retries, validation, budget and routing."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root (mock_llm.py)
import json, random, re, time
import pandas as pd
import mock_llm
from mock_llm import est_tokens, TransientLLMError

MAX_INPUT_TOKENS = 400
BASE_DELAY, MAX_RETRIES = 0.05, 3
ALLOWED = ["billing", "delivery", "technical", "fraud"]
PII = [("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
       ("PHONE", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
       ("ORDER_ID", re.compile(r"ORD-\d{6}"), "[ORDER_ID]")]

ZERO_SHOT = ('You CLASSIFY customer support tickets for a quick-commerce app. '
             'Allowed categories: billing, delivery, technical, fraud. '
             'Reply with JSON only, in exactly this form: {"category": "...", "reason": "..."}.')
FEW_SHOT = ZERO_SHOT + '\nEXAMPLES:\n' \
    'Ticket: I was charged twice for [ORDER_ID]. -> {"category": "billing", "reason": "double charge"}\n' \
    'Ticket: The rider never arrived with [ORDER_ID]. -> {"category": "delivery", "reason": "order not received"}\n' \
    'Ticket: The app freezes when I apply a coupon. -> {"category": "technical", "reason": "app bug"}\n' \
    'Ticket: There is an unauthorized transaction on my card. -> {"category": "fraud", "reason": "unauthorized payment"}'

def redact(text, counts=None):
    for name, pat, ph in PII:
        text, n = pat.subn(ph, text)
        if counts is not None:
            counts[name] += n
    return text

def call_with_retry(messages, ticket_id, stats, log):
    for attempt in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)
        except TransientLLMError:
            stats["transient"] += 1
            if attempt == MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** attempt * random.uniform(0.9, 1.15)
            log.append((ticket_id, attempt, wait))
            print(f"  retry ticket={ticket_id} retry_no={attempt} wait={wait:.4f}s")
            time.sleep(wait)

def validate(content):
    try:
        d = json.loads(content)
        if isinstance(d, dict) and d.get("category") in ALLOWED:
            return "classified_json", d["category"]
        return "flagged_invalid_output", ""
    except (json.JSONDecodeError, TypeError):
        found = {w for w in re.findall(r"[a-z]+", (content or "").lower()) if w in ALLOWED}
        if len(found) == 1:
            return "classified_keyword", next(iter(found))
        return "flagged_invalid_output", ""

def run_pipeline(tickets, system_prompt, token_budget=None, verbose=True):
    mock_llm.reset()
    random.seed(42)
    stats, log, sent, total = {"transient": 0}, [], [], 0
    rows = []
    for t in tickets.itertuples():
        text = t.text
        row = dict(ticket_id=t.ticket_id, customer_id=t.customer_id, status="", category="", tokens_used=0)
        if not text.strip():
            row["status"] = "skipped_empty"
        elif est_tokens(text) > MAX_INPUT_TOKENS:
            row["status"] = "flagged_long"
        elif token_budget is not None and total >= token_budget:
            row["status"] = "budget_exhausted"
        else:
            msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": redact(text)}]
            sent.append(msgs)
            r = call_with_retry(msgs, t.ticket_id, stats, log)
            used = r.usage.total_tokens; total += used; row["tokens_used"] = used
            row["status"], row["category"] = validate(r.choices[0].message.content)
        rows.append(row)
    return pd.DataFrame(rows), dict(stats=stats, log=log, sent=sent, total=total)

def classified(df):
    return df[df.status.isin(["classified_json", "classified_keyword"])]

def route(df, seg_path="outputs/customer_segments.csv"):
    seg = pd.read_csv(seg_path).set_index("customer_id")
    out = []
    for r in df.itertuples():
        s = seg.loc[r.customer_id] if r.customer_id in seg.index else None
        seg_name = s["segment_name"] if s is not None else ""
        prem = bool(s["is_premium"]) if s is not None else False
        if r.status in ("skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted") or r.status not in ("classified_json", "classified_keyword"):
            rt, pr = "human_review", "NORMAL"
        elif r.category == "fraud":
            rt, pr = "security_team", "HIGH"
        elif prem:
            rt, pr = "premium_desk", "HIGH"
        else:
            rt, pr = "support_queue", "NORMAL"
        out.append(dict(ticket_id=r.ticket_id, customer_id=r.customer_id, status=r.status, category=r.category or "",
                        segment_name=seg_name, priority=pr, route=rt, tokens_used=r.tokens_used))
    return pd.DataFrame(out)

if __name__ == "__main__":
    tk = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)
    print("tickets:", len(tk))
    # ---- guardrail + redaction audit over all 164 tickets
    print("empty:", int((tk.text.str.strip() == "").sum()),
          "long:", int(tk.text.map(lambda x: est_tokens(x) > MAX_INPUT_TOKENS).sum()),
          "long token estimate:", sorted({est_tokens(x) for x in tk.text if est_tokens(x) > MAX_INPUT_TOKENS}))
    counts = {"EMAIL": 0, "PHONE": 0, "ORDER_ID": 0}
    red = [redact(x, counts) for x in tk.text]
    print("substitutions:", counts, "total:", sum(counts.values()))
    left = sum(len(p.findall(x)) for x in red for _, p, _ in PII)
    print("patterns found in redacted texts:", left); assert left == 0
    for p in (ZERO_SHOT, FEW_SHOT):
        assert "CLASSIFY" in p and all(c in p for c in ALLOWED)
    assert "EXAMPLES:" not in ZERO_SHOT and "EXAMPLES:" in FEW_SHOT and "fraud" in FEW_SHOT
    print("prompt est tokens: zero-shot", est_tokens(ZERO_SHOT), "few-shot", est_tokens(FEW_SHOT))

    results = {}
    for name, prompt in (("zero-shot", ZERO_SHOT), ("few-shot", FEW_SHOT)):
        print(f"\n=== full run: {name}")
        df, info = run_pipeline(tk, prompt)
        results[name] = (df, info)
        for msgs in info["sent"]:          # privacy assertion on EVERY prompt message
            for m in msgs:
                for _, pat, _ in PII:
                    assert not pat.search(m["content"]), "PII reached the model"
            assert msgs[-1]["content"] == redact(msgs[-1]["content"])
        calls = len(info["sent"])
        assert calls == 156
        ok = classified(df)
        acc = (ok.category == tk.set_index("ticket_id").loc[ok.ticket_id, "true_category"].values).mean()
        waits = [w for _, _, w in info["log"]]
        print(f"model calls={calls} transient errors={info['stats']['transient']} retries logged={len(waits)} "
              f"wait range={min(waits):.4f}-{max(waits):.4f}")
        print("status counts:", df.status.value_counts().to_dict())
        print(f"classified={len(ok)} accuracy={acc*100:.1f}% ({int(round(acc*len(ok)))} of {len(ok)}) total tokens={info['total']} "
              f"tokens/classified ticket={info['total']/len(ok):.1f}")
        if name == "zero-shot":
            truth = tk.set_index("ticket_id").loc[ok.ticket_id, "true_category"].values
            print(pd.crosstab(pd.Series(truth, name="true"), pd.Series(ok.category.values, name="predicted")).to_string())

    # ---- budget demo (zero-shot)
    for budget in (6000, 5000):
        bdf, binfo = run_pipeline(tk, ZERO_SHOT, token_budget=budget)
        n = int((bdf.status == "budget_exhausted").sum())
        print(f"\nbudget demo TOKEN_BUDGET={budget}: budget_exhausted={n}, tokens spent={binfo['total']}")
        if n > 0:
            route(bdf).to_csv("outputs/triage_budget_demo.csv", index=False); break
    # ---- final routed output (few-shot, big budget)
    fdf, finfo = run_pipeline(tk, FEW_SHOT, token_budget=100000)
    final = route(fdf); final.to_csv("outputs/triage_results.csv", index=False)
    print("\nfinal rows:", len(final)); print(final.route.value_counts().to_dict()); print(final.priority.value_counts().to_dict())
    print(pd.crosstab(final.route, final.status).to_string())
