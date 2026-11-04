"""Part D tasks 1-4: chunking, TF-IDF retrieval, grounded answering, evaluation."""
import glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root (mock_llm.py)
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm

POLICY_DIR = os.path.join("data", "policies")

def chunk_words(words, size=40, overlap=10):
    step, out, start = size - overlap, [], 0
    while True:
        out.append(words[start:start + size])
        if start + size >= len(words):      # this chunk reaches/passes the last word -> stop
            return out
        start += step

def build_chunks(size=40, overlap=10):
    chunks = []
    for path in sorted(glob.glob(os.path.join(POLICY_DIR, "*.txt"))):
        words = open(path, encoding="utf-8").read().split()
        for c in chunk_words(words, size, overlap):
            chunks.append({"source": os.path.basename(path), "text": " ".join(c)})
    return chunks

class Index:
    def __init__(self, size=40, overlap=10):
        self.chunks = build_chunks(size, overlap)
        self.vec = TfidfVectorizer()
        self.matrix = self.vec.fit_transform([c["text"] for c in self.chunks])
    def retrieve(self, query, k=3):
        sims = cosine_similarity(self.vec.transform([query]), self.matrix)[0]
        top = sims.argsort()[::-1][:k]
        return [{"source": self.chunks[i]["source"], "score": float(sims[i]), "text": self.chunks[i]["text"]} for i in top]

_default = Index()
def retrieve(query, k=3):
    return _default.retrieve(query, k)

SYSTEM = ("You are a POLICY ASSISTANT for Instamart support. Answer ONLY from the supplied CONTEXT. "
          "If the context does not contain the answer, reply that you could not find that information in the policy documents.")

def answer(question, index=None):
    hits = (index or _default).retrieve(question, 3)
    ctx = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    user = f"CONTEXT:\n{ctx}\nQUESTION:\n{question}"
    r = mock_llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
    return r.choices[0].message.content, [h["source"] for h in hits]

def evaluate(index, label):
    ev = pd.read_csv("data/raw/policy_eval.csv", keep_default_na=False)
    doc_ok = fact_ok = fb_ok = 0; bad = []
    ans_q = ev[ev.expected_doc != "NONE"]
    for r in ev.itertuples():
        text, srcs = answer(r.question, index)
        if r.expected_doc == "NONE":
            fb_ok += text.startswith("I could not find")
        else:
            d, f = r.expected_doc in srcs, r.expected_fact in text
            doc_ok += d; fact_ok += f
            if not (d and f):
                bad.append((r.question, d, f, text))
    print(f"[{label}] chunks={len(index.chunks)} doc-in-top3={doc_ok}/{len(ans_q)} fact-in-answer={fact_ok}/{len(ans_q)} "
          f"fallback={fb_ok}/{len(ev) - len(ans_q)}")
    for q, d, f, t in bad:
        print(f"   FAIL: {q} | doc_ok={d} fact_ok={f} | answer: {t}")

if __name__ == "__main__":
    chunks = _default.chunks
    print("total chunks:", len(chunks), pd.Series([c["source"] for c in chunks]).value_counts().sort_index().to_dict())
    print("query 'refund policy late delivery' ->", [h["source"] for h in retrieve("refund policy late delivery")])
    evaluate(_default, "overlap 10")
    evaluate(Index(40, 0), "overlap 0")
