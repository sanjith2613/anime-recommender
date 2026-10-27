"""Step 2 - Build the models and evaluate them.

  (a) Content-based: sentence embeddings (all-MiniLM-L6-v2) of title + genres + themes + synopsis.
      Also a TF-IDF version for comparison.
  (b) Collaborative filtering: implicit ALS (matrix factorization) on MyAnimeList user ratings.
  (c) Hybrid: weighted blend of (a) and (b).

Evaluation: for every user we hide a random 20% of their ratings. Each model sees the other 80% and
must produce a top-10 list. A hit = a hidden anime the user rated 8/10 or higher.
Users are split in half: 'validation' users are used to pick settings (ALS size, blend weight),
'test' users give the final numbers in results/metrics.json.
"""
import json
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from recommender import ART, EMB_MODEL, content_text, normalize, zscore

SEED = 42
K = 10
LIKE = 8
rng = np.random.default_rng(SEED)
os.makedirs(ART, exist_ok=True)
os.makedirs("results", exist_ok=True)

df = pd.read_parquet("data/anime.parquet")
ratings = pd.read_parquet("data/ratings.parquet")
idx = {a: i for i, a in enumerate(df["anime_id"])}
n_items = len(df)

# ---------------------------------------------------------------- (a) content embeddings
texts = df.apply(content_text, axis=1).tolist()
emb_path = os.path.join(ART, "content_emb.npy")
if not os.path.exists(emb_path):
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(EMB_MODEL)
    emb = st.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    np.save(emb_path, emb.astype(np.float16))
emb = np.load(emb_path).astype(np.float32)

tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), stop_words="english", min_df=2, sublinear_tf=True)
tf = normalize(tfidf.fit_transform(texts).toarray().astype(np.float32))

# ---------------------------------------------------------------- split
ratings = ratings.assign(item=ratings["anime_id"].map(idx))
users = ratings["user_id"].unique()
uidx = {u: i for i, u in enumerate(users)}
ratings["u"] = ratings["user_id"].map(uidx)
mask = rng.random(len(ratings)) < 0.2
train, test = ratings[~mask], ratings[mask]
perm = rng.permutation(len(users))
val_users, test_users = perm[: len(users) // 2], perm[len(users) // 2:]

R = sp.csr_matrix(((train["rating"] - 4).astype(np.float32), (train["u"], train["item"])),
                  shape=(len(users), n_items))  # confidence: 5->1 ... 10->6
seen = [set() for _ in users]
for u, i in zip(train["u"], train["item"]):
    seen[u].add(i)
liked_test = [set() for _ in users]
for u, i, r in zip(test["u"], test["item"], test["rating"]):
    if r >= LIKE:
        liked_test[u].add(i)
pop = np.bincount(train["item"], minlength=n_items).astype(np.float32)


def metrics(score_fn, user_list):
    P, N, Rc = [], [], []
    disc = 1 / np.log2(np.arange(2, K + 2))
    for u in user_list:
        rel = liked_test[u]
        if not rel:
            continue
        s = score_fn(u).copy()
        s[list(seen[u])] = -np.inf
        top = np.argpartition(-s, K)[:K]
        top = top[np.argsort(-s[top])]
        hits = np.array([t in rel for t in top], dtype=float)
        P.append(hits.mean())
        Rc.append(hits.sum() / min(len(rel), K))
        N.append((hits * disc).sum() / disc[: min(len(rel), K)].sum())
    return {"precision@10": float(np.mean(P)), "recall@10": float(np.mean(Rc)), "ndcg@10": float(np.mean(N)),
            "users": len(P)}


def profile(matrix, u):
    row = R.getrow(u)
    w = row.data - row.data.mean()          # above-average ratings pull, below-average push
    if np.allclose(w, 0):
        w = np.ones_like(row.data)
    return matrix @ (matrix[row.indices].T @ w)


# ---------------------------------------------------------------- (b) ALS
from implicit.als import AlternatingLeastSquares
import threadpoolctl
threadpoolctl.threadpool_limits(1, "blas")


def fit_als(factors, reg, alpha):
    m = AlternatingLeastSquares(factors=factors, regularization=reg, alpha=alpha, iterations=20, random_state=SEED)
    m.fit(R, show_progress=False)
    return np.asarray(m.user_factors), np.asarray(m.item_factors)


print("tuning ALS on validation users...")
best = None
for f in (32, 64, 128):
    for reg in (0.05, 0.5):
        for alpha in (1.0, 5.0):
            U, V = fit_als(f, reg, alpha)
            m = metrics(lambda u: V @ U[u], val_users)
            print(f"  factors={f} reg={reg} alpha={alpha} ndcg={m['ndcg@10']:.4f}")
            if best is None or m["ndcg@10"] > best[0]:
                best = (m["ndcg@10"], f, reg, alpha, U, V)
_, F, REG, ALPHA, U, V = best
print("best ALS:", F, REG, ALPHA)

# ---------------------------------------------------------------- (c) hybrid weight
cache_c, cache_cf = {}, {}


def content_score(u):
    if u not in cache_c:
        cache_c[u] = zscore(profile(emb, u))
    return cache_c[u]


def cf_score(u):
    if u not in cache_cf:
        cache_cf[u] = zscore(V @ U[u])
    return cache_cf[u]


best_w = None
for w in np.linspace(0, 1, 11):
    m = metrics(lambda u: w * content_score(u) + (1 - w) * cf_score(u), val_users)
    print(f"  hybrid w_content={w:.1f} ndcg={m['ndcg@10']:.4f}")
    if best_w is None or m["ndcg@10"] > best_w[0]:
        best_w = (m["ndcg@10"], float(w))
W = best_w[1]
print("best hybrid content weight:", W)

# ---------------------------------------------------------------- final test numbers
models = {
    "Most popular (baseline)": lambda u: pop.copy(),
    "Content: TF-IDF": lambda u: profile(tf, u),
    "Content: sentence embeddings": content_score,
    f"Collaborative filtering: ALS ({F} factors)": cf_score,
    f"Hybrid ({W:.1f} content + {1 - W:.1f} ALS)": lambda u: W * content_score(u) + (1 - W) * cf_score(u),
}
results = {}
for name, fn in models.items():
    results[name] = metrics(fn, test_users)
    print(f"{name:45s} " + " ".join(f"{k}={v:.4f}" for k, v in results[name].items() if k != "users"))

# ---------------------------------------------------------------- save artifacts for the app
# Final ALS trained on ALL ratings for the app
R = sp.csr_matrix(((ratings["rating"] - 4).astype(np.float32), (ratings["u"], ratings["item"])),
                  shape=(len(users), n_items))
_, V_all = fit_als(F, REG, ALPHA)
np.save(os.path.join(ART, "cf_item_factors.npy"), V_all.astype(np.float16))
np.save(os.path.join(ART, "cf_support.npy"), np.bincount(ratings["item"], minlength=n_items).astype(np.int32))
# In the app we only have 3-5 picked titles, not a full history; content gets a little more weight there
# because CF item vectors from 812 users are noisier than for a full user profile.
cfg = {"hybrid_weight_content": max(W, 0.5), "eval_best_weight_content": W,
       "als": {"factors": F, "regularization": REG, "alpha": ALPHA}}
json.dump(cfg, open(os.path.join(ART, "config.json"), "w"), indent=2)

summary = {
    "protocol": "per-user random 20% holdout; relevant = held-out anime rated >= 8; settings tuned on half of "
                "the users, numbers below are on the other half; already-seen titles excluded",
    "data": {"anime": n_items, "ratings": int(len(ratings)), "users": int(len(users))},
    "als": cfg["als"], "hybrid_weight_content": W, "results": results,
}
json.dump(summary, open("results/metrics.json", "w"), indent=2)
with open("results/metrics.md", "w") as f:
    f.write("| Model | Precision@10 | Recall@10 | NDCG@10 |\n|---|---|---|---|\n")
    for n, r in results.items():
        f.write(f"| {n} | {r['precision@10']:.3f} | {r['recall@10']:.3f} | {r['ndcg@10']:.3f} |\n")
print(open("results/metrics.md").read())
