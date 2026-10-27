"""Step 2 - Build the models and evaluate them.

  (a) Content-based: sentence embeddings (all-MiniLM-L6-v2, ONNX via fastembed)
      of title + genres + themes + synopsis. Also a TF-IDF version for comparison.
  (b) Collaborative filtering: implicit ALS on MyAnimeList user ratings.
  (c) Hybrid: weighted blend of (a) and (b).

Two tests, both on users the settings were not tuned on:
  * Full history: hide a random 20% of each user's ratings. The model sees the other 80%.
  * Few favorites: the model sees only 3-5 of the user's highest training ratings.
    This is how the website is actually used.
A hit is a hidden anime the user rated 8/10 or higher.
Recall@10 divides by every held-out favorite, not by 10.
"""
import json
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from recommender import ART, EMB_MODEL, content_text, hybrid_scores, normalize, zscore

if __name__ != "__main__":
    raise RuntimeError("Run this file as a script: python build_models.py")

SEED = 42
K = 10
LIKE = 8
rng = np.random.default_rng(SEED)
os.makedirs(ART, exist_ok=True)
os.makedirs("results", exist_ok=True)

df = pd.read_parquet("data/anime.parquet").reset_index(drop=True)
ratings = pd.read_parquet("data/ratings.parquet")
idx = {a: i for i, a in enumerate(df["anime_id"])}
n_items = len(df)

# ---------------------------------------------------------------- (a) content embeddings
texts = df.apply(content_text, axis=1).tolist()
emb_path = os.path.join(ART, "content_emb.npy")
meta_path = os.path.join(ART, "emb_meta.json")
emb_meta = {"model": EMB_MODEL, "n": n_items, "text": "title+genres+themes+audience+synopsis800"}
cached = {}
if os.path.exists(meta_path):
    cached = json.load(open(meta_path))
if not os.path.exists(emb_path) or cached.get("model") != emb_meta["model"] or cached.get("n") != n_items or cached.get("text") != emb_meta["text"]:
    from fastembed import TextEmbedding
    encoder = TextEmbedding(EMB_MODEL)
    vectors = [np.asarray(v, dtype=np.float32) for v in encoder.embed(texts, batch_size=64)]
    emb = normalize(np.vstack(vectors))
    np.save(emb_path, emb.astype(np.float16))
    json.dump(emb_meta, open(meta_path, "w"), indent=2)
    print(f"embedded {n_items} titles with {EMB_MODEL}")
emb = np.load(emb_path).astype(np.float32)


def sparse_normalize(matrix):
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    norms[norms == 0] = 1
    return matrix.multiply(1.0 / norms[:, None])


tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), stop_words="english", min_df=2, sublinear_tf=True)
tf = sparse_normalize(tfidf.fit_transform(texts)).tocsr()

# ---------------------------------------------------------------- split
ratings = ratings[ratings["anime_id"].isin(idx)].copy()
ratings["item"] = ratings["anime_id"].map(idx)
users = ratings["user_id"].unique()
uidx = {u: i for i, u in enumerate(users)}
ratings["u"] = ratings["user_id"].map(uidx)
mask = rng.random(len(ratings)) < 0.2
train, test = ratings[~mask].copy(), ratings[mask].copy()
perm = rng.permutation(len(users))
val_users, test_users = perm[: len(users) // 2], perm[len(users) // 2:]

R = sp.csr_matrix(((train["rating"] - 4).astype(np.float32), (train["u"], train["item"])),
                  shape=(len(users), n_items))
seen = [set() for _ in users]
for u, i in zip(train["u"].to_numpy(), train["item"].to_numpy()):
    seen[u].add(int(i))
liked_test = [set() for _ in users]
for u, i, r in zip(test["u"].to_numpy(), test["item"].to_numpy(), test["rating"].to_numpy()):
    if r >= LIKE:
        liked_test[u].add(int(i))
pop = np.bincount(train["item"].to_numpy(), minlength=n_items).astype(np.float32)


def metrics(score_fn, user_list):
    """Precision@10, Recall@10 (hits / all held-out likes), NDCG@10."""
    precision, recall, ndcg = [], [], []
    discount = 1 / np.log2(np.arange(2, K + 2))
    for u in user_list:
        relevant = liked_test[u]
        if not relevant:
            continue
        scores = np.asarray(score_fn(u), dtype=np.float32).copy()
        if seen[u]:
            scores[list(seen[u])] = -np.inf
        top = np.argpartition(-scores, K)[:K]
        top = top[np.argsort(-scores[top])]
        hits = np.array([int(t) in relevant for t in top], dtype=np.float32)
        precision.append(hits.mean())
        recall.append(hits.sum() / len(relevant))
        ideal = discount[: min(len(relevant), K)].sum()
        ndcg.append((hits * discount).sum() / ideal)
    return {
        "precision@10": float(np.mean(precision)),
        "recall@10": float(np.mean(recall)),
        "ndcg@10": float(np.mean(ndcg)),
        "users": len(precision),
    }


def profile(matrix, u):
    row = R.getrow(int(u))
    weights = row.data.astype(np.float32) - row.data.mean()
    if np.allclose(weights, 0):
        weights = np.ones_like(row.data, dtype=np.float32)
    picked = matrix[row.indices]
    if sp.issparse(picked):
        direction = np.asarray(picked.T.dot(weights)).ravel()
        return np.asarray(matrix.dot(direction)).ravel()
    return matrix @ (picked.T @ weights)


# ---------------------------------------------------------------- (b) ALS
from implicit.als import AlternatingLeastSquares
import threadpoolctl
threadpoolctl.threadpool_limits(1, "blas")


def fit_als(factors, reg, alpha, matrix):
    model = AlternatingLeastSquares(factors=factors, regularization=reg, alpha=alpha,
                                    iterations=20, random_state=SEED)
    model.fit(matrix, show_progress=False)
    return np.asarray(model.user_factors), np.asarray(model.item_factors)


print("tuning ALS on validation users...")
best = None
for factors in (32, 64, 128):
    for reg in (0.05, 0.5):
        for alpha in (1.0, 5.0):
            user_f, item_f = fit_als(factors, reg, alpha, R)
            result = metrics(lambda u, item_f=item_f, user_f=user_f: item_f @ user_f[u], val_users)
            print(f"  factors={factors} reg={reg} alpha={alpha} ndcg={result['ndcg@10']:.4f}")
            if best is None or result["ndcg@10"] > best[0]:
                best = (result["ndcg@10"], factors, reg, alpha, user_f, item_f)
_, F, REG, ALPHA, U, V = best
print("best ALS:", F, REG, ALPHA)

# ---------------------------------------------------------------- (c) hybrid weight on full histories
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
    result = metrics(lambda u, w=w: w * content_score(u) + (1 - w) * cf_score(u), val_users)
    print(f"  full-history w_content={w:.1f} ndcg={result['ndcg@10']:.4f}")
    if best_w is None or result["ndcg@10"] > best_w[0]:
        best_w = (result["ndcg@10"], round(float(w), 1))
W = best_w[1]
print("best full-history content weight:", W)

# ---------------------------------------------------------------- few-favorite protocol (mirrors the app)
support = np.bincount(train["item"].to_numpy(), minlength=n_items).astype(np.float32)
mal = df["score"].fillna(df["score"].median()).to_numpy(dtype=np.float32)
V_norm = normalize(V.astype(np.float32))
train_likes = train[train["rating"] >= LIKE]


def favorite_seeds(u):
    """3, 4, or 5 of this user's highest training ratings. Stable across runs."""
    sub = train_likes[train_likes["u"] == u]
    if len(sub) < 3:
        return None
    sub = sub.sort_values(["rating", "item"], ascending=[False, True])
    k = 3 + (int(u) + SEED) % 3  # 3, 4, or 5, fixed per user
    return sub["item"].head(min(k, len(sub))).to_numpy()


seed_cache = {}


def seeds_of(u):
    if u not in seed_cache:
        seed_cache[u] = favorite_seeds(u)
    return seed_cache[u]


def app_score(u, w):
    seeds = seeds_of(u)
    if seeds is None:
        return np.full(n_items, -np.inf)
    return hybrid_scores(emb, V_norm, support, mal, seeds, w)


def few_metrics(score_fn, user_list):
    usable = [u for u in user_list if seeds_of(u) is not None and liked_test[u]]
    return metrics(score_fn, usable)


print("tuning the app blend on 3-5 favorites...")
best_few = None
for w in np.linspace(0, 1, 11):
    result = few_metrics(lambda u, w=w: app_score(u, w), val_users)
    print(f"  few-seed w_content={w:.1f} ndcg={result['ndcg@10']:.4f} p={result['precision@10']:.4f} r={result['recall@10']:.4f}")
    if best_few is None or result["ndcg@10"] > best_few[0]:
        best_few = (result["ndcg@10"], round(float(w), 1))
W_FEW = best_few[1]
print("best few-seed content weight:", W_FEW)

# The slider matches the full-history evaluation. The few-seed weight is reported beside it.
APP_W = W

# ---------------------------------------------------------------- final test numbers
full_models = {
    "Most popular (baseline)": lambda u: pop.copy(),
    "Content: TF-IDF": lambda u: profile(tf, u),
    "Content: sentence embeddings": content_score,
    f"Collaborative filtering: ALS ({F} factors)": cf_score,
    f"Hybrid ({W:.1f} content + {1 - W:.1f} ALS)": lambda u: W * content_score(u) + (1 - W) * cf_score(u),
}
results = {}
for name, fn in full_models.items():
    results[name] = metrics(fn, test_users)
    print(f"{name:48s} " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in results[name].items()))

few_models = {
    "Most popular (baseline)": lambda u: pop.copy(),
    "Content: sentence embeddings": lambda u: app_score(u, 1.0),
    f"Collaborative filtering: ALS ({F} factors)": lambda u: app_score(u, 0.0),
    f"Hybrid ({W_FEW:.1f} content + {1 - W_FEW:.1f} ALS)": lambda u: app_score(u, W_FEW),
    f"Hybrid at full-history weight ({W:.1f})": lambda u: app_score(u, W),
}
few_results = {}
for name, fn in few_models.items():
    few_results[name] = few_metrics(fn, test_users)
    print(f"few {name:44s} " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in few_results[name].items()))

# ---------------------------------------------------------------- save artifacts for the app
# Final ALS trained on ALL ratings (not just the training split).
R_all = sp.csr_matrix(((ratings["rating"] - 4).astype(np.float32), (ratings["u"], ratings["item"])),
                      shape=(len(users), n_items))
_, V_all = fit_als(F, REG, ALPHA, R_all)
np.save(os.path.join(ART, "cf_item_factors.npy"), V_all.astype(np.float16))
np.save(os.path.join(ART, "cf_support.npy"), np.bincount(ratings["item"].to_numpy(), minlength=n_items).astype(np.int32))

likes = ratings[ratings["rating"] >= LIKE]
like_users = likes["u"].unique()
user_remap = {u: i for i, u in enumerate(like_users)}
lists = [[] for _ in range(n_items)]
for item, user in zip(likes["item"].to_numpy(), likes["u"].to_numpy()):
    lists[int(item)].append(user_remap[int(user)])
indptr = [0]
flat = []
for liked_by in lists:
    flat.extend(sorted(set(liked_by)))
    indptr.append(len(flat))
np.savez_compressed(
    os.path.join(ART, "cf_likers.npz"),
    users=np.asarray(flat, dtype=np.int32),
    indptr=np.asarray(indptr, dtype=np.int32),
    n_users=np.int32(len(like_users)),
)

cfg = {
    "hybrid_weight_content": APP_W,
    "eval_best_weight_content": W,
    "few_seed_best_weight_content": W_FEW,
    "als": {"factors": int(F), "regularization": float(REG), "alpha": float(ALPHA)},
    "embedding_model": EMB_MODEL,
}
json.dump(cfg, open(os.path.join(ART, "config.json"), "w"), indent=2)

summary = {
    "protocol": (
        "Full history: per-user random 20% holdout; relevant = held-out anime rated >= 8; "
        "settings tuned on half the users; numbers below are the other half; already-seen titles excluded. "
        "Recall@10 = hits in the top 10 divided by all held-out likes (not capped at 10)."
    ),
    "data": {"anime": int(n_items), "ratings": int(len(ratings)), "users": int(len(users))},
    "als": cfg["als"],
    "hybrid_weight_content": W,
    "few_seed_best_weight_content": W_FEW,
    "app_default_weight_content": APP_W,
    "results": results,
    "few_seed": {
        "protocol": (
            "Each user contributes only their top 3, 4, or 5 training titles rated >= 8 "
            "(k cycles by user id). Scoring matches the app: mean of those seeds, "
            "crowd score trusted less below 50 ratings, small boost for a higher MyAnimeList score. "
            "Hits are still the held-out titles rated >= 8. Recall@10 uses all of those titles as the denominator."
        ),
        "best_weight_content": W_FEW,
        "results": few_results,
    },
}
json.dump(summary, open("results/metrics.json", "w"), indent=2)
with open("results/metrics.md", "w") as handle:
    handle.write("## Full watch history\n\n")
    handle.write("| Model | Precision@10 | Recall@10 | NDCG@10 |\n|---|---|---|---|\n")
    for name, row in results.items():
        handle.write(f"| {name} | {row['precision@10']:.3f} | {row['recall@10']:.3f} | {row['ndcg@10']:.3f} |\n")
    handle.write("\n## 3–5 favorite shows (how the app is used)\n\n")
    handle.write("| Model | Precision@10 | Recall@10 | NDCG@10 |\n|---|---|---|---|\n")
    for name, row in few_results.items():
        handle.write(f"| {name} | {row['precision@10']:.3f} | {row['recall@10']:.3f} | {row['ndcg@10']:.3f} |\n")
print(open("results/metrics.md").read())
