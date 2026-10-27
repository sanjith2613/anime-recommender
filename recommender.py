"""Shared recommendation logic used by both build_models.py and app.py."""
import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
DATA = os.path.join(HERE, "data")
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------- helpers
_SUFFIX = re.compile(
    r"(\s+(season|part|movie|the movie|ova|ona|special|specials|final|cour|"
    r"\d+(st|nd|rd|th)?( season)?|ii|iii|iv))+$")


def franchise_key(title: str) -> str:
    """Rough 'franchise' name so we can hide sequels: 'Shingeki no Kyojin Season 2' -> 'shingeki no kyojin'."""
    t = title.lower()
    head = t.split(":")[0]
    t = head if len(head) >= 4 else t
    t = re.sub(r"[\(\)\[\]!?.,'\"]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    prev = None
    while prev != t:
        prev, t = t, _SUFFIX.sub("", t).strip()
    return " ".join(t.split()[:3])


def content_text(row) -> str:
    themes = ", ".join(row["themes"][:25]) if len(row["themes"]) else ""
    return f'{row["title"]}. Genres: {row["genres"]}. Themes: {themes}. {row["synopsis"]}'


def normalize(m):
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1
    return m / n


def zscore(x):
    s = x.std()
    return (x - x.mean()) / (s if s > 0 else 1)


# ---------------------------------------------------------------- recommender
class Recommender:
    def __init__(self, art_dir=ART, data_dir=DATA):
        self.df = pd.read_parquet(os.path.join(data_dir, "anime.parquet"))
        self.emb = np.load(os.path.join(art_dir, "content_emb.npy")).astype(np.float32)
        cf = np.load(os.path.join(art_dir, "cf_item_factors.npy")).astype(np.float32)
        self.cf = normalize(cf)
        self.cf_support = np.load(os.path.join(art_dir, "cf_support.npy"))  # ratings per anime
        with open(os.path.join(art_dir, "config.json")) as f:
            self.cfg = json.load(f)
        self.idx = {aid: i for i, aid in enumerate(self.df["anime_id"])}
        self.fkey = self.df["japanese_title"].map(franchise_key).values
        self.fkey_en = self.df["title"].map(franchise_key).values
        # theme document-frequency, used to pick "interesting" shared themes for the explanation
        df_count = Counter(t for ts in self.df["themes"] for t in set(ts))
        n = len(self.df)
        meta = re.compile(r"adapt|based on|canon|filler|award|production|heterosexual|engrish|cgi|remaster|"
                          r"episod|original work|^new$|^place$|^present$|^time$|^earth$|^asia|japan|manga|"
                          r"novel|protagonist|primarily|cast$|tropes|stereotype|pantsu|nudity|^sci|fiction$|"
                          r"version|kuudere|plot continuity|^mina$|spider tank|--|product placement")
        self.theme_df = df_count
        self.good_theme = {t for t, c in df_count.items()
                           if 8 <= c <= 0.15 * n and len(t) > 3 and not meta.search(t)}
        # cosine similarity in the ALS space above which we say "fans overlap" (top ~2% of all pairs)
        self.cf_fan_cut = 0.5
        self._st = None

    # -- embeddings for free-text search (loaded lazily, only when needed)
    def encode(self, text):
        if self._st is None:
            from sentence_transformers import SentenceTransformer
            self._st = SentenceTransformer(EMB_MODEL)
        return self._st.encode([text], normalize_embeddings=True)[0]

    def _same_franchise(self, i, seed_idx):
        return any(self.fkey[i] == self.fkey[s] or self.fkey_en[i] == self.fkey_en[s] for s in seed_idx)

    def scores(self, seed_idx, w_content=None):
        """Hybrid score for every anime given a list of seed rows."""
        w = self.cfg["hybrid_weight_content"] if w_content is None else w_content
        content = self.emb @ self.emb[seed_idx].mean(axis=0)
        cf = self.cf @ self.cf[seed_idx].mean(axis=0)
        # trust collaborative filtering less for titles few users rated
        trust = np.clip(self.cf_support / 50.0, 0, 1)
        blend = w * zscore(content) + (1 - w) * trust * zscore(cf)
        # tiny nudge toward well-rated shows
        quality = zscore(self.df["score"].fillna(self.df["score"].median()).values)
        return blend + 0.1 * quality, content, cf

    def recommend(self, seed_ids, k=10, hide_franchise=True, max_episodes=None, w_content=None):
        seed_idx = [self.idx[s] for s in seed_ids if s in self.idx]
        if not seed_idx:
            return []
        total, content, cf = self.scores(seed_idx, w_content)
        order = np.argsort(-total)
        out, seen_keys = [], set()
        for i in order:
            if i in seed_idx:
                continue
            if hide_franchise and self._same_franchise(i, seed_idx):
                continue
            if max_episodes and (np.isnan(self.df["episodes"].iat[i]) or self.df["episodes"].iat[i] > max_episodes):
                continue
            if self.fkey[i] in seen_keys:      # one entry per franchise in the list
                continue
            seen_keys.add(self.fkey[i])
            out.append(self._card(i, seed_idx, content, cf))
            if len(out) == k:
                break
        return out

    def explain(self, i, seed_idx):
        row = self.df.iloc[i]
        g = set(x.strip() for x in row["genres"].split(",") if x.strip())
        # which seed is it closest to?
        sims = [(float(self.emb[i] @ self.emb[s]) + float(self.cf[i] @ self.cf[s]), s) for s in seed_idx]
        best = max(sims)[1]
        seed = self.df.iloc[best]
        sg = set(x.strip() for x in seed["genres"].split(",") if x.strip())
        shared_g = sorted(g & sg)
        shared_t = sorted((t for t in set(row["themes"]) & set(seed["themes"]) if t in self.good_theme),
                          key=lambda t: self.theme_df[t])[:3]   # rarest (most specific) shared themes first
        parts = []
        if shared_t:
            parts.append("themes like " + ", ".join(shared_t))
        if shared_g:
            parts.append("shared genres " + ", ".join(shared_g[:3]))
        cf_note = (f" — people who liked {seed['title']} also liked this"
                   if float(self.cf[i] @ self.cf[best]) > self.cf_fan_cut else "")
        return f"Similar to {seed['title']}" + (f": {'; '.join(parts)}" if parts else "") + cf_note

    def _card(self, i, seed_idx=None, content=None, cf=None, why=None):
        r = self.df.iloc[i]
        return {
            "anime_id": int(r["anime_id"]), "title": r["display"], "image_url": r["image_url"],
            "score": None if pd.isna(r["score"]) else float(r["score"]),
            "genres": r["genres"], "type": r["type"],
            "episodes": None if pd.isna(r["episodes"]) else int(r["episodes"]),
            "synopsis": r["synopsis"],
            "why": why or self.explain(i, seed_idx),
        }

    # -- free text search: "dark psychological thriller under 25 episodes"
    @staticmethod
    def parse_episode_limit(text):
        m = re.search(r"(under|less than|fewer than|below|max|<|<=|up to)\s*(\d+)\s*(ep|episodes?)?", text.lower())
        return int(m.group(2)) if m else None

    def text_search(self, query, k=10, max_episodes=None, min_score=0.0, types=("TV", "Movie", "ONA")):
        limit = max_episodes or self.parse_episode_limit(query)
        # remove the "under 25 episodes" part so it doesn't distract the meaning search
        clean = re.sub(r"(under|less than|fewer than|below|max|<=|<|up to)\s*\d+\s*(ep|episodes?)?", "",
                       query, flags=re.I).strip() or query
        q = self.encode(clean)
        sim = self.emb @ q
        quality = zscore(self.df["score"].fillna(6).values)
        pop = zscore(np.log1p(self.df["members"].values))
        total = sim + 0.03 * quality + 0.02 * pop
        out, seen = [], set()
        for i in np.argsort(-total):
            r = self.df.iloc[i]
            if limit and (pd.isna(r["episodes"]) or r["episodes"] > limit):
                continue
            if types and r["type"] not in types:
                continue
            if min_score and (pd.isna(r["score"]) or r["score"] < min_score):
                continue
            if self.fkey[i] in seen:
                continue
            seen.add(self.fkey[i])
            why = f"Matches your description (similarity {sim[i]:.2f})" + (f"; {int(r['episodes'])} episodes" if not pd.isna(r['episodes']) else "")
            out.append(self._card(i, why=why))
            if len(out) == k:
                break
        return out, limit
