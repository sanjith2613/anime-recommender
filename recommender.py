"""Shared recommendation logic used by both build_models.py and app.py.

The heavy sentence-embedding toolkit is imported only inside encode(), so picking
favorites never loads it. Anime vectors are precomputed in artifacts/content_emb.npy.
"""
import json
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
DATA = os.path.join(HERE, "data")
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Show the crowd line only when shared fans beat what popularity alone predicts.
COLLAB_LIFT_MIN = 1.45
COLLAB_SUPPORT_MIN = 50

SKIP_GENRES = {"Award Winning", "Erotica", "Hentai", "Ecchi"}
# More specific labels first, so the "why" line leads with the interesting overlap.
THEME_PRIORITY = [
    "Time Travel", "Isekai", "Psychological", "Vampire", "Mecha", "Detective",
    "Survival", "Mystery", "Military", "Historical", "Mythology", "Reincarnation",
    "Post-Apocalyptic", "Love Polygon", "Iyashikei", "Mahou Shoujo", "Super Power",
    "Martial Arts", "Organized Crime", "Music", "Idols (Female)", "Idols (Male)",
    "School", "Workplace", "Adult Cast", "Gore", "Samurai", "Space", "Racing",
    "Team Sports", "Combat Sports", "Video Game", "Otaku Culture", "Parody",
    "Gag Humor", "Childcare", "Medical", "Urban Fantasy", "Villainess",
    "Reverse Harem", "Harem", "Crossdressing", "Delinquents", "CGDCT",
    "Performing Arts", "Visual Arts", "Showbiz", "Strategy Game", "High Stakes Game",
    "Educational", "Anthropomorphic", "Magical Sex Shift", "Pets", "Love Status Quo",
]
GENRE_PRIORITY = [
    "Horror", "Suspense", "Thriller", "Mystery", "Sci-Fi", "Fantasy", "Romance",
    "Sports", "Slice of Life", "Supernatural", "Action", "Adventure", "Comedy",
    "Drama", "Gourmet", "Avant Garde", "Police", "Boys Love", "Girls Love",
]

_ROMAN = {"ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10}
_TOKEN_SUFFIXES = {
    "movie", "movies", "film", "films", "ova", "ovas", "ona", "oad", "special", "specials",
    "recap", "recaps", "final", "gaiden", "prologue", "epilogue", "remaster", "remastered",
    "season", "part", "cour", "the", "picture", "hen", "arc", "chapter",
}
# Extra words that mean "same show, later entry" rather than a different series.
_SEQUEL_WORDS = {
    "shippuden", "shippuuden", "kai", "brotherhood", "next", "super", "gt", "zoku",
    "gx", "burst", "revolution", "r2", "reload", "advance", "advanced", "forever",
    "returns", "origin", "origins", "final", "the", "movie", "film", "ova", "special",
    "season", "part", "hen", "arc",
}
_MULTI_SUFFIXES = (
    "after story", "motion picture", "the movie", "the film", "picture drama",
    "world series", "2nd season", "3rd season", "4th season", "5th season", "6th season",
    "second season", "third season", "fourth season",
    "first stage", "second stage", "third stage", "fourth stage", "fifth stage",
    "final stage", "extra stage",
)
# Roman numerals are sequels ("Tennis II") but also part of a name ("Samurai X").
_ROMAN_SUFFIXES = {"ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"}
_PUNCT = re.compile(r"[&/!?.,'\"★☆~〜:;·・]+")
_EP_LIMIT = re.compile(
    r"\b(under|less than|fewer than|shorter than|below|max|up to|at most|<=|<)\s*(\d+)\s*(episodes?|eps?)?\b",
    re.I,
)
_EP_MIN = re.compile(
    r"\b(over|more than|longer than|at least|minimum|>=|>)\s*(\d+)\s*(episodes?|eps?)?\b",
    re.I,
)
# Each entry is (pattern, set of labels of which at least one must match).
_GENRE_CUES = [
    (re.compile(r"\bslice of life\b", re.I), {"Slice of Life"}),
    (re.compile(r"\bsci[- ]?fi\b|\bscience fiction\b", re.I), {"Sci-Fi"}),
    (re.compile(r"\bromcoms?\b|\bromantic comed(?:y|ies)\b", re.I), {"Romance", "Comedy"}),
    (re.compile(r"\bhorrors?\b", re.I), {"Horror"}),
    (re.compile(r"\bpsycholog\w*\b", re.I), {"Psychological"}),
    (re.compile(r"\bthrillers?\b", re.I), {"Suspense", "Thriller"}),
    (re.compile(r"\bmyster(?:y|ies)\b", re.I), {"Mystery"}),
    (re.compile(r"\bromances?\b|\bromantic\b", re.I), {"Romance"}),
    (re.compile(r"\bcomed(?:y|ies)\b|\bfunny\b", re.I), {"Comedy"}),
    (re.compile(r"\bfantas(?:y|ies)\b", re.I), {"Fantasy"}),
    (re.compile(r"\bactions?\b", re.I), {"Action"}),
    (re.compile(r"\bmecha\b", re.I), {"Mecha"}),
    (re.compile(r"\bisekai\b", re.I), {"Isekai"}),
    (re.compile(r"\bschools?\b", re.I), {"School"}),
    (re.compile(r"\bsports?\b", re.I), {"Sports", "Team Sports", "Combat Sports"}),
    (re.compile(r"\bsupernatural\b", re.I), {"Supernatural"}),
    (re.compile(r"\bdramas?\b", re.I), {"Drama"}),
    (re.compile(r"\badventures?\b", re.I), {"Adventure"}),
    (re.compile(r"\bvampires?\b", re.I), {"Vampire"}),
    (re.compile(r"\btime travels?\b", re.I), {"Time Travel"}),
]
_CUE_WORDS = re.compile(
    r"\b(short|long|lengthy|miniseries|mini-series|brief|series|shows?|tvs?|"
    r"movies?|films?|ovas?|onas?|specials?|episodes?|eps?|anime)\b",
    re.I,
)


def episode_phrase(n):
    """'1 episode' or '12 episodes'. None when the count is unknown."""
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return None
    n = int(n)
    if n <= 0:
        return None
    return "1 episode" if n == 1 else f"{n} episodes"


def _strip_suffixes(title):
    previous = None
    while title and title != previous:
        previous = title
        for phrase in _MULTI_SUFFIXES:
            if title.endswith(phrase):
                title = title[: -len(phrase)].strip()
                break
        else:
            # A single digit is a season ("Psycho-Pass 2"). "Gundam 00" is the series name.
            # "s2" is always a season, even on a one-word title ("Major S2").
            # A bare digit or roman numeral is left alone when it is the only thing
            # keeping "Samurai 7" or "Samurai X" from collapsing into the word "samurai".
            number = re.search(r"\s+(\d+)$", title)
            ordinal = re.search(r"\s+\d(?:st|nd|rd|th)$", title)
            short = re.search(r"\s+s\d{1,2}$", title)
            parts = title.split()
            if short or ordinal or (number and len(number.group(1)) == 1 and len(parts) >= 3):
                title = re.sub(r"\s+(?:\d(?:st|nd|rd|th)?|s\d{1,2})$", "", title).strip()
                continue
            if len(parts) >= 2 and parts[-1] in _TOKEN_SUFFIXES:
                title = " ".join(parts[:-1]).strip()
                continue
            if len(parts) >= 3 and parts[-1] in _ROMAN_SUFFIXES:
                title = " ".join(parts[:-1]).strip()
                continue
            break
    return title


def _soften(text: str) -> str:
    text = text.replace("√", " ")
    text = re.sub(r"\broot\s+a\b", " ", text)
    text = _PUNCT.sub(" ", text)
    text = re.sub(r"[-_]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_subtitle(tail: str) -> bool:
    """True when the text after a colon is a continuation, not a different series name.

    "The Movie" and "Gold" count. "Iron-Blooded Orphans" does not.
    """
    soft = _strip_suffixes(_soften(tail))
    if len(soft) < 3:
        return True
    tokens = soft.split()
    if len(tokens) <= 1:
        return True
    return all(token in _SEQUEL_WORDS or token.isdigit() for token in tokens)


def franchise_key(title: str) -> str:
    """Rough franchise name. 'Major S2' and 'Psycho-Pass: The Movie' share a key with the original."""
    if not isinstance(title, str) or not title.strip():
        return ""
    text = title.lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Use the text before a colon only when the rest is "Season 2" / "The Movie", not a different series.
    if ":" in text:
        head, tail = text.split(":", 1)
        head_key = _strip_suffixes(_soften(head))
        if head_key and _is_subtitle(tail):
            text = head_key
        else:
            text = _strip_suffixes(_soften(f"{head} {tail}"))
    else:
        text = _strip_suffixes(_soften(text))
    text = re.sub(r"^(the|a|an)\s+", "", text).strip()
    return text if len(text) >= 3 else ""


def season_index(title: str) -> int:
    """1 for the original show, 2+ for a later season. Used to prefer where to start."""
    if not isinstance(title, str):
        return 1
    text = title.lower()
    match = re.search(r"\b(\d+)(?:st|nd|rd|th)\s+season\b", text)
    if match:
        return int(match.group(1))
    match = re.search(r"\bseason\s+(\d+)\b", text)
    if match:
        return int(match.group(1))
    match = re.search(r"\bpart\s+(\d+)\b", text)
    if match:
        return int(match.group(1))
    match = re.search(r"\bs(\d{1,2})\b", text)
    if match:
        return int(match.group(1))
    match = re.search(r"\b(ii|iii|iv|v|vi|vii|viii|ix|x)\b", text)
    if match:
        return _ROMAN[match.group(1)]
    match = re.search(r"(\d+)\s*$", text.strip())
    if match:
        number = int(match.group(1))
        words = re.sub(r"\d+", " ", text).split()
        if len(words) >= 1 and 0 <= number <= 9:
            return 2 if number == 0 else number
    return 1


def _find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(parent, a, b):
    ra, rb = _find(parent, a), _find(parent, b)
    if ra != rb:
        parent[rb] = ra


def build_franchise_ids(titles, japanese_titles) -> np.ndarray:
    """One id per franchise. English and Japanese keys are linked, and clear subtitles are too."""
    n = len(titles)
    parent = list(range(n))
    buckets = {}
    for i, (english, japanese) in enumerate(zip(titles, japanese_titles)):
        for key in {franchise_key(english), franchise_key(japanese)}:
            if key:
                buckets.setdefault(key, []).append(i)
    for ids in buckets.values():
        head = ids[0]
        for other in ids[1:]:
            _union(parent, head, other)
    # "Kimetsu no Yaiba" is a prefix of "Kimetsu no Yaiba Yuukaku hen" when the colon was already eaten.
    # Link a specific title to a short arc subtitle ("kimetsu no yaiba" + "yuukaku",
    # "hell girl" + "two mirrors"). Skip umbrella names with many different children.
    import bisect
    ordered = sorted(buckets)
    children = {}
    for key in ordered:
        tokens = key.split()
        if len(tokens) < 2 or (len(tokens) < 3 and len(key) < 8) or len(key) < 8:
            continue
        prefix = key + " "
        start = bisect.bisect_left(ordered, prefix)
        stop = bisect.bisect_right(ordered, prefix + "\uffff")
        tail_limit = 3 if len(tokens) >= 3 else 2
        for longer in ordered[start:stop]:
            tail = longer[len(prefix):].split()
            if 1 <= len(tail) <= tail_limit:
                children.setdefault(key, []).append(longer)
    for key, longer_keys in children.items():
        cap = 8 if len(key.split()) >= 3 else 6
        if len(longer_keys) > cap:
            continue
        for longer in longer_keys:
            _union(parent, buckets[key][0], buckets[longer][0])
    roots = [_find(parent, i) for i in range(n)]
    remap = {root: new for new, root in enumerate(dict.fromkeys(roots))}
    return np.array([remap[r] for r in roots], dtype=np.int32)


def content_text(row) -> str:
    genres = row["genres"] if isinstance(row["genres"], str) else ""
    themes = row["themes"]
    if themes is None or (isinstance(themes, float) and np.isnan(themes)):
        theme_text = ""
    else:
        theme_text = ", ".join(str(t) for t in themes if str(t))
    demographic = ""
    if "demographic" in getattr(row, "index", []):
        demographic = row["demographic"] if isinstance(row["demographic"], str) else ""
    synopsis = row["synopsis"] if isinstance(row["synopsis"], str) else ""
    parts = [str(row["title"])]
    if genres:
        parts.append(f"Genres: {genres}.")
    if theme_text:
        parts.append(f"Themes: {theme_text}.")
    if demographic and demographic not in ("nan", "None"):
        parts.append(f"Audience: {demographic}.")
    if synopsis:
        parts.append(synopsis[:800])
    return " ".join(parts)


def normalize(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms


def zscore(values):
    values = np.asarray(values, dtype=np.float32)
    spread = float(values.std())
    return (values - values.mean()) / (spread if spread > 0 else 1.0)


def hybrid_scores(emb, cf_norm, support, mal_scores, seed_idx, weight):
    """App blend: z-scored content, crowd signal down-weighted when a title has few ratings, plus a small quality nudge."""
    seed_idx = [int(s) for s in seed_idx]
    content = emb @ emb[seed_idx].mean(axis=0)
    supported = [s for s in seed_idx if support[s] > 0]
    if supported:
        collaborative = cf_norm @ cf_norm[supported].mean(axis=0)
    else:
        collaborative = np.zeros(emb.shape[0], dtype=np.float32)
    trust = np.clip(np.asarray(support, dtype=np.float32) / 50.0, 0, 1)
    blend = weight * zscore(content) + (1 - weight) * trust * zscore(collaborative)
    return blend + 0.1 * zscore(mal_scores)


def parse_vibe(query: str) -> dict:
    """Pull format, length, and genre constraints out of a free-text request.

    With no format word, results are TV series. 'short' means one cour (≤ 13 episodes).
    """
    text = query or ""
    lowered = text.lower()
    max_episodes = None
    min_episodes = None
    match = _EP_LIMIT.search(lowered)
    if match:
        max_episodes = int(match.group(2))
    match = _EP_MIN.search(lowered)
    if match:
        min_episodes = int(match.group(2))
    if re.search(r"\b(short|miniseries|mini-series|brief)\b", lowered):
        max_episodes = 13 if max_episodes is None else min(max_episodes, 13)
    if re.search(r"\b(long|lengthy)\b", lowered) and min_episodes is None:
        min_episodes = 24

    if re.search(r"\b(movies?|films?)\b", lowered):
        types = {"Movie"}
    elif re.search(r"\bovas?\b", lowered):
        types = {"OVA"}
    elif re.search(r"\bspecials?\b", lowered):
        types = {"Special"}
    elif re.search(r"\bonas?\b", lowered):
        types = {"ONA"}
    else:
        types = {"TV"}

    # "romantic comedy" is listed before the separate romance/comedy cues; skip the
    # pieces when the combined phrase already required both.
    require = []
    consumed = lowered
    if re.search(r"\bromcoms?\b|\bromantic comed(?:y|ies)\b", lowered):
        require.append({"Romance"})
        require.append({"Comedy"})
        consumed = re.sub(r"\bromcoms?\b|\bromantic comed(?:y|ies)\b", " ", consumed)
    for pattern, labels in _GENRE_CUES:
        if pattern.pattern.startswith(r"\bromcoms?") or "romantic comed" in pattern.pattern:
            continue
        if pattern.search(consumed):
            require.append(labels)

    clean = _EP_LIMIT.sub(" ", text)
    clean = _EP_MIN.sub(" ", clean)
    clean = _CUE_WORDS.sub(" ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return {
        "types": types,
        "max_episodes": max_episodes,
        "min_episodes": min_episodes,
        "require": require,
        "clean_query": clean or text.strip(),
    }


class Recommender:
    def __init__(self, art_dir=ART, data_dir=DATA):
        self.df = pd.read_parquet(os.path.join(data_dir, "anime.parquet"))
        self.emb = np.load(os.path.join(art_dir, "content_emb.npy")).astype(np.float32)
        factors = np.load(os.path.join(art_dir, "cf_item_factors.npy")).astype(np.float32)
        self.cf = normalize(factors)
        self.cf_support = np.load(os.path.join(art_dir, "cf_support.npy"))
        with open(os.path.join(art_dir, "config.json")) as handle:
            self.cfg = json.load(handle)
        self.idx = {aid: i for i, aid in enumerate(self.df["anime_id"])}
        self.franchise = build_franchise_ids(self.df["title"].tolist(), self.df["japanese_title"].tolist())
        self._model = None
        self._load_likers(art_dir)

    def _load_likers(self, art_dir):
        path = os.path.join(art_dir, "cf_likers.npz")
        if not os.path.exists(path):
            self._liker_users = None
            return
        packed = np.load(path)
        self._liker_users = packed["users"]
        self._liker_ptr = packed["indptr"]
        self._n_users = int(packed["n_users"])
        self._nlikes = np.diff(self._liker_ptr)

    def _both_likes(self, i, j):
        left = self._liker_users[self._liker_ptr[i]:self._liker_ptr[i + 1]]
        right = self._liker_users[self._liker_ptr[j]:self._liker_ptr[j + 1]]
        return np.intersect1d(left, right, assume_unique=True).size

    def strong_collab(self, i, j) -> bool:
        """True when the two shows share fans beyond what their popularity predicts."""
        if self._liker_users is None or i == j:
            return False
        ni, nj = int(self._nlikes[i]), int(self._nlikes[j])
        if ni < COLLAB_SUPPORT_MIN or nj < COLLAB_SUPPORT_MIN:
            return False
        both = self._both_likes(i, j)
        if both < COLLAB_SUPPORT_MIN:
            return False
        expected = ni * nj / self._n_users
        return expected > 0 and (both / expected) >= COLLAB_LIFT_MIN

    def encode(self, texts):
        """Embed a query. Loads the ONNX MiniLM model on first call, not at startup."""
        if isinstance(texts, str):
            texts = [texts]
        if self._model is None:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(EMB_MODEL)
        vectors = np.vstack([np.asarray(v, dtype=np.float32) for v in self._model.embed(texts)])
        return normalize(vectors)

    def _labels(self, row):
        genres = set()
        if isinstance(row["genres"], str):
            genres = {part.strip() for part in row["genres"].split(",") if part.strip()} - SKIP_GENRES
        raw = row["themes"]
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            themes = set()
        else:
            themes = {str(t) for t in raw if str(t)}
        return genres, themes

    def _shared(self, i, s):
        genres, themes = self._labels(self.df.iloc[i])
        seed_genres, seed_themes = self._labels(self.df.iloc[s])
        shared_themes = [t for t in THEME_PRIORITY if t in themes and t in seed_themes]
        shared_genres = [g for g in GENRE_PRIORITY if g in genres and g in seed_genres]
        content = float(self.emb[i] @ self.emb[s])
        return shared_themes, shared_genres, content

    def scores(self, seed_idx, w_content=None):
        """Hybrid score for every anime given a list of seed rows."""
        weight = self.cfg["hybrid_weight_content"] if w_content is None else w_content
        filled = self.df["score"].fillna(self.df["score"].median()).to_numpy(dtype=np.float32)
        total = hybrid_scores(self.emb, self.cf, self.cf_support, filled, seed_idx, weight)
        return total

    def recommend(self, seed_ids, k=10, hide_franchise=True, max_episodes=None, w_content=None):
        seed_idx = [self.idx[s] for s in seed_ids if s in self.idx]
        if not seed_idx:
            return []
        total = self.scores(seed_idx, w_content)
        blocked = {int(self.franchise[s]) for s in seed_idx} if hide_franchise else set()
        seed_set = set(seed_idx)
        groups = {}
        for i in range(len(self.df)):
            if i in seed_set:
                continue
            franchise = int(self.franchise[i])
            if franchise in blocked:
                continue
            episodes = self.df["episodes"].iat[i]
            if max_episodes and (pd.isna(episodes) or episodes > max_episodes):
                continue
            groups.setdefault(franchise, []).append(i)
        ranked = []
        for members in groups.values():
            # Surface the franchise if any entry matches, but show where to start.
            best = max(float(total[i]) for i in members)
            face = min(members, key=self._rep_key)
            ranked.append((best, face))
        ranked.sort(key=lambda pair: -pair[0])
        return [self._card(i, seed_idx) for _, i in ranked[:k]]

    def explain(self, i, seed_idx):
        """One line: real shared genres/themes, plus a crowd note only when the overlap is strong."""
        best = None
        for seed in seed_idx:
            shared_themes, shared_genres, content = self._shared(i, seed)
            rank = (len(shared_themes), len(shared_genres), content)
            if best is None or rank > best[0]:
                best = (rank, seed, shared_themes, shared_genres)
        _, seed, shared_themes, shared_genres = best
        seed_row = self.df.iloc[seed]
        parts = []
        if shared_themes:
            parts.append("themes like " + ", ".join(shared_themes[:3]))
        if shared_genres:
            parts.append("shared genres " + ", ".join(shared_genres[:2]))
        line = f"Similar to {seed_row['title']}"
        if parts:
            line += ": " + "; ".join(parts)
        if self.strong_collab(i, seed):
            line += f" — people who liked {seed_row['title']} also liked this"
        return line

    def _card(self, i, seed_idx=None, why=None):
        row = self.df.iloc[i]
        episodes = None if pd.isna(row["episodes"]) else int(row["episodes"])
        return {
            "anime_id": int(row["anime_id"]),
            "title": row["display"],
            "image_url": row["image_url"] or "",
            "score": None if pd.isna(row["score"]) else float(row["score"]),
            "genres": row["genres"],
            "type": row["type"],
            "episodes": episodes,
            "synopsis": row["synopsis"],
            "why": why or self.explain(i, seed_idx),
        }

    def _passes(self, i, types, max_episodes, min_episodes, min_score, require):
        row = self.df.iloc[i]
        if types and row["type"] not in types:
            return False
        episodes = row["episodes"]
        if max_episodes and (pd.isna(episodes) or episodes > max_episodes):
            return False
        if min_episodes and (pd.isna(episodes) or episodes < min_episodes):
            return False
        if min_score and (pd.isna(row["score"]) or row["score"] < min_score):
            return False
        if require:
            genres, themes = self._labels(row)
            labels = genres | themes
            if not all(labels & group for group in require):
                return False
        return True

    def _rep_key(self, i):
        row = self.df.iloc[i]
        title = f"{row['title']} {row['japanese_title']}"
        junk = 1 if re.search(r"recap|picture drama|summary|chibi|digest", title, re.I) else 0
        type_rank = {"TV": 0, "ONA": 1, "Movie": 2, "OVA": 3, "Special": 4}.get(row["type"], 5)
        year = int(row["year"]) if not pd.isna(row["year"]) else 9999
        members = -int(row["members"]) if not pd.isna(row["members"]) else 0
        return (junk, type_rank, season_index(title), year, members)

    def text_search(self, query, k=10, max_episodes=None, min_score=0.0, types=None):
        """Meaning search. TV by default, one title per franchise, first season preferred."""
        parsed = parse_vibe(query)
        chosen_types = set(types) if types else parsed["types"]
        limit = max_episodes if max_episodes else parsed["max_episodes"]
        minimum = None if max_episodes else parsed["min_episodes"]
        require = parsed["require"]
        query_vector = self.encode(parsed["clean_query"])[0]
        similarity = self.emb @ query_vector
        quality = zscore(self.df["score"].fillna(6).to_numpy(dtype=np.float32))
        popularity = zscore(np.log1p(self.df["members"].to_numpy(dtype=np.float32)))
        total = similarity + 0.03 * quality + 0.02 * popularity

        def gather(use_require):
            best = {}
            for i in np.argsort(-total):
                if not self._passes(i, chosen_types, limit, minimum, min_score, use_require):
                    continue
                franchise = int(self.franchise[i])
                if franchise not in best or self._rep_key(i) < self._rep_key(best[franchise][0]):
                    best[franchise] = (i, float(total[i]))
            return sorted(best.values(), key=lambda pair: -pair[1])

        ranked = gather(require)
        if require and len(ranked) < k:
            ranked = gather(None)
        out = []
        for i, _score in ranked[:k]:
            row = self.df.iloc[i]
            phrase = episode_phrase(row["episodes"])
            why = f"Matches your description (similarity {similarity[i]:.2f})"
            if phrase:
                why += f"; {phrase}"
            out.append(self._card(i, why=why))
        info = {"types": sorted(chosen_types)}
        if limit:
            info["max_episodes"] = int(limit)
        if minimum:
            info["min_episodes"] = int(minimum)
        return out, info
