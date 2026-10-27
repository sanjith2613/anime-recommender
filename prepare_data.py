"""Step 1 - Download and clean the anime catalog.

Sources (public, no login):
  * TheRealVigilante/Processed_Top_15k_Anime on Hugging Face
      Cleaned mirror of Quan Than's "Top 15,000 Ranked Anime Dataset"
      (Kaggle, updated March 2025), Apache License 2.0.
      Synopsis, poster, score, members, type, episodes, studios, titles.
  * lyfesan/myanimelist-top-anime-dataset on Hugging Face (MIT, September 2025)
      Official MyAnimeList genres, themes, and demographic. No synopsis.
  * krishnaveni76/UserRatings and krishnaveni76/Animes
      Ratings, plus metadata for rated titles missing from the 2025 dump.

Outputs:
  data/anime.parquet    cleaned catalog used by the app
  data/ratings.parquet  ratings restricted to that catalog
"""
import os
import re
import subprocess

import pandas as pd

RAW = "data/raw"
FILES = {
    "top15k.csv": "https://huggingface.co/datasets/TheRealVigilante/Processed_Top_15k_Anime/resolve/main/anime.csv",
    "lyfesan.csv": "https://huggingface.co/datasets/lyfesan/myanimelist-top-anime-dataset/resolve/main/mal_top_anime_dataset.csv",
    "UserRatings.csv": "https://huggingface.co/datasets/krishnaveni76/UserRatings/resolve/main/UserRatings.csv",
    "Animes.csv": "https://huggingface.co/datasets/krishnaveni76/Animes/resolve/main/Animes.csv",
}

MIN_MEMBERS = 5000
KEEP_TYPES = {"TV", "Movie", "OVA", "ONA", "Special", "TV Special"}
BLOCK_GENRES = {"Hentai", "Erotica"}

# Current MyAnimeList vocabulary (plus a few labels the older dump still uses).
OFFICIAL_GENRES = {
    "Action", "Adventure", "Avant Garde", "Comedy", "Drama", "Ecchi", "Fantasy",
    "Gourmet", "Horror", "Mystery", "Romance", "Sci-Fi", "Slice of Life", "Sports",
    "Supernatural", "Suspense", "Boys Love", "Girls Love", "Thriller", "Police",
}
OFFICIAL_THEMES = {
    "Adult Cast", "Anthropomorphic", "CGDCT", "Childcare", "Combat Sports",
    "Crossdressing", "Delinquents", "Detective", "Educational", "Gag Humor", "Gore",
    "Harem", "High Stakes Game", "Historical", "Idols (Female)", "Idols (Male)",
    "Isekai", "Iyashikei", "Love Polygon", "Love Status Quo", "Magical Sex Shift",
    "Mahou Shoujo", "Martial Arts", "Mecha", "Medical", "Military", "Music",
    "Mythology", "Organized Crime", "Otaku Culture", "Parody", "Performing Arts",
    "Pets", "Psychological", "Racing", "Reincarnation", "Reverse Harem", "Samurai",
    "School", "Showbiz", "Space", "Strategy Game", "Super Power", "Survival",
    "Team Sports", "Time Travel", "Urban Fantasy", "Vampire", "Video Game",
    "Villainess", "Visual Arts", "Workplace",
}
DEMOGRAPHICS = {"Shounen", "Shoujo", "Seinen", "Josei", "Kids"}
# Older dumps parked some of today's themes in the genre column.
GENRE_ALIASES = {"Thriller": "Suspense", "Sci Fi": "Sci-Fi", "Slice of life": "Slice of Life"}
DROP_LABELS = {"Award Winning", "Hentai", "Erotica"}


def download():
    os.makedirs(RAW, exist_ok=True)
    for name, url in FILES.items():
        path = os.path.join(RAW, name)
        if not os.path.exists(path) or os.path.getsize(path) < 1000:
            print("downloading", name)
            subprocess.run(["curl", "-fsSL", "-o", path, url], check=True)


def strip_frame(df):
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
            df[col] = df[col].fillna("").astype(str).str.strip().replace({"nan": "", "None": "", "<NA>": "", "UNKNOWN": ""})
    return df


def clean_synopsis(text):
    s = re.sub(r"\s+", " ", str(text)).strip()
    s = re.sub(r"\[Written by MAL Rewrite\]", "", s)
    s = re.sub(r"\(Source:.*?\)", "", s).strip()
    return s


def split_labels(text):
    """Split a comma-separated MAL label string into genres, themes, demographic."""
    genres, themes, demo = [], [], ""
    for raw in str(text).split(","):
        label = raw.strip()
        if not label or label in DROP_LABELS:
            continue
        label = GENRE_ALIASES.get(label, label)
        if label in DEMOGRAPHICS and not demo:
            demo = label
        elif label in OFFICIAL_THEMES and label not in themes:
            themes.append(label)
        elif label in OFFICIAL_GENRES and label not in genres:
            genres.append(label)
    return genres, themes, demo


def year_from(*values):
    for value in values:
        match = re.search(r"(19|20)\d{2}", str(value))
        if match:
            return int(match.group(0))
    return None


def as_text(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "unknown"} else text


def display_name(english, primary):
    english, primary = as_text(english), as_text(primary)
    if not english or english.lower() == primary.lower() or english.lower() in primary.lower():
        return primary or english
    return f"{english} ({primary})"


def from_top15k(lyfesan):
    v = strip_frame(pd.read_csv(os.path.join(RAW, "top15k.csv")))
    v["anime_id"] = pd.to_numeric(v["anime_id"], errors="coerce")
    v = v.dropna(subset=["anime_id"])
    v["anime_id"] = v["anime_id"].astype(int)
    v["members"] = pd.to_numeric(v["members"], errors="coerce").fillna(0)
    v["score"] = pd.to_numeric(v["score"], errors="coerce")
    v["episodes"] = pd.to_numeric(v["episodes"], errors="coerce")
    v["type"] = v["type"].str.strip()
    v["rating"] = v["rating"].str.strip()
    v = v[v["type"].isin(KEEP_TYPES)]
    v = v[~v["rating"].str.contains("Hentai", case=False, na=False)]
    v = v[~v["genres"].str.contains("Hentai|Erotica", case=False, na=False)]
    v = v[v["members"] >= MIN_MEMBERS]
    v = v.sort_values(["members", "score"], ascending=False)
    v = v.drop_duplicates("anime_id", keep="first")

    meta = lyfesan.set_index("anime_id")
    rows = []
    for rec in v.itertuples(index=False):
        synopsis = clean_synopsis(rec.synopsis)
        if len(synopsis) < 40:
            continue
        extra = meta.loc[rec.anime_id] if rec.anime_id in meta.index else None
        if extra is not None and isinstance(extra, pd.DataFrame):
            extra = extra.iloc[0]
        if extra is not None and str(extra["Genres"]).strip():
            genres, themes, demo = split_labels(
                ",".join(p for p in (str(extra["Genres"]), str(extra["Themes"]), str(extra["Demographic"])) if p and p != "nan")
            )
        else:
            genres, themes, demo = split_labels(rec.genres)
        english = as_text(rec.english_name)
        if extra is not None and not english:
            english = as_text(extra["English"])
        primary = as_text(rec.name)
        kind = "Special" if rec.type == "TV Special" else rec.type
        rows.append({
            "anime_id": int(rec.anime_id),
            "title": english or primary,
            "japanese_title": primary,
            "display": display_name(english, primary),
            "synopsis": synopsis,
            "genres": ", ".join(genres),
            "themes": themes,
            "demographic": demo,
            "studios": as_text(rec.studios),
            "type": kind,
            "episodes": rec.episodes,
            "score": rec.score,
            "members": int(rec.members),
            "year": year_from(
                extra["Premiered"] if extra is not None else "",
                extra["Aired"] if extra is not None else "",
                rec.premiered,
            ),
            "image_url": as_text(rec.image_url),
        })
    return pd.DataFrame(rows)


def load_lyfesan():
    m = strip_frame(pd.read_csv(os.path.join(RAW, "lyfesan.csv")))
    m["anime_id"] = pd.to_numeric(m["URL"].str.extract(r"/anime/(\d+)")[0], errors="coerce")
    m = m.dropna(subset=["anime_id"]).drop_duplicates("anime_id")
    m["anime_id"] = m["anime_id"].astype(int)
    return m


def gap_fill(have_ids):
    """Rated titles the March 2025 dump does not contain, from the older metadata file."""
    a = pd.read_csv(os.path.join(RAW, "Animes.csv"))
    a = strip_frame(a)
    a = a.rename(columns={"name": "primary", "overview": "synopsis", "image url": "image_url",
                          "average_rating": "score"})
    a["anime_id"] = pd.to_numeric(a["anime_id"], errors="coerce")
    a = a.dropna(subset=["anime_id"])
    a["anime_id"] = a["anime_id"].astype(int)
    a = a[~a["anime_id"].isin(have_ids)]
    a["score"] = pd.to_numeric(a["score"], errors="coerce")
    a["episodes"] = pd.to_numeric(a["episodes"], errors="coerce")
    a["members"] = pd.to_numeric(a["members"], errors="coerce").fillna(0)
    a["type"] = a["type"].str.strip()
    a = a[a["type"].isin(KEEP_TYPES - {"TV Special"})]
    a = a[a["members"] >= MIN_MEMBERS]
    a = a[~a["genres"].str.contains("Hentai|Erotica", case=False, na=False)]
    rows = []
    for rec in a.itertuples(index=False):
        synopsis = clean_synopsis(rec.synopsis)
        if len(synopsis) < 40:
            continue
        genres, themes, demo = split_labels(rec.genres)
        if any(b in rec.genres for b in BLOCK_GENRES):
            continue
        primary = as_text(rec.primary)
        rows.append({
            "anime_id": int(rec.anime_id),
            "title": primary,
            "japanese_title": primary,
            "display": primary,
            "synopsis": synopsis,
            "genres": ", ".join(genres),
            "themes": themes,
            "demographic": demo,
            "studios": as_text(rec.studios),
            "type": rec.type,
            "episodes": rec.episodes,
            "score": rec.score,
            "members": int(rec.members),
            "year": None,
            "image_url": as_text(getattr(rec, "image_url", "")),
        })
    return pd.DataFrame(rows)


def main():
    download()
    lyfesan = load_lyfesan()
    catalog = from_top15k(lyfesan)
    extra = gap_fill(set(catalog["anime_id"]))
    if len(extra):
        catalog = pd.concat([catalog, extra], ignore_index=True)
    catalog = catalog.drop_duplicates("anime_id", keep="first")
    catalog = catalog.sort_values("members", ascending=False).reset_index(drop=True)
    dup = catalog["display"].duplicated(keep=False)
    catalog.loc[dup, "display"] = catalog.loc[dup, "display"] + " #" + catalog.loc[dup, "anime_id"].astype(str)
    catalog["themes"] = catalog["themes"].apply(lambda xs: list(xs) if isinstance(xs, (list, tuple)) else [])
    catalog["demographic"] = catalog["demographic"].fillna("").astype(str)
    catalog["image_url"] = catalog["image_url"].fillna("").astype(str)
    catalog["genres"] = catalog["genres"].fillna("").astype(str)
    catalog["synopsis"] = catalog["synopsis"].fillna("").astype(str)

    os.makedirs("data", exist_ok=True)
    catalog.to_parquet("data/anime.parquet", index=False)

    ratings = pd.read_csv(os.path.join(RAW, "UserRatings.csv"))
    ratings = ratings[["user_id", "anime_id", "rating"]]
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings = ratings.dropna()
    ratings = ratings[ratings["anime_id"].isin(set(catalog["anime_id"]))]
    ratings = ratings.drop_duplicates(["user_id", "anime_id"])
    ratings.to_parquet("data/ratings.parquet", index=False)

    names = (catalog["title"] + " " + catalog["japanese_title"] + " " + catalog["display"]).str.lower()
    wanted = {
        "Demon Slayer / Kimetsu": "kimetsu|demon slayer",
        "Jujutsu Kaisen": "jujutsu kaisen",
        "Spy x Family": "spy x family",
        "Frieren": "frieren",
        "Kaguya-sama": "kaguya-sama",
    }
    print(f"catalog: {len(catalog)} anime | ratings: {len(ratings):,} from {ratings.user_id.nunique()} users "
          f"on {ratings.anime_id.nunique()} anime | years {catalog.year.min()}–{catalog.year.max()}")
    for label, pat in wanted.items():
        hit = names.str.contains(pat, regex=True)
        print(f"  {label}: {int(hit.sum())} titles, top: {catalog.loc[hit, 'display'].head(1).tolist()}")


if __name__ == "__main__":
    main()
