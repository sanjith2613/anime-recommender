"""Step 1 - Download and clean the anime data.

Sources (all public Hugging Face datasets, no login needed):
  * krishnaveni76/Animes            -> Animes.csv  (MyAnimeList metadata: synopsis, genres, studios, score, members...)
  * krishnaveni76/UserRatings       -> UserRatings.csv (1.1M ratings from 813 MyAnimeList users)
  * Mikami063/User-Animelist-Dataset -> animes.csv (used only for English titles + sequel flag)

Outputs:
  data/anime.parquet    cleaned catalog used by the app
  data/ratings.parquet  ratings restricted to that catalog
"""
import ast
import os
import re
import subprocess

import pandas as pd

RAW = "data/raw"
FILES = {
    "Animes.csv": "https://huggingface.co/datasets/krishnaveni76/Animes/resolve/main/Animes.csv",
    "UserRatings.csv": "https://huggingface.co/datasets/krishnaveni76/UserRatings/resolve/main/UserRatings.csv",
    "animes_english_titles.csv": "https://huggingface.co/datasets/Mikami063/User-Animelist-Dataset/resolve/main/animes.csv",
}
MIN_MEMBERS = 5000          # drop very obscure titles
KEEP_TYPES = {"TV", "Movie", "OVA", "ONA", "Special"}
BLOCK_GENRES = {"Hentai", "Erotica"}


def download():
    os.makedirs(RAW, exist_ok=True)
    for name, url in FILES.items():
        path = os.path.join(RAW, name)
        if not os.path.exists(path):
            print("downloading", name)
            subprocess.run(["curl", "-sL", "-o", path, url], check=True)


def to_num(s):
    return pd.to_numeric(s.replace("UNKNOWN", None), errors="coerce")


def main():
    download()
    a = pd.read_csv(os.path.join(RAW, "Animes.csv"))
    a = a.rename(columns={"name": "japanese_title", "overview": "synopsis",
                          "image url": "image_url", "average_rating": "score"})
    a["score"] = to_num(a["score"])
    a["episodes"] = to_num(a["episodes"])
    a["genres"] = a["genres"].fillna("").replace("UNKNOWN", "")
    a["studios"] = a["studios"].fillna("").replace("UNKNOWN", "")
    a["synopsis"] = a["synopsis"].fillna("").str.replace(r"\s+", " ", regex=True).str.strip()

    # English titles from the second dataset (joined on the MyAnimeList id in its URL)
    m = pd.read_csv(os.path.join(RAW, "animes_english_titles.csv"))
    m["anime_id"] = pd.to_numeric(m["mal_url"].str.extract(r"/anime/(\d+)")[0], errors="coerce")
    m = m.dropna(subset=["anime_id"]).drop_duplicates("anime_id")
    m["anime_id"] = m["anime_id"].astype(int)
    m["themes"] = m["genres_detailed"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
    a = a.merge(m[["anime_id", "title", "year", "themes"]], on="anime_id", how="left")
    a["title"] = a["title"].fillna(a["japanese_title"])
    a["themes"] = a["themes"].apply(lambda x: x if isinstance(x, list) else [])

    # Filters
    a = a[a["type"].isin(KEEP_TYPES)]
    a = a[a["members"] >= MIN_MEMBERS]
    a = a[~a["genres"].apply(lambda g: any(b in g for b in BLOCK_GENRES))]
    a = a[a["synopsis"].str.len() > 40]

    # A display name that shows both titles when they differ
    def display(r):
        en, jp = r["title"], r["japanese_title"]
        if en == jp or en.lower() in jp.lower():
            return jp
        return f"{en} ({jp})"
    a["display"] = a.apply(display, axis=1)
    a = a.sort_values("members", ascending=False).reset_index(drop=True)
    # drop duplicate display names (rare) by appending id
    dup = a["display"].duplicated(keep=False)
    a.loc[dup, "display"] = a.loc[dup, "display"] + " #" + a.loc[dup, "anime_id"].astype(str)

    cols = ["anime_id", "title", "japanese_title", "display", "synopsis", "genres", "themes",
            "studios", "type", "episodes", "score", "members", "year", "image_url"]
    a = a[cols]
    os.makedirs("data", exist_ok=True)
    a.to_parquet("data/anime.parquet", index=False)

    r = pd.read_csv(os.path.join(RAW, "UserRatings.csv"))[["user_id", "anime_id", "rating"]]
    r = r[r["anime_id"].isin(a["anime_id"])].drop_duplicates(["user_id", "anime_id"])
    r.to_parquet("data/ratings.parquet", index=False)
    print(f"catalog: {len(a)} anime | ratings: {len(r):,} from {r.user_id.nunique()} users "
          f"on {r.anime_id.nunique()} anime")


if __name__ == "__main__":
    main()
