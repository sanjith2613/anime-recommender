# 🎌 Anime Recommender

A web app that recommends anime. Pick 3–5 shows you love and it returns 10 picks, each with its cover art, MyAnimeList score, genres, and a one-line reason ("Similar to Death Note: themes like serial killers, fugitive, suspense; people who liked Death Note also liked this").
A second tab lets you describe what you want in plain English, e.g. *"dark psychological thriller under 25 episodes"*. It runs a meaning-based (embedding) search and applies the episode limit from your text automatically.

**🔗 Live demo:** [https://frisky-anime-recommender.streamlit.app/](https://frisky-anime-recommender.streamlit.app/)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://frisky-anime-recommender.streamlit.app/)

![screenshot](screenshot.png)

## How it works

| Piece | What it does |
|---|---|
| **Content-based model** | Turns each anime's title, genres, ~25 themes, and synopsis into a 384-number "meaning vector" with the `all-MiniLM-L6-v2` sentence-embedding model. Anime with similar vectors are similar in story and theme. A TF-IDF (keyword) version is included for comparison. |
| **Collaborative filtering** | Implicit **ALS** matrix factorization (`implicit` library) on ~830K MyAnimeList ratings from 812 users. It learns "people who liked X also liked Y" patterns. |
| **Hybrid** | Blends both scores after putting them on the same scale (z-score). Titles with few ratings lean more on content, which handles the cold-start problem. A small bonus goes to well-rated shows. |
| **Sequel filter** | Hides seasons, movies, and spin-offs of the shows you picked, and shows at most one title per franchise, so you don't get "Attack on Titan Season 2". |
| **Explanations** | Lists the rarest (most specific) themes shared with the closest seed show, plus shared genres. It adds "people who liked X also liked this" when the two shows sit close together in the ALS space. |

## Data

All of it comes from public Hugging Face datasets, downloaded with no login:

* [`krishnaveni76/Animes`](https://huggingface.co/datasets/krishnaveni76/Animes): MyAnimeList metadata (synopsis, genres, studios, score, members, episodes, image URL). 12,194 titles.
* [`krishnaveni76/UserRatings`](https://huggingface.co/datasets/krishnaveni76/UserRatings): 1.11M ratings (scale 5–10) from 813 MyAnimeList users.
* [`Mikami063/User-Animelist-Dataset`](https://huggingface.co/datasets/Mikami063/User-Animelist-Dataset) (`animes.csv` only): English titles and detailed theme tags, joined on the MyAnimeList ID.

After cleaning, the app uses **4,898 anime** (TV/Movie/OVA/ONA/Special, ≥5,000 MAL members, adult genres removed) and **829,934 ratings from 812 users**.

## Model comparison

Protocol: for every user, a random 20% of their ratings are hidden. A model sees the other 80% and must produce a top-10 list, excluding shows the user already rated. A **hit** is a hidden show the user rated **≥ 8/10**. Settings (ALS size, blend weight) were tuned on half the users, and the numbers below come from the **other 406 users** (`results/metrics.json`).

| Model | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---|---|
| Most popular (baseline) | 0.430 | 0.430 | 0.460 |
| Content: TF-IDF | 0.335 | 0.336 | 0.367 |
| Content: sentence embeddings | 0.109 | 0.109 | 0.115 |
| Collaborative filtering: ALS (32 factors) | 0.543 | 0.543 | 0.556 |
| **Hybrid (0.2 content + 0.8 ALS)** | **0.580** | **0.580** | **0.597** |

Takeaways:
* The hybrid beats the popularity baseline by about **+30% NDCG** and beats ALS alone by about +7%.
* Content-only scores below "most popular" on this test. These users each rated 1,000+ shows, so their hidden favourites skew toward popular titles, and "similar story" alone doesn't capture that. Content still helps when blended in, and it's what makes the free-text search and the recommendations for rarely-rated titles work.
* Precision and Recall are nearly identical because almost every test user has more than 10 hidden liked shows, so the recall denominator caps at 10.
* The numbers are high in absolute terms because each user has a very long history (avg ~1,000 ratings). They are best read as a *comparison* between models, not as real-world accuracy.

In the app, the default blend is 0.5 content / 0.5 ALS, adjustable with a slider. With only 3–5 picks, a pure-ALS list tends to show generic popular hits, so content gets more weight there than in the full-history evaluation.

## Project layout

```
prepare_data.py        downloads + cleans data  -> data/anime.parquet, data/ratings.parquet
build_models.py        embeddings, TF-IDF, ALS, hybrid, evaluation -> artifacts/, results/
recommender.py         shared recommendation, explanation, and text-search logic
app.py                 Streamlit web app
artifacts/             precomputed embeddings + ALS item vectors (small, committed)
results/metrics.json   evaluation numbers
EXPLAINED.md           plain-English walkthrough + interview Q&A
```

## Run it locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py                # opens http://localhost:8501
```

To rebuild everything from scratch (about 5 minutes on a laptop CPU):

```bash
pip install -r requirements-build.txt
python prepare_data.py
rm -f artifacts/content_emb.npy     # force re-embedding
python build_models.py
```

## Deploy free on Streamlit Community Cloud

This repo is ready to deploy as is. Everything the app needs is already committed: `app.py`, `recommender.py`, `requirements.txt`, `data/anime.parquet`, and `artifacts/`.

1. Go to **<https://share.streamlit.io>** and click **Continue with GitHub**. Sign in as the account that owns this repo.
2. If asked, authorize Streamlit to access your GitHub repositories.
3. Click **Create app**, then choose **"Deploy a public app from GitHub"**.
4. Fill in:
   * **Repository:** `<your-github-username>/anime-recommender`
   * **Branch:** `main`
   * **Main file path:** `app.py`
   * **App URL:** pick a name, e.g. `anime-recommender-<yourname>`
5. Open **Advanced settings** and set **Python version** to **3.12**. Leave secrets empty, because none are needed. (`runtime.txt` also says 3.12, but Streamlit Cloud uses the version picked here.)
6. Click **Deploy**. The first build takes about 3–5 minutes while it installs CPU-only PyTorch and downloads the small embedding model. Later loads are fast.
7. Copy the `https://….streamlit.app` URL into the **Live demo** line at the top of this README, and onto your resume and LinkedIn.

Notes: free apps go to sleep after a few days without visitors, and the first visitor after that clicks "Wake up". If the build fails, open **Manage app → logs**.

## Limitations / next steps

* The ratings come from only 812 (very active) users, so collaborative filtering is biased toward what heavy MAL users watch.
* The theme tags come from a community source and are sometimes noisy.
* Ideas for next steps: a time-based split, more users (e.g. the 148M-rating Mikami063 dataset), and an LLM-written "why" text.
