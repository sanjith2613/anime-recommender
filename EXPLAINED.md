# The Anime Recommender, Explained Simply

This guide is for you if you've never coded. It explains what each part does, why it's there, and how to talk about it in an interview.

---

## 1. The big idea

Netflix, Spotify, and YouTube all answer the same question: *"Given what this person likes, what will they like next?"*
There are two classic ways to answer it, and this project uses both:

1. **"This show is similar to that show"** (content-based). You look at the shows themselves: story, genres, themes.
2. **"People like you also liked…"** (collaborative filtering). You look at what *other viewers* rated, and ignore the story completely.

Mixing both is called a **hybrid recommender**. That's what the app does. You pick 3–5 favorites and get 10 suggestions, each with a short reason. A second box lets you describe a vibe in ordinary words.

The live site is [https://frisky-anime-recommender.streamlit.app/](https://frisky-anime-recommender.streamlit.app/).

---

## 2. The data (where the information comes from)

* **MyAnimeList (MAL)** is the largest anime-tracking website. People log shows they watch and rate them.
* The catalog is **7,423 anime**, from 1958 through 2025, including Demon Slayer, Jujutsu Kaisen, Spy x Family, Frieren, and Kaguya-sama. The ratings are **1,015,487** scores from **813** real MAL users.
* The files are public. No login is required to download them:
  * **Top 15,000 Ranked Anime** by Quan Than (Kaggle, updated March 2025, Apache 2.0). We use a cleaned copy on Hugging Face (`TheRealVigilante/Processed_Top_15k_Anime`): title, plot summary, poster, score, member count, type, and episode count.
  * **lyfesan/myanimelist-top-anime-dataset** (Hugging Face, September 2025, MIT): the official MAL genre, theme, and audience labels. These are the words that show up in the "why" line.
  * **krishnaveni76/UserRatings** (Apache 2.0): the ratings, each row like *"user 357 gave Steins;Gate a 9"*.
  * **krishnaveni76/Animes** (Apache 2.0): used only to fill in a few older rated titles that the 2025 list does not have.
* **Cleaning** drops very obscure shows (fewer than 5,000 members), music videos, adult titles, and rows with no real synopsis.

Newer shows are pickable even when few of these 813 people have rated them. In that case the app leans on the description.

---

## 3. Model A: content-based with "embeddings"

**What's an embedding?** An AI model reads a piece of text and turns it into a list of 384 numbers. Think of those numbers as GPS coordinates on a "meaning map". Texts that *mean* similar things land close together, even when they use different words. "A boy finds a notebook that kills people" lands near "a detective hunts a serial killer", even though they share no words.

* The embedding model is **all-MiniLM-L6-v2**. It's small, free, and good at this kind of matching.
* For each anime we feed it: *title + genres + themes + audience + synopsis*.
* Those 384 numbers are **saved in advance** (`artifacts/content_emb.npy`). Opening the website does not load the AI model.
* To compare two anime we use **cosine similarity**: how much the two arrows point in the same direction. 1.0 means nearly identical meaning, 0 means unrelated.
* **TF-IDF** is the older, simpler alternative. It counts words, giving more weight to rare words like "titan" than to common ones like "the". We built it for the comparison table. The app itself uses the embeddings.

**Strength:** works for a brand-new show, as long as it has a description.
**Weakness:** knows little about what people actually enjoy together. On the big test, embeddings alone lose to "just show popular anime".

When someone types a vibe, the same MiniLM model turns that sentence into numbers. That copy is an ONNX file (a lightweight format), loaded only on the first search, so the favorites page stays fast and the site does not need PyTorch.

---

## 4. Model B: collaborative filtering with ALS

**The idea:** if lots of people who loved *Death Note* also loved *Psycho-Pass*, then recommend *Psycho-Pass* to the next *Death Note* fan, without reading the plots.

* Picture a giant table: 813 users down the side, thousands of anime across the top, and each cell holds that user's rating. Most cells are empty.
* **Matrix factorization** squeezes this huge, mostly empty table into two small tables:
  * each **user** gets 32 hidden "taste" numbers,
  * each **anime** gets 32 hidden "flavour" numbers.
  * The model adjusts these numbers until *user taste × anime flavour* reproduces the ratings we know.
* Nobody tells the model what the 32 numbers mean. It discovers patterns on its own.
* **ALS (Alternating Least Squares)** is the training method. It fixes the user numbers and solves for the anime numbers, then flips and repeats about 20 times. We used the `implicit` Python library.
* "32 factors", a regularization of 0.05, and an alpha of 1.0 were chosen by trying a small grid and keeping what scored best on a validation group. Ratings are treated as confidence ("implicit" feedback): a 10 counts as a stronger signal than a 6.

**Strength:** captures real taste.
**Weakness:** weak for shows almost nobody in this file has rated (the **"cold-start" problem**).

---

## 5. Model C: the hybrid

* Each model gives every anime a score. The scores are on different scales, so we first **standardize** them (a "z-score": how many steps above or below average).
* Then: `final = 0.2 × content + 0.8 × crowd`, plus a small bonus for a higher MAL score.
* **0.2 / 0.8 is the default in the app**, because that mix won the full-history test below. The slider starts there. You can drag it.
* A second test, closer to how the site is used (only 3–5 favorites), prefers **0.1 content / 0.9 crowd**. That is one notch over on the slider. The two winners are neighbors, so the page starts at the headline result (0.2).
* If a show has fewer than 50 ratings in our file, the crowd score is trusted less (the trust grows smoothly from 0 ratings up to 50). The description takes over. That is how a new show can still appear.

---

## 6. The extras that make results feel smart

* **One show per franchise.** "Major S2" and "Major S3" are the same franchise, and so are both Prince of Tennis TV series, Psycho-Pass and Psycho-Pass 2, and the Demon Slayer arcs. The list keeps the best-matching entry and the card shows where to start: usually the first TV season, not a movie or a later season. Franchises you already picked are hidden. The match is based on the title (English and Japanese), so a side story with a totally different name can still slip through.
* **The "why" line** uses official MAL genres and themes, such as School, Psychological, or Time Travel. It anchors the sentence to the favorite that shares the most of those labels. It does **not** use the old noisy tags that produced lines like "basketball" for Clannad.
* **"People who liked X also liked this"** used to appear on every card, because almost every popular show sits near every other popular show in the ALS space (these 813 people watched a lot of the same hits). It now appears only when the shared fandom is genuinely strong: at least 50 people rated both shows 8 or higher, and that overlap is at least 1.45 times what their individual popularity would predict. On a typical Death Note + Steins;Gate + Clannad list, the note shows up rarely or not at all. The useful part of the line is the shared theme.
* **Vibe search** defaults to **TV series** and prefers season 1 of a franchise. "Short" means one cour, 13 episodes or fewer. "Movie", "OVA", "long", and "under 25 episodes" are read as filters. "Horror" and "psychological thriller" require those genres or themes, not just a similar sentence. Counts read "1 episode" or "12 episodes".

Examples after the fix:

* Favorites Death Note, Steins;Gate, and Clannad: "Similar to Clannad: themes like School; shared genres Romance, Drama" and "Similar to Steins;Gate: themes like Psychological; shared genres Suspense, Sci-Fi". No basketball, baseball, or "united states".
* "Dark psychological thriller": TV only, including Psycho-Pass (22 episodes, season 1) and Death Note. Not Psycho-Pass 2, and not a movie.
* "Short horror series": TV, 13 episodes or fewer, Horror on the card (Another, Hellsing, Mononoke). Not Monster (74 episodes) and not Clannad.

---

## 7. How we know it works (evaluation)

You can't just eyeball recommendations, so we ran two tests. Both use the same idea of a hit: a show the user rated **8 or higher** that the model was not allowed to see.

### Test 1: full watch history

1. For each user, **hide 20%** of their ratings at random.
2. Let each model see the other 80% and produce a **top 10**.
3. Settings were chosen on half the users. The numbers below are the other **407** users.

* **Precision@10:** out of 10 suggestions, what fraction were hits. 0.60 means about 6 of 10.
* **Recall@10:** hits divided by *all* hidden favorites. These users often have hundreds of hidden 8+ ratings and we only have 10 slots, so recall is small (around 0.06) even when precision is high. An earlier version capped that denominator at 10, which made recall a copy of precision. That is fixed.
* **NDCG@10:** like precision, but a hit in slot 1 counts more than a hit in slot 10.
* **Baseline:** "just recommend the most popular shows". A real model has to beat this.

| Model | Precision@10 | Recall@10 | NDCG@10 |
|---|---:|---:|---:|
| Most popular | 0.443 | 0.043 | 0.471 |
| Content: TF-IDF | 0.252 | 0.028 | 0.268 |
| Content: embeddings | 0.084 | 0.009 | 0.090 |
| Collaborative (ALS) | 0.572 | 0.059 | 0.583 |
| **Hybrid (0.2 / 0.8)** | **0.600** | **0.063** | **0.614** |

**In plain words:** the hybrid is about 30% higher than popularity on NDCG (0.614 vs 0.471) and a bit higher than ALS alone. Content-only loses to popularity here. That is an honest finding: these are heavy watchers, and their hidden favorites lean famous. Content still helps the blend, and it powers search and new shows.

### Test 2: 3–5 favorites (how the app is actually used)

Each user contributes only their top 3, 4, or 5 training titles rated 8 or higher. Scoring matches the website, including the "trust the crowd less under 50 ratings" rule. Hits are still the hidden 8+ titles.

| Model | Precision@10 | Recall@10 | NDCG@10 |
|---|---:|---:|---:|
| Most popular | 0.443 | 0.043 | 0.471 |
| Content: embeddings | 0.090 | 0.010 | 0.103 |
| ALS | 0.486 | 0.051 | 0.516 |
| **Hybrid (0.1 / 0.9)** | **0.488** | **0.051** | **0.521** |
| Hybrid at the app default (0.2) | 0.480 | 0.050 | 0.511 |

With only a handful of favorites, the crowd signal still does most of the work. Adding a little content is a small gain, not a transformation. The old slider default of 0.5 was a guess that short lists needed more content. This test says 0.1 is best and 0.2 is close, so the page starts at 0.2 to match the headline evaluation.

The absolute scores are optimistic because every test user is a power user. Read them as "which model wins", not as "60% of strangers will love the list".

---

## 8. What each file does

| File | Plain-English job |
|---|---|
| `prepare_data.py` | Downloads the datasets, cleans them, and saves two tidy tables in `data/`. |
| `build_models.py` | Creates the embeddings, trains ALS, tries settings, runs both tests, and saves `results/` and `artifacts/`. |
| `recommender.py` | The brain: combines the scores, groups franchises, writes the why line, and runs vibe search. |
| `app.py` | The website (Streamlit). The multiselect, the slider, and the cards. |
| `artifacts/` | Saved embeddings, ALS vectors, and the fan lists used for the crowd note. The app does not retrain on startup. |
| `results/metrics.json` | Both evaluation tables. |
| `tests/test_recommender.py` | A short automatic check: franchises, why-lines, vibe filters, the slider default, and that startup does not load PyTorch. |
| `.github/workflows/test.yml` | Runs that check on GitHub whenever code is pushed. |
| `requirements.txt` | Packages the website needs. No PyTorch. |
| `README.md` | The project's front page on GitHub. |

Startup is a fraction of a second locally because it only reads the saved tables. The search model downloads the first time someone uses the vibe box.

---

## 9. Likely interview questions (with simple answers)

**1. Walk me through the project.**
"I built a hybrid anime recommender on public MyAnimeList data: about 7,400 anime and a million ratings from 813 users. It combines sentence embeddings of each synopsis with ALS collaborative filtering, served as a Streamlit app. You pick 3–5 favorites or describe a vibe. On a held-out test the hybrid's NDCG@10 is 0.614, about 30% above a popularity baseline."

**2. Why a hybrid instead of one model?**
"Collaborative filtering is the most accurate on this data but is weak for titles with few ratings. Content works for any title with a description, including recent shows. The blend won the full-history test (NDCG 0.614 vs 0.583 for ALS alone). With only 3–5 seeds the best mix is even more crowd-weighted, 0.1 content."

**3. What is an embedding, and why all-MiniLM-L6-v2?**
"It's a numeric vector that captures the meaning of text, so similar meanings end up close together. MiniLM is small, free, and good at semantic similarity. The anime vectors are precomputed. A live query uses the ONNX build through fastembed, so the deployed app does not install PyTorch."

**4. How does ALS work?**
"It factorizes the sparse user-by-item rating matrix into 32-dimensional user and item vectors. It alternates: fix users and solve for items with least squares, then fix items and solve for users. I treated ratings as confidence weights and tuned factors, regularization, and alpha on a validation half. The winner was 32 factors, regularization 0.05, alpha 1."

**5. How did you evaluate it?**
"Two protocols. First, a per-user random 20% holdout, relevant if the hidden rating is at least 8. I report Precision@10, Recall@10, and NDCG@10 on 407 users the tuner never saw, against a most-popular baseline. Recall divides by every held-out favorite, so it is much smaller than precision. Second, the same hits, but each user only contributes 3–5 favorites, which matches the app."

**6. Why compare to a popularity baseline?**
"Popularity is surprisingly hard to beat. If a model can't beat 'show everyone the top shows', it isn't adding value. Content-only lost to it. The hybrid beat it."

**7. What is the cold-start problem, and how did you handle it?**
"Rarely rated items have no interaction data. I scale the collaborative score by how many ratings the title has, full trust at 50 ratings and none at zero, so the description takes over. A new visitor doesn't need an account: they pick a few favorites and we score from those."

**8. Why is cosine similarity used?**
"It measures the angle between vectors, not their length, so a long synopsis doesn't automatically look more similar than a short one."

**9. Why doesn't every card say 'people who liked X also liked this'?**
"With only 813 heavy users, the ALS vectors of popular shows all point roughly the same way, so a similarity cutoff fired on every recommendation and carried no information. The line now requires a real co-like lift: enough shared high ratings, and more overlap than popularity alone predicts."

**10. What would you improve with more time?**
"A much larger rating file so new shows get a crowd signal, a time-based split, a diversity pass in the top 10, and a cleaner franchise list for side stories whose titles don't look like sequels."

**11. What were the limitations or surprises?**
"Only 813 users, all very active, so the model leans toward heavy-watcher taste and the absolute metrics are optimistic. Official genre labels fixed nonsense theme tags. A title-based franchise rule was necessary, or the top results were Season 2 of what you picked. And with 3–5 favorites, extra content weight did not help: 0.1 beat 0.5."

---

## 10. Words to know (cheat sheet)

* **Dataset**: a table of information.
* **Model**: a program that learned patterns from data.
* **Vector**: a list of numbers.
* **Embedding**: a vector that represents meaning.
* **Cosine similarity**: how similar two vectors' directions are (0 to 1 here, after we normalize them).
* **Matrix factorization**: compressing a big table into two small ones.
* **ALS**: the training method that alternates between solving for users and solving for items.
* **Hyperparameters**: settings you choose before training (for example, 32 factors).
* **Validation vs test set**: validation is for picking settings, and test is for the final, honest score.
* **Precision@10**: fraction of the 10 suggestions that were hits.
* **Recall@10**: fraction of all hidden favorites that landed in those 10.
* **NDCG@10**: precision that gives more credit to hits near the top.
* **Baseline**: the simple approach your model must beat.
* **Cold start**: a show or a person with too little history for collaborative filtering.
* **Lift**: shared fans divided by how many shared fans popularity alone would predict.
* **Streamlit**: turns a Python script into a website.
* **Deploy**: put it online so anyone can use it.
