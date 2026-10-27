# The Anime Recommender, Explained Simply

This guide is for you if you've never coded. It explains what each part does, why it's there, and how to talk about it in an interview.

---

## 1. The big idea

Netflix, Spotify, and YouTube all answer the same question: *"Given what this person likes, what will they like next?"*
There are two classic ways to answer it, and this project uses both:

1. **"This show is similar to that show"** (content-based). You look at the shows themselves: story, genres, themes.
2. **"People like you also liked…"** (collaborative filtering). You look at what *other viewers* rated, and ignore the story completely.

Mixing both is called a **hybrid recommender**. That's what the app does.

---

## 2. The data (where the information comes from)

* **MyAnimeList (MAL)** is the largest anime-tracking website. People log every show they watch and rate it.
* People have published copies of MAL data on **Hugging Face**, a free site for sharing datasets and AI models. We downloaded three files from there, with no login needed:
  * **Animes.csv**: one row per anime with its title, synopsis (plot summary), genres, studio, average score, number of members, episode count, and poster image link.
  * **UserRatings.csv**: about 1.1 million rows like *"user 357 gave anime 9253 (Steins;Gate) a 9"*, from 813 real MAL users.
  * **animes.csv** (a second dataset): used only for **English titles** and detailed **theme tags** like "time travel" and "serial killers".
* **Cleaning** means throwing out what we don't want: very obscure shows (fewer than 5,000 members), music videos, adult content, and shows with no synopsis. That leaves **4,898 anime** and **~830,000 ratings**.

---

## 3. Model A: content-based with "embeddings"

**What's an embedding?** An AI model reads a piece of text and turns it into a list of 384 numbers. Think of those numbers as GPS coordinates on a "meaning map". Texts that *mean* similar things land close together, even when they use different words. "A boy finds a notebook that kills people" lands near "a detective hunts a serial killer", even though they share no words.

* The embedding model we used is called **all-MiniLM-L6-v2**. It's small, free, and fast enough to run on a normal computer.
* For each anime we feed it: *title + genres + themes + synopsis*, and it gives back 384 numbers.
* To compare two anime we use **cosine similarity**: how much the two arrows point in the same direction. 1.0 means nearly identical meaning, 0 means unrelated.
* **TF-IDF** is the older, simpler alternative. It just counts words, giving more weight to rare words like "titan" than to common ones like "the". We built it too, for comparison.

**Strength:** works even for a brand-new or rarely watched show, as long as it has a description.
**Weakness:** knows nothing about quality or what people actually enjoy together.

---

## 4. Model B: collaborative filtering with ALS

**The idea:** if lots of people who loved *Death Note* also loved *Psycho-Pass*, then recommend *Psycho-Pass* to the next *Death Note* fan, without ever reading the plots.

* Picture a giant table: 812 users down the side, 4,898 anime across the top, and each cell holds that user's rating. Most cells are empty, because nobody has watched everything.
* **Matrix factorization** squeezes this huge, mostly empty table into two small tables:
  * each **user** gets 32 hidden "taste" numbers,
  * each **anime** gets 32 hidden "flavour" numbers.
  * The model adjusts these numbers until *user taste × anime flavour* reproduces the ratings we know.
* Nobody tells the model what the 32 numbers mean. It discovers patterns on its own, like "dark & serious", "cute & romantic", or "long shounen action".
* **ALS (Alternating Least Squares)** is the training method. It fixes the user numbers and solves for the anime numbers, then flips and repeats about 20 times. It's fast and reliable. We used the `implicit` Python library.
* "32 factors" was chosen by trying 32, 64, and 128 and keeping whichever scored best on a validation group.

**Strength:** captures real taste and quality.
**Weakness:** useless for shows with few ratings (the **"cold-start" problem**).

---

## 5. Model C: the hybrid

* Each model gives every anime a score. The scores are on different scales, so we first **standardize** them (a "z-score": how many steps above or below average).
* Then: `final = w × content + (1 − w) × collaborative`. In testing, w = 0.2 worked best. In the app it defaults to 0.5, and you can move the **slider**.
* If a show has fewer than 50 ratings, the collaborative part is trusted less, so the content part takes over. That's how the cold-start problem is handled.
* A small bonus goes to shows with a higher MAL score.

---

## 6. The extras that make results feel smart

* **Sequel filter:** if you pick *Attack on Titan*, you don't want "Attack on Titan Season 2" as a suggestion. A small rule strips words like "Season 2", "Movie", "OVA" from titles to find the franchise, then hides those and keeps one show per franchise.
* **The "why" line:** for each recommendation we find which of *your* picks it's closest to, then list the most *specific* themes they share (rare tags like "serial killers" beat common tags like "action") and their shared genres. If the collaborative model says the two have the same fans, it adds "people who liked X also liked this".
* **Free-text search:** your sentence ("dark psychological thriller under 25 episodes") gets turned into an embedding and compared with all 4,898 anime. A simple pattern-matcher spots "under 25 episodes" and applies it as a filter.

---

## 7. How we know it works (evaluation)

You can't just eyeball recommendations, so we ran a test:

1. For each user, **hide 20%** of their ratings.
2. Let each model see the other 80% and produce a **top 10** list.
3. Count how many of those 10 are hidden shows the user actually rated **8 or higher**.

We did this for 406 users the model was never tuned on.

* **Precision@10:** out of 10 suggestions, what fraction were hits. 0.58 means about 6 of 10.
* **Recall@10:** out of the hidden favourites, what fraction we found (capped at 10).
* **NDCG@10:** like precision, but hits near the **top** of the list count more. Getting the #1 slot right matters more than the #10 slot.
* **Baseline:** "just recommend the most popular shows". Any real model has to beat this, or it's pointless.

| Model | Precision@10 | NDCG@10 |
|---|---|---|
| Most popular (baseline) | 0.430 | 0.460 |
| Content: TF-IDF | 0.335 | 0.367 |
| Content: embeddings | 0.109 | 0.115 |
| Collaborative (ALS) | 0.543 | 0.556 |
| **Hybrid** | **0.580** | **0.597** |

**In plain words:** the hybrid is about 30% better than "just show popular stuff", and better than either model alone. Content-only loses to popularity in this test, which is an honest and interesting finding: these users are heavy watchers whose favourites lean toward popular shows. Content still adds value in the blend and powers the text search.

---

## 8. What each file does

| File | Plain-English job |
|---|---|
| `prepare_data.py` | Downloads the three datasets, cleans them, and saves two tidy tables in `data/`. |
| `build_models.py` | Creates the embeddings, trains ALS, tries settings, runs the test above, and saves the results to `results/` and the trained pieces to `artifacts/`. |
| `recommender.py` | The "brain": combines the scores, filters sequels, writes the "why" line, and runs the text search. |
| `app.py` | The website (built with **Streamlit**, a Python library that turns a script into a web page). Has the multiselect box, sliders, and recommendation cards. |
| `artifacts/` | The saved embeddings (`content_emb.npy`) and ALS anime vectors (`cf_item_factors.npy`), so the app doesn't retrain each time it starts. |
| `results/metrics.json` | The evaluation numbers. |
| `requirements.txt` | The list of Python packages the app needs, which the free hosting service reads. |
| `README.md` | The project's front page on GitHub. |

---

## 9. Likely interview questions (with simple answers)

**1. Walk me through the project.**
"I built a hybrid anime recommender on MyAnimeList data: 4,900 anime and 830K ratings. It combines a content model using sentence embeddings with collaborative filtering using ALS matrix factorization. It's served as a Streamlit app where you pick favourites or describe what you want in plain English. The hybrid beat a popularity baseline by about 30% on NDCG@10."

**2. Why a hybrid instead of one model?**
"Collaborative filtering is most accurate but fails for titles with few ratings. Content-based works for any title with a description. Blending them gave the best test score (NDCG 0.60 vs 0.56 for ALS alone), and it handles cold-start."

**3. What is an embedding, and why all-MiniLM-L6-v2?**
"It's a numeric vector that captures the meaning of text, so similar meanings end up close together. MiniLM is small, fast on CPU, free, and good at semantic similarity. That made it practical for 4,900 synopses and for embedding a user's search query live."

**4. How does ALS work?**
"It factorizes the sparse user × item rating matrix into low-dimensional user and item vectors. It alternates: fix users and solve for items with least squares, then fix items and solve for users. I treated ratings as confidence weights (implicit feedback) and tuned factors, regularization, and alpha on a validation set."

**5. How did you evaluate it?**
"Per-user random 20% holdout. A hit is a held-out title rated ≥ 8. I measured Precision@10, Recall@10, and NDCG@10, tuned on half the users, and reported on the other half, all compared against a most-popular baseline."

**6. Why compare to a popularity baseline?**
"Popularity is surprisingly hard to beat in recommenders. If a model can't beat 'show everyone the top shows', it isn't adding value. Here the content-only model actually lost to it, which is why I blended it in instead of using it alone."

**7. What is the cold-start problem, and how did you handle it?**
"New or rarely rated items, or new users, have no interaction data. I down-weight the collaborative score for titles with under 50 ratings so content similarity takes over. A new user just picks a few favourites, and we score from those directly."

**8. Why is cosine similarity used?**
"It measures the angle between vectors, not their length. So it compares *direction of meaning*, regardless of how long a synopsis is or how many ratings an item has."

**9. What would you improve with more time?**
"More users (there's a 148M-rating public dataset), a time-based split so we predict future watches, A/B testing with real users, diversity in the top 10, and an LLM to write nicer explanations."

**10. What were the limitations or surprises?**
"Only 812 users, all very active, so it's biased toward heavy-watcher taste and the absolute metrics are optimistic. Theme tags were noisy, so I filtered out meta tags and ranked shared themes by rarity for the explanations. A simple rule-based sequel filter was necessary, because otherwise the top results were just 'Season 2' of what you picked."

---

## 10. Words to know (cheat sheet)

* **Dataset**: a table of information.
* **Model**: a program that learned patterns from data.
* **Vector**: a list of numbers.
* **Embedding**: a vector that represents meaning.
* **Cosine similarity**: how similar two vectors' directions are (−1 to 1).
* **Matrix factorization**: compressing a big table into two small ones.
* **Hyperparameters**: settings you choose before training (e.g. 32 factors).
* **Validation vs test set**: validation is for picking settings, and test is for the final, honest score.
* **Baseline**: the simple approach your model must beat.
* **Streamlit**: turns a Python script into a website.
* **Deploy**: put it online so anyone can use it.
