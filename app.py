"""Step 3 - The web app. Run with:  streamlit run app.py"""
import streamlit as st

from recommender import Recommender, episode_phrase

st.set_page_config(page_title="Anime Recommender", page_icon="🎌", layout="wide")


@st.cache_resource
def load():
    # Precomputed vectors only. The embedding model stays unloaded until a vibe search.
    return Recommender()


rec = load()
df = rec.df

st.title("🎌 Anime Recommender")
st.caption(
    f"Pick a few anime you love and get 10 picks. Hybrid model: precomputed sentence embeddings "
    f"+ collaborative filtering (ALS) over {len(df):,} MyAnimeList titles."
)


def show_cards(cards):
    for start in range(0, len(cards), 5):
        cols = st.columns(5)
        for col, card in zip(cols, cards[start:start + 5]):
            with col:
                if card["image_url"]:
                    st.image(card["image_url"], use_container_width=True)
                st.markdown(f"**{card['title']}**")
                meta = []
                if card["score"]:
                    meta.append(f"⭐ {card['score']:.2f}")
                meta.append(card["type"])
                phrase = episode_phrase(card["episodes"])
                if phrase:
                    meta.append(phrase)
                st.caption(" · ".join(meta))
                st.caption(card["genres"])
                st.info(card["why"])
                with st.expander("Synopsis"):
                    st.write(card["synopsis"])


tab1, tab2 = st.tabs(["Recommend from favorites", "Describe what you want"])

with tab1:
    options = df["display"].tolist()  # sorted by popularity, so famous shows come first
    default = [o for o in options if o.startswith(("Attack on Titan (Shingeki", "Death Note", "Steins;Gate"))][:3]
    picks = st.multiselect(
        "Your favorite anime (pick 3-5, type to search)",
        options,
        default=default,
        max_selections=5,
    )
    c1, c2, c3 = st.columns(3)
    hide = c1.checkbox("Hide sequels / same franchise", value=True)
    max_ep = c2.number_input("Max episodes (0 = any)", min_value=0, value=0, step=1)
    w = c3.slider(
        "Content ↔ crowd balance",
        0.0,
        1.0,
        float(rec.cfg["hybrid_weight_content"]),
        0.1,
        help="1.0 = only story and genre similarity. 0.0 = only people with similar taste. "
             "The starting point is the blend that scored best in the full-history evaluation.",
    )
    if len(picks) < 1:
        st.warning("Pick at least one anime (3-5 works best).")
    else:
        if len(picks) < 3:
            st.caption("Tip: 3-5 picks gives better results.")
        ids = df.loc[df["display"].isin(picks), "anime_id"].tolist()
        cards = rec.recommend(ids, k=10, hide_franchise=hide, max_episodes=max_ep or None, w_content=w)
        st.subheader("Your 10 recommendations")
        show_cards(cards)

with tab2:
    q = st.text_input("Describe a show", "dark psychological thriller under 25 episodes")
    st.caption("TV series by default, starting at season 1. Words like short, movie, horror, or under 12 episodes are applied as filters.")
    c1, c2 = st.columns(2)
    ep = c1.number_input("Max episodes (0 = read it from the text)", min_value=0, value=0, step=1)
    ms = c2.slider("Minimum MyAnimeList score", 0.0, 9.0, 7.0, 0.5)
    if q.strip():
        with st.spinner("Searching..."):
            try:
                cards, info = rec.text_search(q, k=10, max_episodes=ep or None, min_score=ms)
            except Exception as exc:
                st.error(f"The search model could not be loaded ({exc}). Favorite-based recommendations still work.")
                cards, info = [], {}
        bits = []
        if info.get("types"):
            bits.append("format: " + ", ".join(info["types"]))
        if info.get("max_episodes"):
            bits.append(f"≤ {info['max_episodes']} episodes")
        if info.get("min_episodes"):
            bits.append(f"≥ {info['min_episodes']} episodes")
        if bits:
            st.caption("Filters from your description: " + "; ".join(bits))
        show_cards(cards)

st.divider()
st.caption(
    "Catalog metadata: Quan Than's Apache-2.0 Top 15,000 Ranked Anime dataset (March 2025) and "
    "lyfesan's MIT-licensed MyAnimeList genre and theme list (September 2025). "
    "Ratings: krishnaveni76/UserRatings."
)
