"""Smoke tests for recommendations, explanations, vibe search, and franchises."""
import json

import pytest

from recommender import (
    Recommender,
    build_franchise_ids,
    episode_phrase,
    franchise_key,
    parse_vibe,
    season_index,
)


def test_episode_phrase():
    assert episode_phrase(1) == "1 episode"
    assert episode_phrase(12) == "12 episodes"
    assert episode_phrase(None) is None


def test_franchise_keys_collapse_sequels_only():
    assert franchise_key("Major S1") == franchise_key("Major S2") == franchise_key("Major S3")
    assert franchise_key("Psycho-Pass") == franchise_key("Psycho-Pass 2") == franchise_key("Psycho-Pass: The Movie")
    assert franchise_key("The Prince of Tennis") == franchise_key("The Prince of Tennis II")
    assert franchise_key("Spy x Family") == franchise_key("Spy x Family Part 2")
    assert franchise_key("Clannad") == franchise_key("Clannad: After Story")
    assert franchise_key("Monster") != franchise_key("Monster Musume no Iru Nichijou")
    assert franchise_key("Fate/Zero") != franchise_key("Fate/stay night")
    assert franchise_key("Mobile Suit Gundam") != franchise_key("Mobile Suit Gundam 00")
    assert franchise_key("Samurai Champloo") != franchise_key("Samurai Flamenco")
    assert franchise_key("Samurai 7") != franchise_key("Samurai X")
    assert franchise_key("Initial D First Stage") == franchise_key("Initial D Second Stage")
    assert franchise_key("Initial D Fifth Stage") == franchise_key("Initial D Extra Stage 2")
    assert season_index("Psycho-Pass") == 1
    assert season_index("Psycho-Pass 2") == 2
    assert season_index("Major S3") == 3
    assert season_index("Jujutsu Kaisen 2nd Season") == 2
    assert season_index("Spy x Family Part 2") == 2
    # English and Japanese titles of one sequel link the whole franchise.
    ids = build_franchise_ids(
        ["The Prince of Tennis", "The Prince of Tennis II"],
        ["Tennis no Ouji-sama", "Shin Tennis no Ouji-sama"],
    )
    assert ids[0] == ids[1]


def test_vibe_parser_reads_format_length_and_genre():
    short = parse_vibe("short horror series")
    assert short["types"] == {"TV"}
    assert short["max_episodes"] == 13
    assert {"Horror"} in short["require"]

    thriller = parse_vibe("dark psychological thriller")
    assert thriller["types"] == {"TV"}
    assert thriller["max_episodes"] is None
    assert {"Psychological"} in thriller["require"]
    assert {"Suspense", "Thriller"} in thriller["require"]

    movie = parse_vibe("a movie about samurai")
    assert movie["types"] == {"Movie"}
    assert parse_vibe("under 25 episodes")["max_episodes"] == 25
    assert parse_vibe("long fantasy series")["min_episodes"] == 24


@pytest.fixture(scope="module")
def rec():
    return Recommender()


def _id(rec, query):
    hits = rec.df[rec.df["display"].str.contains(query, regex=False)]
    assert len(hits), query
    return int(hits.iloc[0]["anime_id"])


def _franchise(rec, anime_id):
    return int(rec.franchise[rec.idx[anime_id]])


def test_catalog_includes_recent_shows(rec):
    blob = (rec.df["title"].fillna("") + " " + rec.df["display"].fillna("")).str.lower()
    for needle in ("kimetsu", "jujutsu kaisen", "spy x family", "frieren", "kaguya-sama"):
        assert blob.str.contains(needle, regex=False).any(), needle


def test_recommend_smoke_and_explanations(rec):
    seeds = [_id(rec, "Death Note"), _id(rec, "Steins;Gate"), _id(rec, "Clannad")]
    # Death Note the series, not the relight special, is the first popularity hit.
    cards = rec.recommend(seeds, k=10)
    assert len(cards) == 10
    assert len({c["title"] for c in cards}) == 10
    blocked = {_franchise(rec, seed) for seed in seeds}
    for card in cards:
        assert _franchise(rec, card["anime_id"]) not in blocked
        why = card["why"].lower()
        for junk in ("basketball", "baseball", "united states", "mundane made awesome", "colour coded"):
            assert junk not in why
        assert card["why"].startswith("Similar to ")
    # The crowd line used to be on every card. It should be rare.
    assert sum("also liked" in card["why"] for card in cards) <= 3
    franchises = [_franchise(rec, card["anime_id"]) for card in cards]
    assert len(franchises) == len(set(franchises))


def test_reported_franchises_collapse(rec):
    assert _franchise(rec, _id(rec, "Major S2")) == _franchise(rec, _id(rec, "Major S3"))
    assert _franchise(rec, _id(rec, "The Prince of Tennis (")) == _franchise(rec, _id(rec, "The Prince of Tennis II ("))
    assert _franchise(rec, _id(rec, "Psycho-Pass")) == _franchise(rec, _id(rec, "Psycho-Pass 2"))
    assert _franchise(rec, _id(rec, "Kimetsu no Yaiba)")) == _franchise(
        rec, _id(rec, "Entertainment District Arc")
    )
    assert _franchise(rec, _id(rec, "Samurai 7")) != _franchise(rec, _id(rec, "Rurouni Kenshin"))
    assert _franchise(rec, _id(rec, "Initial D First Stage")) == _franchise(
        rec, _id(rec, "Initial D Second Stage")
    )
    assert _franchise(rec, _id(rec, "Tokyo Ghoul")) == _franchise(rec, _id(rec, "Tokyo Ghoul √A"))


def test_new_show_gets_content_recommendations(rec):
    demon = _id(rec, "Demon Slayer: Kimetsu no Yaiba (Kimetsu")
    cards = rec.recommend([demon], k=10, w_content=1.0)
    assert len(cards) == 10
    assert _franchise(rec, demon) not in {_franchise(rec, c["anime_id"]) for c in cards}
    shared = 0
    for card in cards:
        genres = card["genres"].lower()
        if any(word in genres for word in ("action", "supernatural", "fantasy", "adventure")):
            shared += 1
        assert "basketball" not in card["why"].lower()
    assert shared >= 6


def test_clannad_explanation_uses_real_tags(rec):
    cards = rec.recommend([_id(rec, "Clannad")], k=10, w_content=0.8)
    blob = " ".join(card["why"].lower() for card in cards)
    assert "basketball" not in blob
    assert "baseball" not in blob
    assert "school" in blob or "romance" in blob or "drama" in blob


def test_vibe_search_respects_format_length_and_season(rec):
    horror, horror_info = rec.text_search("short horror series", k=10, min_score=7)
    assert horror_info["types"] == ["TV"]
    assert horror_info["max_episodes"] == 13
    assert len(horror) >= 8
    for card in horror:
        assert card["type"] == "TV"
        assert card["episodes"] <= 13
        assert "Horror" in card["genres"]
        assert " 1 episodes" not in f" {card['why']}"
        assert card["title"] != "Monster"

    thriller, info = rec.text_search("dark psychological thriller", k=10, min_score=7)
    assert info["types"] == ["TV"]
    assert len(thriller) == 10
    for card in thriller:
        assert card["type"] == "TV"
        assert not card["title"].startswith("Psycho-Pass 2")
        assert "Movie" not in card["type"]
    titles = [card["title"] for card in thriller]
    if any(title.startswith("Psycho-Pass") for title in titles):
        assert any(title.startswith("Psycho-Pass") and "2" not in title[:12] for title in titles)

    movies, movie_info = rec.text_search("a ghibli movie", k=5, min_score=0)
    assert movie_info["types"] == ["Movie"]
    assert movies
    for card in movies:
        assert card["type"] == "Movie"
        if card["episodes"] == 1:
            assert "1 episode" in card["why"]
            assert "1 episodes" not in card["why"]


def test_default_weight_matches_evaluation(rec):
    metrics = json.load(open("results/metrics.json"))
    assert rec.cfg["hybrid_weight_content"] == metrics["hybrid_weight_content"]
    assert rec.cfg["eval_best_weight_content"] == metrics["hybrid_weight_content"]
    assert "few_seed" in metrics
    # Recall is no longer capped at 10, so it is not a copy of precision.
    pairs = [(row["precision@10"], row["recall@10"]) for row in metrics["results"].values()]
    assert any(abs(precision - recall) > 0.02 for precision, recall in pairs)


def test_embedding_toolkit_stays_out_of_startup():
    import sys
    assert "torch" not in sys.modules
    assert "sentence_transformers" not in sys.modules
