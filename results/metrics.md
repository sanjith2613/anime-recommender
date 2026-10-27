| Model | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---|---|
| Most popular (baseline) | 0.430 | 0.430 | 0.460 |
| Content: TF-IDF | 0.335 | 0.336 | 0.367 |
| Content: sentence embeddings | 0.109 | 0.109 | 0.115 |
| Collaborative filtering: ALS (32 factors) | 0.543 | 0.543 | 0.556 |
| Hybrid (0.2 content + 0.8 ALS) | 0.580 | 0.580 | 0.597 |
