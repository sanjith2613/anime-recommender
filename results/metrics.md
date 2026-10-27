## Full watch history

| Model | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---|---|
| Most popular (baseline) | 0.443 | 0.043 | 0.471 |
| Content: TF-IDF | 0.252 | 0.028 | 0.268 |
| Content: sentence embeddings | 0.084 | 0.009 | 0.090 |
| Collaborative filtering: ALS (32 factors) | 0.572 | 0.059 | 0.583 |
| Hybrid (0.2 content + 0.8 ALS) | 0.600 | 0.063 | 0.614 |

## 3–5 favorite shows (how the app is used)

| Model | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---|---|
| Most popular (baseline) | 0.443 | 0.043 | 0.471 |
| Content: sentence embeddings | 0.090 | 0.010 | 0.103 |
| Collaborative filtering: ALS (32 factors) | 0.486 | 0.051 | 0.516 |
| Hybrid (0.1 content + 0.9 ALS) | 0.488 | 0.051 | 0.521 |
| Hybrid at full-history weight (0.2) | 0.480 | 0.050 | 0.511 |
