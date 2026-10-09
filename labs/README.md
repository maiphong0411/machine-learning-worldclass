# Labs

Every lab is a single, dependency-light Python file (NumPy only). Algorithms are
implemented **from scratch** so that nothing is magic, and each file ends with
`assert` statements that act as the auto-grader: the lab passes when the script exits 0.

```bash
python3 -m pip install numpy
for f in labs/*.py; do python3 "$f" > /dev/null && echo "PASS $f" || echo "FAIL $f"; done
```

| Lab | File | Module | You will build |
|---|---|---|---|
| 1 | [01_gradient_descent.py](01_gradient_descent.py) | M01–M02 | GD, SGD, momentum, Adam |
| 2 | [02_linear_logistic_regression.py](02_linear_logistic_regression.py) | M03 | Normal equation, ridge, logistic regression, gradient check |
| 3 | [03_metrics_from_scratch.py](03_metrics_from_scratch.py) | M04 | Confusion matrix, ROC/PR-AUC, calibration, NDCG, cost thresholds |
| 4 | [04_decision_tree_and_boosting.py](04_decision_tree_and_boosting.py) | M05 | CART, random forest, gradient boosting |
| 5 | [05_kmeans_pca.py](05_kmeans_pca.py) | M06 | k-means++, PCA via SVD, anomaly detection |
| 6 | [06_neural_network_backprop.py](06_neural_network_backprop.py) | M07 | MLP with hand-written backprop |
| 7 | [07_attention_and_embeddings.py](07_attention_and_embeddings.py) | M08 | Scaled dot-product and multi-head attention, cosine retrieval |
| 8 | [08_matrix_factorization.py](08_matrix_factorization.py) | M09 | Matrix factorization with SGD and ALS, recall@k |

**How to work on a lab:** read the module first, run the lab, then do the
"student exercises" listed at the bottom of each file. Exercises change the code;
the asserts must keep passing.
