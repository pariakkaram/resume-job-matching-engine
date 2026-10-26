"""Contrastive fine-tuning of the bi-encoder (System C) and the negative-sampling ablation.

* ``pairs``                training data, run configs, positive pairs and negative eligibility
* ``mine_hard_negatives``  random and BM25 negatives for training resumes (persisted)
* ``losses``               MultipleNegativesRankingLoss with in-batch false-negative masking (D24)
* ``validation``           weak-label NDCG@10 on validation resumes (checkpoint selection)
* ``train_biencoder``      SentenceTransformerTrainer run with checkpoints and config export
"""
