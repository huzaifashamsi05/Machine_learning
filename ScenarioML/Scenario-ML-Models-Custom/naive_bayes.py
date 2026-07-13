# ─────────────────────────────────────────────
# NAIVE BAYES — FROM SCRATCH
# Text classification
# advisor_note se issue type predict karo
# ─────────────────────────────────────────────

import numpy as np

class NaiveBayesCustom:

    def __init__(self, alpha=1.0):
        # alpha = Laplace smoothing
        self.alpha      = alpha
        self.classes    = None
        self.log_priors = {}
        self.log_likelihoods = {}
        self.vocabulary = []

    # ─────────────────────────────────────────
    # FIT — TRAIN KARO
    # ─────────────────────────────────────────
    def fit(self, X_tokens, y):
        # X_tokens = list of token lists
        # y = class labels

        self.classes = list(np.unique(y))
        n_samples    = len(y)

        # ─────────────────────────────────────
        # STEP 1: Vocabulary banao
        # sirf training data se!
        # ─────────────────────────────────────
        vocab_set = set()
        for tokens in X_tokens:
            for token in tokens:
                vocab_set.add(token)
        self.vocabulary = list(vocab_set)
        vocab_size = len(self.vocabulary)
        word_index = {w: i for i, w in
                      enumerate(self.vocabulary)}

        # ─────────────────────────────────────
        # STEP 2: Prior calculate karo
        # prior = class kitni baar aya / total
        # ─────────────────────────────────────
        for cls in self.classes:
            cls_count = np.sum(y == cls)
            self.log_priors[cls] = np.log(
                cls_count / n_samples)

        # ─────────────────────────────────────
        # STEP 3: Likelihood calculate karo
        # har class mein har word kitni baar aya
        # ─────────────────────────────────────
        for cls in self.classes:
            # is class ki emails nikalo
            cls_indices = [i for i in range(n_samples)
                           if y[i] == cls]

            # word counts
            word_counts = np.zeros(vocab_size)
            for i in cls_indices:
                for token in X_tokens[i]:
                    if token in word_index:
                        word_counts[word_index[token]] += 1

            # Laplace smoothing ke saath likelihood
            total_words = word_counts.sum()
            log_like = np.log(
                (word_counts + self.alpha) /
                (total_words + self.alpha * vocab_size)
            )
            self.log_likelihoods[cls] = {
                w: log_like[i]
                for i, w in enumerate(self.vocabulary)
            }

        # top words per class
        self.top_words = {}
        for cls in self.classes:
            sorted_words = sorted(
                self.log_likelihoods[cls].items(),
                key=lambda x: x[1],
                reverse=True
            )
            self.top_words[cls] = sorted_words[:5]

    # ─────────────────────────────────────────
    # PREDICT — CLASS NIKALO
    # ─────────────────────────────────────────
    def predict(self, X_tokens):
        predictions = []
        scores_list = []

        for tokens in X_tokens:
            class_scores = {}

            for cls in self.classes:
                # prior se shuru karo
                score = self.log_priors[cls]

                # har word ka likelihood add karo
                for token in tokens:
                    if token in self.log_likelihoods[cls]:
                        score += self.log_likelihoods[cls][token]
                    # unseen word → smoothing handle karta hai

                class_scores[cls] = score

            # sabse zyada score wali class
            best_cls = max(class_scores,
                          key=class_scores.get)
            predictions.append(best_cls)
            scores_list.append(class_scores[best_cls])

        return np.array(predictions), np.array(scores_list)


print("naive_bayes.py ready!")