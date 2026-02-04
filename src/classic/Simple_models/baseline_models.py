from collections import Counter

# Heuristic 1: compare the single most frequent fake vs real token.
def is_fake1_from_counts(counter: Counter, top_fake: str, top_real: str) -> str:
    return "fake" if counter[top_fake] > counter[top_real] else "real"

# Heuristic 2: compare overlap with aggregate fake/real token counters.
def is_fake2_from_counts(counter: Counter,
                         fake_train_counts: Counter,
                         real_train_counts: Counter) -> str:
    return "fake" if (counter & fake_train_counts).total() > (counter & real_train_counts).total() else "real"
