import pickle
import numpy as np
from collections import Counter
from skmultilearn.problem_transform import LabelPowerset
from imblearn.under_sampling import RandomUnderSampler

def random_undersample_balanced(train_X, train_y, train_perc):
    lp = LabelPowerset()

    train_y = lp.transform(train_y)
    new_label_count = Counter(train_y)
    for k, v in new_label_count.items():
        new_label_count[k] = max(1, int(new_label_count[k] * train_perc)) # keep at least 1

    rus = RandomUnderSampler(sampling_strategy=new_label_count, random_state=42, replacement=False)

    # Applies the above stated multi-label (ML) to multi-class (MC) transformation.
    train_X, y_resampled = rus.fit_sample(train_X, train_y)
    train_y = lp.inverse_transform(y_resampled)
    train_y = train_y.todense()
    return train_X, train_y
