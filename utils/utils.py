def get_class_weight(labeled_dataset, unlabeled_dataset):
    n_samples = len(labeled_dataset) + len(unlabeled_dataset)
    n_labeled = len(labeled_dataset)
    n_unlabeled = len(unlabeled_dataset)

    class_weight = [1 - n_labeled / n_samples, 1 - n_unlabeled / n_samples]