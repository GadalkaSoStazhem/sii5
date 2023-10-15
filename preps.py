class LabelEncoder:
    def __init__(self):
        self.label_map = {}
        self.inverse_label_map = {}

    def fit(self, labels):
        unique_labels = set(labels)
        for i, label in enumerate(unique_labels):
            self.label_map[label] = i
            self.inverse_label_map[i] = label

    def transform(self, labels):
        return [self.label_map[label] for label in labels]

    def inverse_transform(self, encoded_labels):
        return [self.inverse_label_map[label] for label in encoded_labels]



