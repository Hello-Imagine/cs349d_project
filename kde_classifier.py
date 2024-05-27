class KDEClassifier:
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth
        self.kde_class0 = None
        self.kde_class1 = None

    def fit(self, X, y):
        X_class0 = X[y == 0]
        X_class1 = X[y == 1]

        # Train a KDE model for each class
        self.kde_class0 = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth).fit(X_class0)
        self.kde_class1 = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth).fit(X_class1)

    def predict(self, X):
        log_density_class0 = self.kde_class0.score_samples(X)
        log_density_class1 = self.kde_class1.score_samples(X)
        return np.where(log_density_class0 > log_density_class1, 0, 1)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)

# Example usage
def kde(X_train, y_train, model_filepath='kde_model.pkl'):
    classifier = KDEClassifier(bandwidth=1.0)
    classifier.fit(X_train, y_train)
    classifier.save(model_filepath)
    print(f"KDE model saved to {model_filepath}")
    return classifier