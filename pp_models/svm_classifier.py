import time
import pickle
from sklearn.svm import SVC

from config import SVM_MODEL_PATH

def SVM(X_train, y_train):
    time_svm_start = time.time()

    # Initialize and train the SVM
    svm_classifier = SVC(kernel='linear', random_state=42)
    svm_classifier.fit(X_train, y_train)
    
    time_svm_end = time.time()

    with open(SVM_MODEL_PATH, 'wb+') as f:
        pickle.dump(svm_classifier, f)
    
    print(f"SVM model saved to {SVM_MODEL_PATH}")
    print(f"Training time: {time_svm_end - time_svm_start:.2f} seconds")

    return svm_classifier
