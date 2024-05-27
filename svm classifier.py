def SVM(X_train,y_train):
  import pickle
  from sklearn.svm import SVC
  time_svm_start = time.time()
  # Initialize and train the SVM
  svm_classifier = SVC(kernel='linear', random_state=42)
  svm_classifier.fit(X_train, y_train)
  time_svm_end = time.time()
  model_filepath = "classifier_svm.pkl"
  with open(model_filepath, 'wb+') as f:
    pickle.dump(svm_classifier, f)
  return svm_classifier