import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, roc_curve, roc_auc_score

# acquire accuracy rate
def get_accuracy(clf, X, y, cv, scoring="accuracy"):
  score = cross_val_score(clf, X, y, cv=cv, scoring=scoring)
  return score

# acquire cross-val precision, recall for Stochastic GD
def get_precisions_recalls_thresholds_sgd(clf, X, y, cv):
  y_scores = cross_val_predict(clf, X, y, cv=cv, method="decision_function")
  precisions, recalls, thresholds = precision_recall_curve(y, y_scores)
  return precisions, recalls, thresholds

# acquire cross-val FPR, TPR 
def get_fpr_tpr_thresholds(clf, X, y, cv, method):
  if method == "decision_function": # sgd
    y_ = cross_val_predict(clf, X, y, cv=cv, method=method)
  elif method == "predict_proba": # random forest classifier 
    y_proba = cross_val_predict(clf, X, y, cv=cv, method=method)
    y_ = y_proba[:,1] # = proba of the positive class
  else:
    return "Invalid Method"

  fpr, tpr, thresholds = roc_curve(y, y_)

  return fpr, tpr, thresholds

# acquire discrete precision, recall for RF classifier with defined threshold:
def get_precision_recall_given_threshold_rf(clf, X, y, cv, threshold):
  y_proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")
  y_ = y_proba[:,1] # = proba of the positive class
  y_pred = (y_ >= threshold)
  precision_rf = precision_score(y, y_pred)
  recall_rf = recall_score(y, y_pred)
  return precision_rf, recall_rf


# plot precision-recall curve at 90% precision threshold
def plot_recall_precision(recalls, precisions):
  recall_90_precision = recalls[np.argmax(precisions >= 0.90)]

  plt.plot(recalls, precisions, "b")
  plt.xlabel("Recall")
  plt.ylabel("Precision")
  plt.axis([0, 1, 0, 1])    
  plt.plot([0, recall_90_precision], [0.9, 0.9], "r:")  
  plt.plot([recall_90_precision, recall_90_precision], [0.9, 0], "r:")
  plt.plot(recall_90_precision, 0.9, "ro")
  plt.grid(True) 
  plt.show()
  


