# data handling
import numpy as np

# ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process.kernels import RBF
from mlxtend.classifier import StackingCVClassifier
from sklearn import model_selection



class StackedClassifiers:

    def __init__(self, clfs):
        self.clfs = clfs

    @classmethod
    def clfs_fit(cls, sf_matrix_train: np.array, label_list_train: list, cross_val: int = 5):

        clf_1 = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", n_jobs=-1)
        clf_2 = SVC(gamma='auto', kernel="rbf", class_weight="balanced")
        clf_3 = GaussianProcessClassifier(kernel=1.0 * RBF(1.0), n_jobs=-1)

        estimators = [clf_1, clf_2, clf_3]
        lr = LogisticRegression()

        sclf = StackingCVClassifier(classifiers=estimators,
                                    meta_classifier=lr)

        print(f'{cross_val}-fold cross validation:\n')

        for clf, label in zip([clf_1, clf_2, clf_3],
                              ['Random Forest',
                               'SVC',
                               'Gaussian Process']):
            scores_f1 = model_selection.cross_val_score(clf, sf_matrix_train, label_list_train,
                                                        cv=cross_val, scoring='f1')
            scores_acc = model_selection.cross_val_score(clf, sf_matrix_train, label_list_train,
                                                         cv=cross_val, scoring='accuracy')
            print("Accuracy: %0.2f (+/- %0.2f) [%s]"
                  % (scores_acc.mean(), scores_acc.std(), label))
            print("F1: %0.2f (+/- %0.2f) [%s]"
                  % (scores_f1.mean(), scores_f1.std(), label))

        return cls(sclf.fit(sf_matrix_train, label_list_train))


    def clfs_proba(self, sf_matrix_pred):
        pred = self.clfs.predict_proba(sf_matrix_pred)
        return pred














