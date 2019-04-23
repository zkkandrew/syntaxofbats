from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
	# setup marker generator and color map
	markers = ('s', 'x', 'o', '^', 'v')
	colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
	cmap = ListedColormap(colors[:len(np.unique(y))])
	# plot the decision surface
	x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
	Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
	Z = Z.reshape(xx1.shape)
	plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
	plt.xlim(xx1.min(), xx1.max())
	plt.ylim(xx2.min(), xx2.max())

	# plot all samples
	for idx, cl in enumerate(np.unique(y)):
		plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)

	# highlight test samples
	if test_idx:
		X_test, y_test = X[test_idx, :], y[test_idx]
		plt.scatter(X_test[:, 0], X_test[:, 1], c='', alpha=1.0, linewidths=1, marker='o', s=55, label='test set')


import pandas as pd
df = pd.read_csv('all_ml_8_rdy.csv',header=None)
df_param = df.loc[:, 3:]

# print(df_param.head(100))

# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder(categorical_features=[0])
# ohe.fit_transform(X)
df_rdy = pd.get_dummies(df_param)
# print(df_rdy.head())
df_rdy.columns = ['syl_number', 'syl_type_number', 'tran_type_num', 'linearity', 'consistency', 'seq_entropy', 'seq_po_product', 'seq_tran_product', 'syl_type_devide', 'uncertainty', 'context', '3_L', '3_L2', '3_LR', '3_LR2', '3_LR3','3_LR4', '3_N', '3_R', '4_F', '4_M']
# X = df_rdy.loc[:, [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, '3_1', '3_3', '3_4', '3_B00201', '3_B00212', '3_B00236', '3_B00238','3_B00244', '3_B00250', '3_C00345', '3_C00454', '3_C00474', '3_C00475','3_JH4053', '3_L', '3_L2', '3_LR', '3_LR2', '3_LR3','3_LR4', '3_N', '3_R', '4_F', '4_M']].values
X = df_rdy.loc[:, ['syl_number', 'syl_type_number', 'tran_type_num', 'linearity', 'consistency', 'seq_entropy', 'seq_po_product', 'seq_tran_product', 'syl_type_devide', 'uncertainty', '3_L', '3_L2', '3_LR', '3_LR2', '3_LR3','3_LR4', '3_N', '3_R', '4_F', '4_M']].values
y = df_rdy.loc[:, 'context'].values
# print(y[1])
# print(X[:, 1])
# print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# print(len(X_train), len(X_test))
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
# print(X_train_std[10])

# from sklearn.linear_model import Perceptron
# ppn = Perceptron(max_iter=40, eta0=0.1, random_state=0)
# ppn.fit(X_train_std, y_train)
# y_pred = ppn.predict(X_test_std)
# print('Misclassified samples: %d' % (y_test != y_pred).sum())
# # Misclassified samples: 69
from sklearn.metrics import accuracy_score
# print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
# # Accuracy: 0.95


# X_combined_std = np.vstack((X_train_std, X_test_std))
# y_combined = np.hstack((y_train, y_test))

# plot_decision_regions(X_combined_std, y_combined, classifier=ppn, test_idx=range(2982,4260))
# plt.xlabel('petal length [standardized]')
# plt.ylabel('petal width [standardized]')
# plt.legend(loc='upper left')
# plt.show()

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)
y_pred = lr.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())
# 25
print('Accuracy: %.4f' % accuracy_score(y_test, y_pred))
# 0.98
lr_predict = lr.predict_proba(X_test_std[0,:].reshape(1,-1))
print(lr_predict)
# print(lr.coef_)

# from sklearn.svm import SVC
# svm = SVC(kernel='linear', C=1.0, random_state=0)
# svm.fit(X_train_std, y_train)
# y_pred = svm.predict(X_test_std)
# print('Misclassified samples: %d' % (y_test != y_pred).sum())
# # 25
# print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
# # 0.98 

# from sklearn.svm import SVC
# svm = SVC(kernel='rbf', random_state=0, gamma=0.2, C=1.0)
# svm.fit(X_train_std, y_train)
# y_pred = svm.predict(X_test_std)
# print('Misclassified samples: %d' % (y_test != y_pred).sum())
# # 117
# print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
# # 0.91 

# from sklearn.tree import DecisionTreeClassifier
# tree = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=0)
# tree.fit(X_train, y_train)
# y_pred = tree.predict(X_test_std)
# print('Misclassified samples: %d' % (y_test != y_pred).sum())
# # 152
# print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
# # 0.88


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='entropy', n_estimators=10000, random_state=1, n_jobs=2)
forest.fit(X_train, y_train)
print(df_rdy.columns)
feature_labels = []
for col in df_rdy.columns:
	if col != 'context':
		feature_labels.append(col)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
	print(feature_labels[indices[f]], importances[indices[f]])
plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]), importances[indices], color='lightblue', align='center')
plt.xticks(range(X_train.shape[1]), ['(e)','(f)','(g)','(h)','(j)','(l)_LR','(k)_male','(k)_female','(c)','(b)','(d)','(l)_LR2','(i)','(l)_R','(a)','(l)_LR(l)','(l)_L','(l)_LR4','(l)_N','(l)_L2'], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()
# y_pred = forest.predict(X_test_std)
# print('Misclassified samples: %d' % (y_test != y_pred).sum())
# # 152
# print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
# # 0.88


# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
# knn.fit(X_train_std, y_train)
# y_pred = knn.predict(X_test_std)
# print('Misclassified samples: %d' % (y_test != y_pred).sum())
# # 152
# print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
# # 0.88

# from sklearn.linear_model import LogisticRegression
# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# lr = LogisticRegression()
# X_train_pca = pca.fit_transform(X_train_std)
# X_test_pca = pca.transform(X_test_std)
# lr.fit(X_train_pca, y_train)
# y_pred = lr.predict(X_test_pca)
# print('Misclassified samples: %d' % (y_test != y_pred).sum())
# # 283
# print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
# 0.78
# plot_decision_regions(X_train_pca, y_train, classifier=lr)
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.legend(loc='lower left')
# plt.show()

# plot_decision_regions(X_test_pca, y_test, classifier=lr)
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.legend(loc='lower left')
# plt.show()
# pca = PCA(n_components=None)
# X_train_pca = pca.fit_transform(X_train_std)
# print(pca.explained_variance_ratio_)
# [  2.04252551e-01   8.84743746e-02   4.12523410e-02   3.93525282e-02
#    3.40885389e-02   3.21817745e-02   3.17046606e-02   3.15768148e-02
#    3.08314298e-02   3.07393570e-02   3.06701600e-02   3.05659329e-02
#    3.04656164e-02   3.03839563e-02   3.03705201e-02   3.03334460e-02
#    3.02969283e-02   3.02342168e-02   3.01462078e-02   2.98317016e-02
#    2.96744314e-02   2.86981213e-02   2.74773126e-02   2.37888435e-02
#    8.75355518e-03   8.40276735e-03   1.75796087e-03   1.46753264e-03
#    1.03101952e-03   6.61425791e-04   3.35684661e-04   1.98288374e-04
#    4.02817131e-32   1.46817380e-32]


# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.linear_model import LogisticRegression
# lda = LinearDiscriminantAnalysis(n_components=2)
# X_train_lda = lda.fit_transform(X_train_std, y_train)
# lr = LogisticRegression()
# lr = lr.fit(X_train_lda, y_train)

# X_test_lda = lda.transform(X_test_std)
# y_pred = lr.predict(X_test_lda)
# print('Misclassified samples: %d' % (y_test != y_pred).sum())
# # 52
# print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
# # 0.96


# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import Pipeline
# pipe_lr = Pipeline([('scl', StandardScaler()), ('clf', LogisticRegression(random_state=1))])


# from sklearn.model_selection import StratifiedKFold
# kfold = StratifiedKFold(n_splits=10, random_state=1)
# n_splits = kfold.split(X_train, y_train)

# from sklearn.model_selection import cross_val_score
# scores = cross_val_score(estimator=lr, X=X_train_std, y=y_train, cv=10, n_jobs=1)
# print('CV accuracy scores: %s' % scores)
# print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

# from sklearn.model_selection import learning_curve
# pipe_lr = Pipeline([ ('scl', StandardScaler()), ('clf', LogisticRegression(penalty='l2', random_state=0)) ])
# train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr, X=X_train, y=y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=10, n_jobs=1)
# train_mean = np.mean(train_scores, axis=1)
# train_std = np.std(train_scores, axis=1)
# test_mean = np.mean(test_scores, axis=1)
# test_std = np.std(test_scores, axis=1)

# plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
# plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
# plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
# plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
# plt.grid()
# plt.xlabel('Number of training samples')
# plt.ylabel('Accuracy')
# plt.legend(loc='lower right')
# plt.ylim([0.8, 1.0])
# plt.show()

# Addressing overftting and underftting with validation curves
# from sklearn.model_selection import validation_curve
# param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 500.0, 1000.0]
# train_scores, test_scores = validation_curve(estimator=lr, X=X_train_std, y=y_train, param_name='C', param_range=param_range, cv=10)
# train_mean = np.mean(train_scores, axis=1)
# train_std = np.std(train_scores, axis=1)
# test_mean = np.mean(test_scores, axis=1)
# test_std = np.std(test_scores, axis=1)

# plt.plot(param_range, train_mean,
# 	color='blue', marker='o',
# 	markersize=7,
# 	label='training accuracy')
# plt.fill_between(param_range, train_mean + train_std,
# 	train_mean - train_std, alpha=0.1,
# 	color='blue')
# plt.plot(param_range, test_mean,
# 	color='green', linestyle='--',
# 	marker='s', markersize=7,
# 	label='validation accuracy')
# plt.fill_between(param_range,
# 	test_mean + test_std,
# 	test_mean - test_std,
# 	alpha=0.15, color='green')
# plt.grid()
# plt.xscale('log')
# plt.legend(loc='lower right')
# plt.xlabel('Parameter C')
# plt.ylabel('Accuracy')
# plt.ylim([0.75, 1.0])
# plt.show()

# from sklearn.metrics import roc_curve, auc
# from scipy import interp
# from sklearn.pipeline import Pipeline
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import StratifiedKFold

# pipe_lr = Pipeline([('scl', StandardScaler()), ('clf', LogisticRegression(random_state=0, C=1000.0))])
# # cv = StratifiedKFold(n_splits=10, random_state=1)
# svm = Pipeline([('scl', StandardScaler()), ('clf', SVC(kernel='linear', C=1.0, random_state=0, probability=True))])
# tree = Pipeline([('clf', DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=0))])

# fig = plt.figure(figsize=(7, 5))
# mean_tpr = 0.0
# mean_fpr = np.linspace(0, 1, 100)

# probas = pipe_lr.fit(X_train, y_train).predict_proba(X_train)
# fpr, tpr, thresholds = roc_curve(y_train, probas[:, 1], pos_label=1)
# roc_auc = auc(fpr, tpr)
# plt.plot(fpr, tpr, lw=1, label='Logistic regression (area = %0.2f)' % (roc_auc))

# probas = svm.fit(X_train, y_train).predict_proba(X_train)
# fpr, tpr, thresholds = roc_curve(y_train, probas[:, 1], pos_label=1)
# roc_auc = auc(fpr, tpr)
# plt.plot(fpr, tpr, lw=1, label='SVM (area = %0.2f)' % (roc_auc))

# probas = tree.fit(X_train, y_train).predict_proba(X_train)
# fpr, tpr, thresholds = roc_curve(y_train, probas[:, 1], pos_label=1)
# roc_auc = auc(fpr, tpr)
# plt.plot(fpr, tpr, lw=1, label='Decision tree (area = %0.2f)' % (roc_auc))

# plt.plot([0, 1],
# 	[0, 1],
# 	linestyle='--',
# 	color=(0.6, 0.6, 0.6),
# 	label='random guessing')

# plt.plot([0, 0, 1],
# 	[0, 1, 1],
# 	lw=2,
# 	linestyle=':',
# 	color='black',
# 	label='perfect performance')
# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.xlabel('false positive rate')
# plt.ylabel('true positive rate')
# plt.title('Receiver Operator Characteristic')
# plt.legend(loc="lower right")
# plt.show()












