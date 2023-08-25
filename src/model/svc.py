import numpy as np

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV


# load data

FEATURES_PATH = "../feature"

aco, label = np.load(f"{FEATURES_PATH}/acoustic_0.npy"), np.load(F"{FEATURES_PATH}/label_0.npy").reshape(-1, 1)

for i in range(1, 6):
    # load chunk
    aco_i, label_i = np.load(f"{FEATURES_PATH}/acoustic_{i}.npy"), np.load(F"{FEATURES_PATH}/label_{i}.npy")
    # stack
    aco, label = np.vstack((aco, aco_i)), np.vstack((label, label_i.reshape(-1, 1)))


print(f"acoustic: {aco.shape}", f"labels: {label.shape}")


enc = OneHotEncoder()
label_one = enc.fit_transform(label.reshape(-1, 1))

x_train, x_test, y_train, y_test = train_test_split(aco, label.ravel(), random_state=5, test_size=0.2, shuffle=True)


pipe = Pipeline([('pca', PCA()),
                 ('scaler', StandardScaler()),
                 ('clf', SVC())])
params = {
    'pca__n_components' : [15, 30, 45, 60],
    'clf__kernel' : ['linear', 'rbf'],
    'clf__gamma' : ['scale', 'auto']
}

gs = GridSearchCV(estimator=pipe, param_grid=params, scoring='accuracy', verbose=0)

gs.fit(x_train, y_train)

print(f"best params: {gs.best_params_}")


from sklearn.metrics import accuracy_score

y_pred = gs.predict(x_test)
acc = accuracy_score(y_test, y_pred)

print(f"test accuracy: {acc}")
