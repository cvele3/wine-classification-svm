from sklearn.model_selection import train_test_split, PredefinedSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
from sklearn.datasets import load_wine

# 1. Učitavanje podataka
wine = load_wine()
X, y = wine.data, wine.target

# 2. Standardizacija
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Podjela podataka (80% trening, 20% test)
X_train_full, X_test, y_train_full, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Ako želiš koristiti grid search na trening skupu s jednim fiksnim splitom, 
# dodatno podijeli trening skup na "pravi" trening i validacijski podskup (npr. 75%/25%)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full)
# Sada je X_train: 60%, X_val: 20%, a X_test: 20%

# 4. Kombiniraj trening i validacijski podskup radi grid searcha
X_grid = np.concatenate((X_train, X_val), axis=0)
y_grid = np.concatenate((y_train, y_val), axis=0)

# Kreiraj test_fold - u PredefinedSplit, oznaka -1 označava trening, a 0 označava validaciju
test_fold = [-1] * len(X_train) + [0] * len(X_val)
ps = PredefinedSplit(test_fold=test_fold)

# 5. Definiranje hiperparametara za pretragu
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'kernel': ['rbf', 'linear', 'poly'],
}

# 6. Grid Search s jednim fiksnim splitom (bez k-fold cross-validacije)
svm = SVC(random_state=42)
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=ps, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_grid, y_grid)

print("Najbolji parametri:", grid_search.best_params_)
print("Najbolja validacijska točnost: {:.4f}".format(grid_search.best_score_))

# 7. Evaluacija na test skupu (koji nije korišten tijekom grid searcha)
best_svm = grid_search.best_estimator_
test_accuracy = best_svm.score(X_test, y_test)
print("Točnost na test skupu: {:.4f}".format(test_accuracy))
