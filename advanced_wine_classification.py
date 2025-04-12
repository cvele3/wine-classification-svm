import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# 1. Učitavanje podataka
wine = load_wine()
X, y = wine.data, wine.target

# 2. Standardizacija
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Definiranje cross-validacije
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 4. Definicija hiperparametara za pretragu
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'kernel': ['rbf', 'linear', 'poly'],  # add 'poly' to see if it offers benefits
}

# 5. Grid Search s cross-validacijom
svm = SVC(random_state=42, verbose=False)
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, 
                           cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_scaled, y)

print("Najbolji parametri:", grid_search.best_params_)
print("Najbolji cross-validated accuracy: {:.4f}".format(grid_search.best_score_))

# Use the best estimator for further evaluation:
best_svm = grid_search.best_estimator_

# 6. Dobivanje predikcija za izračun konfuzijske matrice pomoću cross_val_predict
y_pred_cv = cross_val_predict(best_svm, X_scaled, y, cv=cv)
cm = confusion_matrix(y, y_pred_cv)
#print("Confusion Matrix:\n", cm)

# 7. Plot konfuzijske matrice
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Konfuzijska matrica")
plt.colorbar()
tick_marks = np.arange(len(wine.target_names))
plt.xticks(tick_marks, wine.target_names, rotation=45)
plt.yticks(tick_marks, wine.target_names)
plt.xlabel("Predviđena oznaka")
plt.ylabel("Stvarna oznaka")
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], "d"), horizontalalignment="center", 
                 color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.show()

# 8. PCA za vizualizaciju (2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 9. Za iscrtavanje granica odluke, treniramo SVM na cijelom PCA prostoru
# We use the best parameters here as well.
svm_pca = SVC(
    C=grid_search.best_params_['C'],
    gamma=grid_search.best_params_['gamma'],
    kernel=grid_search.best_params_['kernel'],
    random_state=42
)
# If kernel is poly, also set degree.
if svm_pca.kernel == 'poly':
    svm_pca.degree = grid_search.best_params_.get('degree', 3)

svm_pca.fit(X_pca, y)

# 10. Definiranje colormap i norm
cmap = plt.cm.coolwarm
norm = mcolors.Normalize(vmin=y.min(), vmax=y.max())

# 11. Kreiranje mreže točaka za granice odluke
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
Z = svm_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 12. Plot granica odluke
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap, norm=norm)

# 13. Scatter plot svih podataka u PCA prostoru
sc = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=cmap, norm=norm, 
                 marker="o", edgecolor="k", label="Podaci")
plt.xlabel("Prva PCA komponenta")
plt.ylabel("Druga PCA komponenta")
plt.title("PCA vizualizacija podataka sa granicama odluke")

# 14. Legenda za klase (mapiranje boja)
unique_classes = np.unique(y)
class_handles = []
for cls in unique_classes:
    color = cmap(norm(cls))
    class_handles.append(mpatches.Patch(color=color, label=wine.target_names[cls]))

plt.legend(handles=class_handles, loc="upper right", title="Klase")
plt.tight_layout()
plt.show()
