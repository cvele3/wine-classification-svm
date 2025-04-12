import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
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

# 3. Podjela na trening i test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 4. Treniranje SVM klasifikatora na originalnim značajkama
svm = SVC(C=0.01, gamma='scale', kernel='linear', random_state=42, verbose=True)
svm.fit(X_train, y_train)

# 5. Evaluacija
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Točnost modela:", accuracy)

# 6. Konfuzijska matrica
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Konfuzijska matrica")
plt.colorbar()
tick_marks = np.arange(len(wine.target_names))
plt.xticks(tick_marks, wine.target_names, rotation=45)
plt.yticks(tick_marks, wine.target_names)
plt.xlabel("Predviđena oznaka")
plt.ylabel("Stvarna oznaka")

# Prikaz vrijednosti unutar matrice
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], "d"), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.show()

# 7. PCA za vizualizaciju (2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Podjela PCA rezultata na trening i test (isti indeksni raspored)
X_pca_train, X_pca_test, _, _ = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)

# Treniranje novog SVM-a na PCA projekciji (za iscrtavanje granica odluke)
svm_pca = SVC(kernel="linear", random_state=42, C=0.01, gamma='scale')
svm_pca.fit(X_pca_train, y_train)

# 8. Definiramo colormap i norm, koje ćemo koristiti posvuda
cmap = plt.cm.coolwarm
# Pošto imamo tri klase: 0, 1, 2
norm = mcolors.Normalize(vmin=y.min(), vmax=y.max())

# 9. Kreiramo mrežu točaka za granice odluke
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
Z = svm_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 10. Plot granica odluke
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap, norm=norm)

# 11. Scatter plot trening i test podataka u PCA prostoru
sc_tr = plt.scatter(X_pca_train[:, 0], X_pca_train[:, 1], c=y_train, cmap=cmap, norm=norm, marker="o", edgecolor="k", label="Trening skup")
sc_ts = plt.scatter(X_pca_test[:, 0], X_pca_test[:, 1], c=y_test, cmap=cmap, norm=norm, marker="s", edgecolor="k", label="Test skup")

plt.xlabel("Prva PCA komponenta")
plt.ylabel("Druga PCA komponenta")
plt.title("PCA vizualizacija podataka sa granicama odluke")

# 12. Legenda za "Trening skup" i "Test skup" (razlikujemo po markeru)
train_handle = mlines.Line2D([], [], marker="o", color="w", markerfacecolor="gray", markeredgecolor="k", markersize=10, label="Trening skup")
test_handle = mlines.Line2D([], [], marker="s", color="w", markerfacecolor="gray", markeredgecolor="k", markersize=10, label="Test skup")

legend1 = plt.legend(handles=[train_handle, test_handle], loc="upper left", title="Skupovi")
plt.gca().add_artist(legend1)

# 13. Legenda za klase (mapiranje boja)
unique_classes = np.unique(y)
class_handles = []
for cls in unique_classes:
    # Boja koja se koristi za tu klasu
    color = cmap(norm(cls))
    class_handles.append(mpatches.Patch(color=color, label=wine.target_names[cls]))

legend2 = plt.legend(handles=class_handles, loc="upper right", title="Klase")
plt.gca().add_artist(legend2)

plt.tight_layout()
plt.show()
