import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Učitavanje i standardizacija podataka
wine = load_wine()
X, y = wine.data, wine.target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Definicija raspona test_size vrijednosti
test_sizes = np.arange(0.2, 0.76, 0.01)  # 0.2 do 0.6 s inkrementom od 0.02
accuracies = []

# Petlja kroz sve vrijednosti test_size
for ts in test_sizes:
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=ts, random_state=42, stratify=y
    )
    svm = SVC(C=1, gamma=0.01, degree=2, kernel='rbf', random_state=42)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"Test size: {ts:.2f}, Accuracy: {acc:.4f}")

# Pronalazak najbolje točnosti i odgovarajućeg test_size-a
best_index = np.argmax(accuracies)
best_ts = test_sizes[best_index]
best_acc = accuracies[best_index]
print(f"\nNajbolji test_size: {best_ts:.2f} s točnošću: {best_acc:.4f}")

# Plot rezultata
plt.plot(test_sizes, accuracies, marker='o')
plt.xlabel("Test size")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Test Size")
plt.show()
