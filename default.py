# Uvoz potrebnih knjižnica
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Učitavanje podataka s UCI repozitorija
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
# Definiranje imena stupaca prema opisu skupa podataka
column_names = [
    "Class", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash",
    "Magnesium", "Total phenols", "Flavanoids", "Nonflavanoid phenols",
    "Proanthocyanins", "Color intensity", "Hue",
    "OD280/OD315 of diluted wines", "Proline"
]
data = pd.read_csv(url, header=None, names=column_names)

# Odvajanje značajki (X) i ciljne varijable (y)
X = data.drop("Class", axis=1)
y = data["Class"]

# a) Standardizacija numeričkih značajki
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# b) Podjela podataka na trening i test skup (80% trening, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.3, random_state=42)

# Razvijanje SVM klasifikatora s uključenim verbose modom za ispis procesa treniranja
svm = SVC(kernel="rbf", random_state=42, verbose=True)
svm.fit(X_train, y_train)

# Predikcija na test skupu
y_pred = svm.predict(X_test)

# Evaluacija klasifikatora korištenjem točnosti
accuracy = accuracy_score(y_test, y_pred)
print("Točnost klasifikatora:", accuracy)
