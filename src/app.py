from utils import db_connect
engine = db_connect()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Paso 1: Carga del conjunto de datos
url = "https://raw.githubusercontent.com/4GeeksAcademy/naive-bayes-project-tutorial/main/playstore_reviews.csv"
df = pd.read_csv(url)

# Paso 2: Estudio de variables y su contenido
def apply_preprocess(df):
    df = df.drop("package_name", axis=1)
    df["review"] = df["review"].str.strip().str.lower()
    return df

df = apply_preprocess(df)

# Paso 3: Dividir el conjunto de datos en train y test
X = df["review"]
y = df["polarity"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Paso 4: Transformar el texto en una matriz de recuento de palabras
vec_model = CountVectorizer(stop_words='english')
X_train_vec = vec_model.fit_transform(X_train).toarray()
X_test_vec = vec_model.transform(X_test).toarray()

# Paso 5: Construir y evaluar modelos Naive Bayes
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

gnb.fit(X_train_vec, y_train)
y_pred_gnb = gnb.predict(X_test_vec)
acc_gnb = accuracy_score(y_test, y_pred_gnb)

mnb.fit(X_train_vec, y_train)
y_pred_mnb = mnb.predict(X_test_vec)
acc_mnb = accuracy_score(y_test, y_pred_mnb)

bnb.fit(X_train_vec, y_train)
y_pred_bnb = bnb.predict(X_test_vec)
acc_bnb = accuracy_score(y_test, y_pred_bnb)

print(f'Exactitud de GaussianNB: {acc_gnb}')
print(f'Exactitud de MultinomialNB: {acc_mnb}')
print(f'Exactitud de BernoulliNB: {acc_bnb}')

# Paso 6: Optimizar el modelo MultinomialNB
hyperparams = {
    "alpha": np.linspace(0.01, 10.0, 200),
    "fit_prior": [True, False]
}

random_search = RandomizedSearchCV(mnb, hyperparams, n_iter=50, scoring="accuracy", cv=5, random_state=42)
random_search.fit(X_train_vec, y_train)

print(f"Mejores hiperparámetros: {random_search.best_params_}")

best_mnb = MultinomialNB(alpha=random_search.best_params_['alpha'], fit_prior=random_search.best_params_['fit_prior'])
best_mnb.fit(X_train_vec, y_train)
y_pred_best_mnb = best_mnb.predict(X_test_vec)
acc_best_mnb = accuracy_score(y_test, y_pred_best_mnb)

print(f'Exactitud del modelo optimizado MultinomialNB: {acc_best_mnb}')

# Paso 7: Evaluar otros modelos

# SVM
svm_model = SVC(random_state=42)
svm_model.fit(X_train_vec, y_train)
y_pred_svm = svm_model.predict(X_test_vec)
acc_svm = accuracy_score(y_test, y_pred_svm)
print(f'Exactitud de SVM: {acc_svm}')

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_vec, y_train)
y_pred_rf = rf_model.predict(X_test_vec)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f'Exactitud del modelo Random Forest: {acc_rf}')

# Gradient Boosting
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train_vec, y_train)
y_pred_gb = gb_model.predict(X_test_vec)
acc_gb = accuracy_score(y_test, y_pred_gb)
print(f'Exactitud de Gradient Boosting: {acc_gb}')

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_vec, y_train)
y_pred_lr = lr_model.predict(X_test_vec)
acc_lr = accuracy_score(y_test, y_pred_lr)
print(f'Exactitud de Logistic Regression: {acc_lr}')

# Paso 8: Guardar el mejor modelo
joblib.dump(lr_model, 'best_logistic_regression_model.pkl')
print("Modelo de Regresión Logística guardado exitosamente.")
