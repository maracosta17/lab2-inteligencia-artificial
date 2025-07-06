# Clasificación de churn de clientes (laboratorio adaptado a Thonny)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Cargar el archivo CSV
df = pd.read_csv("classificationdata.csv")

# 2. Mostrar las primeras filas
print("📄 Primeras filas del dataset:")
print(df.head())

# 3. Verificar si la columna objetivo es 'churn'
if "churn" not in df.columns:
    print("❌ La columna 'churn' no está en el archivo.")
else:
    print("\n📊 Distribución de clases (churn):")
    print(df["churn"].value_counts())
    df["churn"].value_counts().plot(kind="bar", title="Distribución de clases (Churn)")
    plt.xlabel("Clase")
    plt.ylabel("Cantidad")
    plt.tight_layout()
    plt.show()

    # 4. Preparar datos (eliminar columnas no numéricas o irrelevantes)
    X = df.drop(columns=["churn", "id_state", "area_code", "voice_mail_plan", "number_vmail_messages"])  # Puedes ajustar
    y = df["churn"]

    # Convertir variables categóricas (si quedan) a numéricas automáticamente
    X = pd.get_dummies(X)

    # 5. Separar en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print(f"\n✅ Conjunto de entrenamiento: {len(X_train)}")
    print(f"✅ Conjunto de prueba: {len(X_test)}")

    # 6. Modelo de árbol de decisión
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # 7. Predicción y evaluación
    y_pred = model.predict(X_test)
    print("\n🎯 Accuracy:", accuracy_score(y_test, y_pred))
    print("\n📄 Reporte de clasificación:\n", classification_report(y_test, y_pred))
