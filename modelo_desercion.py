import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def entrenar_modelos(datos_modelo):
    columnas_features = [
        "carga_academica_score",
        "horas_estudio_score",
        "promedio_score",
        "apoyo_economico_score",
        "dificultad_economica_score",
        "trabajo_score",
        "estres_score",
        "motivacion_score",
        "pensamiento_abandono_score",
        "apoyo_institucional_score",
        "accesibilidad_score",
        "satisfaccion_score"
    ]

    X = datos_modelo[columnas_features]
    y = datos_modelo["nivel_riesgo_desercion"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    modelos = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced"
        ),
        "Árbol de Decisión": DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced"
        ),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=5))
        ]),
        "Regresión Logística": Pipeline([
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=1000, class_weight="balanced"))
        ]),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42
        )
    }

    resultados = []
    modelos_entrenados = {}

    # Entrenar y comparar todos los modelos
    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        pred = modelo.predict(X_test)

        accuracy = accuracy_score(y_test, pred)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test,
            pred,
            average="weighted",
            zero_division=0
        )

        resultados.append({
            "Modelo": nombre,
            "Accuracy": accuracy,
            "Precision ponderada": precision,
            "Recall ponderado": recall,
            "F1-score ponderado": f1
        })

        modelos_entrenados[nombre] = modelo

    metricas = pd.DataFrame(resultados)

    # Ordenar métricas para mostrar comparación
    metricas = metricas.sort_values(
        by="F1-score ponderado",
        ascending=False
    ).reset_index(drop=True)

    # Modelo principal seleccionado para la tesis
    modelo_principal_nombre = "Random Forest"
    modelo_principal = modelos_entrenados[modelo_principal_nombre]

    # Evaluar Random Forest en datos de prueba
    pred_test = modelo_principal.predict(X_test)

    etiquetas = ["Bajo", "Medio", "Alto"]

    matriz = confusion_matrix(
        y_test,
        pred_test,
        labels=etiquetas
    )

    matriz_df = pd.DataFrame(
        matriz,
        index=[f"Real {e}" for e in etiquetas],
        columns=[f"Predicho {e}" for e in etiquetas]
    )

    # Entrenar Random Forest con toda la base válida
    modelo_principal.fit(X, y)

    predicciones = datos_modelo.copy()
    predicciones["riesgo_predicho_modelo"] = modelo_principal.predict(X)

    probabilidades = modelo_principal.predict_proba(X)
    clases = list(modelo_principal.classes_)

    predicciones["prob_bajo"] = 0.0
    predicciones["prob_medio"] = 0.0
    predicciones["prob_alto"] = 0.0

    if "Bajo" in clases:
        predicciones["prob_bajo"] = probabilidades[:, clases.index("Bajo")]

    if "Medio" in clases:
        predicciones["prob_medio"] = probabilidades[:, clases.index("Medio")]

    if "Alto" in clases:
        predicciones["prob_alto"] = probabilidades[:, clases.index("Alto")]

    predicciones["recomendacion_modelo"] = predicciones["riesgo_predicho_modelo"].apply(
        generar_recomendacion_modelo
    )

    importancia_df = calcular_importancia_variables(
        modelo_principal,
        columnas_features
    )

    return metricas, modelo_principal_nombre, matriz_df, importancia_df, predicciones


def generar_recomendacion_modelo(riesgo):
    if riesgo == "Alto":
        return "Intervención temprana: tutoría, bienestar y seguimiento económico/académico."
    elif riesgo == "Medio":
        return "Monitoreo preventivo: revisar asistencia, estrés y apoyo institucional."
    else:
        return "Seguimiento general: mantener acompañamiento regular."


def calcular_importancia_variables(modelo, columnas_features):
    nombres_amigables = {
        "carga_academica_score": "Carga académica",
        "horas_estudio_score": "Horas de estudio",
        "promedio_score": "Promedio académico",
        "apoyo_economico_score": "Apoyo económico",
        "dificultad_economica_score": "Dificultad económica",
        "trabajo_score": "Situación laboral",
        "estres_score": "Estrés académico",
        "motivacion_score": "Motivación para continuar",
        "pensamiento_abandono_score": "Pensamiento de abandono",
        "apoyo_institucional_score": "Apoyo institucional",
        "accesibilidad_score": "Accesibilidad de servicios",
        "satisfaccion_score": "Satisfacción con acompañamiento"
    }

    importancias = modelo.feature_importances_

    importancia_df = pd.DataFrame({
        "Variable": [nombres_amigables[col] for col in columnas_features],
        "Campo técnico": columnas_features,
        "Importancia": importancias
    })

    importancia_df = importancia_df.sort_values(
        by="Importancia",
        ascending=False
    ).reset_index(drop=True)

    return importancia_df