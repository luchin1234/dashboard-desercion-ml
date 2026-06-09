import pandas as pd
import unicodedata


def limpiar_texto(texto):
    if pd.isna(texto):
        return ""

    texto = str(texto).strip().lower()
    texto = texto.replace("\xa0", " ")
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join([c for c in texto if not unicodedata.combining(c)])
    texto = " ".join(texto.split())
    return texto


def obtener_columnas_preguntas(df):
    columnas_excluir = [
        "id",
        "hora de inicio",
        "hora de finalizacion",
        "hora de finalización",
        "correo electronico",
        "correo electrónico",
        "nombre",
        "hora de la ultima modificacion",
        "hora de la última modificación"
    ]

    columnas_preguntas = []

    for col in df.columns:
        col_limpia = limpiar_texto(col)
        if col_limpia not in columnas_excluir:
            columnas_preguntas.append(col)

    return columnas_preguntas


def escala_riesgo_nada_mucho(valor):
    texto = limpiar_texto(valor)

    if texto.startswith("nada"):
        return 1
    if texto.startswith("poco"):
        return 2
    if texto == "regular" or texto == "moderada":
        return 3
    if texto.startswith("bastante"):
        return 4
    if texto.startswith("mucho") or texto.startswith("muy"):
        return 5

    return None


def escala_invertida_nada_mucho(valor):
    texto = limpiar_texto(valor)

    if texto.startswith("mucho") or texto.startswith("muy"):
        return 1
    if texto.startswith("bastante"):
        return 2
    if texto == "regular" or texto == "moderada":
        return 3
    if texto.startswith("poco"):
        return 4
    if texto.startswith("nada"):
        return 5

    return None


def puntaje_carga_academica(valor):
    return escala_riesgo_nada_mucho(valor)


def puntaje_horas_estudio(valor):
    texto = limpiar_texto(valor)

    if "mas de 20" in texto:
        return 1
    if "16" in texto and "20" in texto:
        return 2
    if "11" in texto and "15" in texto:
        return 3
    if "5" in texto and "10" in texto:
        return 4
    if "menos de 5" in texto:
        return 5

    return None


def puntaje_promedio(valor):
    texto = limpiar_texto(valor)

    if "< 11" in texto or "menor" in texto:
        return 5
    if "11" in texto and "13" in texto:
        return 4
    if "14" in texto and "16" in texto:
        return 2
    if "> 16" in texto or "mayor" in texto:
        return 1

    return None


def puntaje_apoyo_economico(valor):
    texto = limpiar_texto(valor)

    if "totalmente" in texto:
        return 1
    if "parcialmente" in texto:
        return 3
    if texto == "no":
        return 5

    return None


def puntaje_dificultad_economica(valor):
    return escala_riesgo_nada_mucho(valor)


def puntaje_trabajo(valor):
    texto = limpiar_texto(valor)

    if "no trabajo" in texto:
        return 1
    if "medio tiempo" in texto:
        return 3
    if "tiempo completo" in texto:
        return 5

    return None


def puntaje_estres(valor):
    return escala_riesgo_nada_mucho(valor)


def puntaje_motivacion(valor):
    return escala_invertida_nada_mucho(valor)


def puntaje_pensamiento_abandono(valor):
    texto = limpiar_texto(valor)

    if texto == "nunca":
        return 1
    if "alguna vez" in texto:
        return 3
    if "varias veces" in texto:
        return 4
    if "frecuentemente" in texto:
        return 5

    return None


def puntaje_apoyo_institucional(valor):
    texto = limpiar_texto(valor)

    if texto == "no":
        return 5
    if "parcial" in texto:
        return 3
    if "suficiente" in texto or texto == "si":
        return 1

    return None


def puntaje_accesibilidad(valor):
    texto = limpiar_texto(valor)

    if "muy accesibles" in texto:
        return 1
    if "bastante accesibles" in texto:
        return 2
    if texto == "regular":
        return 3
    if "poco accesibles" in texto:
        return 4
    if "nada accesibles" in texto:
        return 5

    return None


def puntaje_satisfaccion(valor):
    texto = limpiar_texto(valor)

    if "muy satisfecho" in texto:
        return 1
    if "bastante satisfecho" in texto:
        return 2
    if texto == "regular":
        return 3
    if "poco satisfecho" in texto:
        return 4
    if "nada satisfecho" in texto:
        return 5

    return None


def clasificar_riesgo(porcentaje):
    if porcentaje < 0.40:
        return "Bajo"
    elif porcentaje < 0.65:
        return "Medio"
    else:
        return "Alto"


def generar_recomendacion(riesgo):
    if riesgo == "Alto":
        return "Intervención temprana: tutoría, bienestar y seguimiento económico/académico."
    elif riesgo == "Medio":
        return "Monitoreo preventivo: revisar asistencia, estrés y apoyo institucional."
    else:
        return "Seguimiento general: mantener acompañamiento regular."


def procesar_encuesta(df_original):
    df = df_original.copy()

    columnas_preguntas = obtener_columnas_preguntas(df)

    if len(columnas_preguntas) < 12:
        raise ValueError(
            f"No se encontraron suficientes preguntas. Se detectaron {len(columnas_preguntas)} columnas de preguntas."
        )

    columnas = {
        "carga_academica": columnas_preguntas[0],
        "horas_estudio": columnas_preguntas[1],
        "promedio": columnas_preguntas[2],
        "apoyo_economico": columnas_preguntas[3],
        "dificultad_economica": columnas_preguntas[4],
        "trabajo": columnas_preguntas[5],
        "estres": columnas_preguntas[6],
        "motivacion": columnas_preguntas[7],
        "pensamiento_abandono": columnas_preguntas[8],
        "apoyo_institucional": columnas_preguntas[9],
        "accesibilidad": columnas_preguntas[10],
        "satisfaccion": columnas_preguntas[11],
    }

    datos = pd.DataFrame()

    if "ID" in df.columns:
        datos["id"] = df["ID"]
    elif "id" in df.columns:
        datos["id"] = df["id"]
    else:
        datos["id"] = range(1, len(df) + 1)

    datos["carga_academica_score"] = df[columnas["carga_academica"]].apply(puntaje_carga_academica)
    datos["horas_estudio_score"] = df[columnas["horas_estudio"]].apply(puntaje_horas_estudio)
    datos["promedio_score"] = df[columnas["promedio"]].apply(puntaje_promedio)
    datos["apoyo_economico_score"] = df[columnas["apoyo_economico"]].apply(puntaje_apoyo_economico)
    datos["dificultad_economica_score"] = df[columnas["dificultad_economica"]].apply(puntaje_dificultad_economica)
    datos["trabajo_score"] = df[columnas["trabajo"]].apply(puntaje_trabajo)
    datos["estres_score"] = df[columnas["estres"]].apply(puntaje_estres)
    datos["motivacion_score"] = df[columnas["motivacion"]].apply(puntaje_motivacion)
    datos["pensamiento_abandono_score"] = df[columnas["pensamiento_abandono"]].apply(puntaje_pensamiento_abandono)
    datos["apoyo_institucional_score"] = df[columnas["apoyo_institucional"]].apply(puntaje_apoyo_institucional)
    datos["accesibilidad_score"] = df[columnas["accesibilidad"]].apply(puntaje_accesibilidad)
    datos["satisfaccion_score"] = df[columnas["satisfaccion"]].apply(puntaje_satisfaccion)

    columnas_score = [col for col in datos.columns if col.endswith("_score")]

    datos["encuesta_completa"] = datos[columnas_score].notna().all(axis=1)

    datos_validos = datos[datos["encuesta_completa"] == True].copy()
    datos_excluidos = datos[datos["encuesta_completa"] == False].copy()

    datos_validos["puntaje_academico"] = datos_validos[
        ["carga_academica_score", "horas_estudio_score", "promedio_score"]
    ].mean(axis=1)

    datos_validos["puntaje_socioeconomico"] = datos_validos[
        ["apoyo_economico_score", "dificultad_economica_score", "trabajo_score"]
    ].mean(axis=1)

    datos_validos["puntaje_emocional_motivacional"] = datos_validos[
        ["estres_score", "motivacion_score", "pensamiento_abandono_score"]
    ].mean(axis=1)

    datos_validos["puntaje_institucional"] = datos_validos[
        ["apoyo_institucional_score", "accesibilidad_score", "satisfaccion_score"]
    ].mean(axis=1)

    datos_validos["puntaje_total_promedio"] = datos_validos[columnas_score].mean(axis=1)

    datos_validos["porcentaje_riesgo"] = datos_validos["puntaje_total_promedio"] / 5

    datos_validos["nivel_riesgo_desercion"] = datos_validos["porcentaje_riesgo"].apply(clasificar_riesgo)

    datos_validos["recomendacion"] = datos_validos["nivel_riesgo_desercion"].apply(generar_recomendacion)

    return datos_validos, datos_excluidos, columnas