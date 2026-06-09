import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO

from procesamiento_datos import procesar_encuesta
from modelo_desercion import entrenar_modelos


# =========================
# CONFIGURACIÓN VISUAL
# =========================
COLOR_RIESGO = {
    "Bajo": "#2ECC71",    # verde
    "Medio": "#F1C40F",   # amarillo
    "Alto": "#E74C3C"     # rojo
}


# =========================
# FUNCIONES DE EXPORTACIÓN
# =========================
def convertir_excel(datos_modelo, metricas, matriz_confusion, importancia_variables, predicciones):
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        datos_modelo.to_excel(writer, index=False, sheet_name="Datos_Modelo")
        metricas.to_excel(writer, index=False, sheet_name="Metricas_Modelo")
        matriz_confusion.to_excel(writer, sheet_name="Matriz_Confusion")
        importancia_variables.to_excel(writer, index=False, sheet_name="Importancia_Variables")
        predicciones.to_excel(writer, index=False, sheet_name="Predicciones")

    output.seek(0)
    return output


def convertir_excel_filtrado(tabla_predicciones):
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        tabla_predicciones.to_excel(
            writer,
            index=False,
            sheet_name="Predicciones_Filtradas"
        )

    output.seek(0)
    return output


def cantidad_por_riesgo(conteo_riesgo, riesgo):
    fila = conteo_riesgo[conteo_riesgo["Nivel de riesgo"] == riesgo]
    if fila.empty:
        return 0
    return int(fila["Cantidad"].sum())


st.set_page_config(
    page_title="Dashboard de Deserción Estudiantil",
    layout="wide"
)

st.title("Dashboard BI Analytics para el Diagnóstico Temprano del Riesgo de Deserción Estudiantil")

st.markdown("""
Aplicación basada en **Machine Learning** que permite cargar un archivo Excel con encuestas,
procesar los datos, aplicar un modelo predictivo y visualizar el nivel de riesgo de deserción estudiantil.
""")

with st.expander("¿Cómo interpretar este dashboard?"):
    st.markdown("""
    **Riesgo bajo:** el estudiante presenta pocas señales asociadas a deserción.  
    **Riesgo medio:** el estudiante presenta señales que requieren monitoreo preventivo.  
    **Riesgo alto:** el estudiante requiere intervención temprana por parte de tutoría, bienestar o coordinación académica.  

    El sistema transforma las respuestas de la encuesta en puntajes numéricos y aplica un modelo de **Machine Learning**
    para clasificar el nivel de riesgo.

    🟢 **Bajo** = seguimiento general.  
    🟡 **Medio** = monitoreo preventivo.  
    🔴 **Alto** = intervención temprana.
    """)

archivo = st.file_uploader(
    "Cargar archivo Excel con resultados de la encuesta",
    type=["xlsx"]
)

if archivo is not None:
    df = pd.read_excel(archivo)

    try:
        datos_modelo, datos_excluidos, columnas_detectadas = procesar_encuesta(df)
        metricas, modelo_principal, matriz_confusion, importancia_variables, predicciones = entrenar_modelos(datos_modelo)

        fila_modelo = metricas[metricas["Modelo"] == modelo_principal].iloc[0]
        modelo_mayor_desempeno = metricas.iloc[0]["Modelo"]

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Resumen general",
            "Modelo Machine Learning",
            "Variables influyentes",
            "Predicciones por estudiante",
            "Exportación"
        ])

        with tab1:
            st.subheader("Resumen general del diagnóstico")
            st.caption(
                "Esta sección presenta una vista global de la base cargada y la distribución de estudiantes "
                "según el nivel de riesgo identificado por el modelo."
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Registros cargados", len(df))
            with col2:
                st.metric("Encuestas válidas", len(datos_modelo))
            with col3:
                st.metric("Encuestas incompletas", len(datos_excluidos))
            with col4:
                st.metric("Variables del modelo", 12)

            st.markdown("---")

            conteo_riesgo = predicciones["riesgo_predicho_modelo"].value_counts().reset_index()
            conteo_riesgo.columns = ["Nivel de riesgo", "Cantidad"]

            riesgo_bajo = cantidad_por_riesgo(conteo_riesgo, "Bajo")
            riesgo_medio = cantidad_por_riesgo(conteo_riesgo, "Medio")
            riesgo_alto = cantidad_por_riesgo(conteo_riesgo, "Alto")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Riesgo bajo", riesgo_bajo)
            with col2:
                st.metric("Riesgo medio", riesgo_medio)
            with col3:
                st.metric("Riesgo alto", riesgo_alto)

            col1, col2 = st.columns(2)

            with col1:
                fig_pie = px.pie(
                    conteo_riesgo,
                    names="Nivel de riesgo",
                    values="Cantidad",
                    hole=0.45,
                    title="Distribución general del riesgo de deserción",
                    color="Nivel de riesgo",
                    color_discrete_map=COLOR_RIESGO
                )

                fig_pie.update_traces(
                    textinfo="label+percent+value",
                    hovertemplate="<b>%{label}</b><br>Cantidad: %{value}<br>Porcentaje: %{percent}<extra></extra>"
                )

                fig_pie.update_layout(
                    legend_title_text="Nivel de riesgo",
                    title_x=0.05
                )

                st.caption(
                    "Este gráfico muestra la proporción de estudiantes clasificados en riesgo bajo, medio y alto."
                )

                st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                fig_bar = px.bar(
                    conteo_riesgo,
                    x="Nivel de riesgo",
                    y="Cantidad",
                    text="Cantidad",
                    title="Cantidad de estudiantes por nivel de riesgo",
                    color="Nivel de riesgo",
                    color_discrete_map=COLOR_RIESGO,
                    category_orders={"Nivel de riesgo": ["Bajo", "Medio", "Alto"]}
                )

                fig_bar.update_traces(
                    textposition="outside",
                    hovertemplate="<b>Riesgo: %{x}</b><br>Estudiantes: %{y}<extra></extra>"
                )

                fig_bar.update_layout(
                    xaxis_title="Nivel de riesgo",
                    yaxis_title="Cantidad de estudiantes",
                    legend_title_text="Clasificación",
                    title_x=0.05
                )

                st.caption(
                    "Este gráfico permite identificar cuántos estudiantes se encuentran en cada nivel de riesgo. "
                    "Los casos de riesgo alto deben priorizarse para intervención temprana."
                )

                st.plotly_chart(fig_bar, use_container_width=True)

            st.subheader("Interpretación del diagnóstico")

            total_estudiantes = len(predicciones)
            porcentaje_alto = (riesgo_alto / total_estudiantes) * 100 if total_estudiantes > 0 else 0

            if porcentaje_alto >= 20:
                st.error(
                    f"El {porcentaje_alto:.2f}% de estudiantes presenta riesgo alto. "
                    "Se recomienda priorizar acciones inmediatas de tutoría, bienestar y seguimiento académico."
                )
            elif porcentaje_alto >= 10:
                st.warning(
                    f"El {porcentaje_alto:.2f}% de estudiantes presenta riesgo alto. "
                    "Se recomienda realizar monitoreo preventivo y acompañamiento focalizado."
                )
            else:
                st.success(
                    f"El {porcentaje_alto:.2f}% de estudiantes presenta riesgo alto. "
                    "El nivel crítico es controlado, pero se recomienda mantener seguimiento preventivo."
                )

            st.info(
                f"Además, {riesgo_medio} estudiantes se encuentran en riesgo medio. "
                "Estos casos deben monitorearse para evitar que evolucionen a riesgo alto."
            )

            st.subheader("Vista previa de la base cargada")
            st.caption(
                "Se muestran los primeros registros del archivo cargado para verificar que la lectura del Excel fue correcta."
            )
            st.dataframe(df.head(10), use_container_width=True)

            st.subheader("Columnas detectadas para el modelo")
            st.caption(
                "Esta tabla muestra qué columna del Excel fue asociada con cada variable utilizada por el modelo predictivo."
            )
            columnas_df = pd.DataFrame(
                list(columnas_detectadas.items()),
                columns=["Variable del modelo", "Columna detectada en el Excel"]
            )
            st.dataframe(columnas_df, use_container_width=True)

        with tab2:
            st.subheader("Evaluación del modelo predictivo")
            st.caption(
                "En esta sección se comparan distintos algoritmos de Machine Learning. "
                "Aunque un modelo puede obtener mayor desempeño comparativo, Random Forest se utiliza como modelo principal "
                "porque permite interpretar la importancia de las variables."
            )

            col1, col2, col3, col4, col5, col6 = st.columns(6)

            with col1:
                st.metric("Modelo principal", modelo_principal)
            with col2:
                st.metric("Mayor desempeño", modelo_mayor_desempeno)
            with col3:
                st.metric("Accuracy", f"{fila_modelo['Accuracy'] * 100:.2f}%")
            with col4:
                st.metric("Precision", f"{fila_modelo['Precision ponderada'] * 100:.2f}%")
            with col5:
                st.metric("Recall", f"{fila_modelo['Recall ponderado'] * 100:.2f}%")
            with col6:
                st.metric("F1-score", f"{fila_modelo['F1-score ponderado'] * 100:.2f}%")

            st.markdown(f"""
            El sistema compara distintos algoritmos de Machine Learning. En esta ejecución, el modelo con mayor desempeño comparativo fue **{modelo_mayor_desempeno}**.  
            Sin embargo, se selecciona **{modelo_principal}** como modelo principal porque permite clasificar el riesgo de deserción e interpretar la importancia de las variables utilizadas en la predicción.
            """)

            metricas_grafico = metricas.melt(
                id_vars="Modelo",
                value_vars=[
                    "Accuracy",
                    "Precision ponderada",
                    "Recall ponderado",
                    "F1-score ponderado"
                ],
                var_name="Métrica",
                value_name="Valor"
            )

            fig_metricas = px.bar(
                metricas_grafico,
                x="Modelo",
                y="Valor",
                color="Métrica",
                barmode="group",
                text_auto=".2f",
                title="Comparación de desempeño entre algoritmos"
            )

            fig_metricas.update_layout(
                xaxis_title="Algoritmo evaluado",
                yaxis_title="Valor de la métrica",
                legend_title_text="Métrica de evaluación",
                yaxis_tickformat=".0%",
                title_x=0.05
            )

            fig_metricas.update_traces(
                hovertemplate="<b>%{x}</b><br>Métrica: %{fullData.name}<br>Valor: %{y:.2%}<extra></extra>"
            )

            st.plotly_chart(fig_metricas, use_container_width=True)

            st.info(
                "Interpretación: mientras más alto sea el valor de Accuracy, Precision, Recall y F1-score, "
                "mejor es el desempeño del modelo. Random Forest se mantiene como modelo principal por su capacidad interpretativa."
            )

            st.subheader("Métricas de validación por algoritmo")
            st.caption(
                "Esta tabla muestra los valores obtenidos por cada algoritmo. Sirve para comparar el rendimiento general de los modelos evaluados."
            )
            st.dataframe(metricas, use_container_width=True)

            st.subheader("Matriz de confusión del modelo principal")
            st.caption(
                "La matriz de confusión permite observar cuántos estudiantes fueron clasificados correctamente "
                "y en qué casos el modelo confundió un nivel de riesgo con otro."
            )
            st.dataframe(matriz_confusion, use_container_width=True)

            st.info(
                "Lectura: los valores ubicados en la diagonal principal representan aciertos del modelo. "
                "Los valores fuera de la diagonal representan clasificaciones incorrectas."
            )

        with tab3:
            st.subheader("Factores más influyentes en la deserción")
            st.caption(
                "Esta sección muestra qué variables tuvieron mayor peso en la clasificación del riesgo de deserción. "
                "Una mayor importancia indica que esa variable influyó más en la decisión del modelo."
            )

            variable_top = importancia_variables.iloc[0]["Variable"]
            importancia_top = importancia_variables.iloc[0]["Importancia"]

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Variable más influyente", variable_top)
            with col2:
                st.metric("Importancia máxima", f"{importancia_top * 100:.2f}%")

            fig_importancia = px.bar(
                importancia_variables,
                x="Importancia",
                y="Variable",
                orientation="h",
                text=importancia_variables["Importancia"].apply(lambda x: f"{x*100:.2f}%"),
                title="Ranking de variables más influyentes",
                labels={
                    "Importancia": "Nivel de importancia",
                    "Variable": "Variable analizada"
                }
            )

            fig_importancia.update_layout(
                yaxis=dict(autorange="reversed"),
                xaxis_tickformat=".0%",
                title_x=0.05
            )

            fig_importancia.update_traces(
                hovertemplate="<b>%{y}</b><br>Importancia: %{x:.2%}<extra></extra>"
            )

            st.plotly_chart(fig_importancia, use_container_width=True)

            st.info(
                f"La variable con mayor influencia fue **{variable_top}**, con una importancia de "
                f"**{importancia_top * 100:.2f}%** dentro del modelo Random Forest."
            )

            st.subheader("Detalle de importancia por variable")
            st.caption(
                "Esta tabla muestra el nombre de cada variable, su campo técnico y el peso asignado por el modelo."
            )
            st.dataframe(importancia_variables, use_container_width=True)

            st.markdown("""
            Esta información permite orientar acciones preventivas hacia los factores con mayor impacto,
            como tutorías académicas, apoyo socioeconómico, seguimiento emocional o mejora del acompañamiento institucional.
            """)

        with tab4:
            st.subheader("Seguimiento individual de estudiantes")
            st.caption(
                "Esta sección permite identificar a cada estudiante según su nivel de riesgo predicho, "
                "sus probabilidades asociadas y la recomendación sugerida por el sistema."
            )

            columnas_prediccion = [
                "id",
                "nivel_riesgo_desercion",
                "riesgo_predicho_modelo",
                "prob_alto",
                "prob_medio",
                "prob_bajo",
                "recomendacion_modelo"
            ]

            predicciones_filtradas = predicciones.copy()

            st.markdown("### Filtros de consulta")

            col1, col2, col3 = st.columns(3)

            with col1:
                filtro_riesgo = st.multiselect(
                    "Filtrar por riesgo predicho",
                    options=sorted(predicciones["riesgo_predicho_modelo"].unique()),
                    default=sorted(predicciones["riesgo_predicho_modelo"].unique())
                )

            with col2:
                filtro_recomendacion = st.multiselect(
                    "Filtrar por recomendación",
                    options=sorted(predicciones["recomendacion_modelo"].unique()),
                    default=sorted(predicciones["recomendacion_modelo"].unique())
                )

            with col3:
                buscar_id = st.text_input("Buscar por ID de estudiante")

            st.info(
                "Use los filtros para priorizar estudiantes en riesgo alto o para descargar una lista específica "
                "según el tipo de recomendación."
            )

            predicciones_filtradas = predicciones_filtradas[
                predicciones_filtradas["riesgo_predicho_modelo"].isin(filtro_riesgo)
            ]

            predicciones_filtradas = predicciones_filtradas[
                predicciones_filtradas["recomendacion_modelo"].isin(filtro_recomendacion)
            ]

            if buscar_id.strip() != "":
                predicciones_filtradas = predicciones_filtradas[
                    predicciones_filtradas["id"].astype(str).str.contains(buscar_id.strip())
                ]

            st.markdown("### Resumen filtrado")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Estudiantes filtrados", len(predicciones_filtradas))
            with col2:
                st.metric(
                    "Riesgo alto",
                    len(predicciones_filtradas[predicciones_filtradas["riesgo_predicho_modelo"] == "Alto"])
                )
            with col3:
                st.metric(
                    "Riesgo medio",
                    len(predicciones_filtradas[predicciones_filtradas["riesgo_predicho_modelo"] == "Medio"])
                )
            with col4:
                st.metric(
                    "Riesgo bajo",
                    len(predicciones_filtradas[predicciones_filtradas["riesgo_predicho_modelo"] == "Bajo"])
                )

            tabla_predicciones = predicciones_filtradas[columnas_prediccion].copy()

            tabla_predicciones["prob_alto"] = (tabla_predicciones["prob_alto"] * 100).round(2).astype(str) + "%"
            tabla_predicciones["prob_medio"] = (tabla_predicciones["prob_medio"] * 100).round(2).astype(str) + "%"
            tabla_predicciones["prob_bajo"] = (tabla_predicciones["prob_bajo"] * 100).round(2).astype(str) + "%"

            tabla_predicciones = tabla_predicciones.rename(columns={
                "id": "ID estudiante",
                "nivel_riesgo_desercion": "Riesgo semáforo",
                "riesgo_predicho_modelo": "Riesgo predicho",
                "prob_alto": "Prob. riesgo alto",
                "prob_medio": "Prob. riesgo medio",
                "prob_bajo": "Prob. riesgo bajo",
                "recomendacion_modelo": "Recomendación"
            })

            st.caption(
                "La tabla muestra la clasificación individual de cada estudiante. "
                "Las probabilidades indican qué tan probable es que pertenezca a cada nivel de riesgo."
            )
            st.dataframe(tabla_predicciones, use_container_width=True)

            archivo_filtrado = convertir_excel_filtrado(tabla_predicciones)

            st.download_button(
                label="Descargar predicciones filtradas",
                data=archivo_filtrado,
                file_name="predicciones_filtradas_desercion.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        with tab5:
            st.subheader("Exportación de resultados del análisis")

            st.markdown("""
            Descarga el archivo completo generado por el sistema. Este archivo contiene:

            - Datos codificados para el modelo.
            - Métricas de validación.
            - Matriz de confusión.
            - Importancia de variables.
            - Predicciones por estudiante.
            """)

            archivo_excel = convertir_excel(
                datos_modelo,
                metricas,
                matriz_confusion,
                importancia_variables,
                predicciones
            )

            st.download_button(
                label="Descargar resultados completos en Excel",
                data=archivo_excel,
                file_name="resultados_modelo_desercion.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            st.subheader("Base codificada utilizada por el modelo")
            st.caption(
                "Esta tabla muestra la base ya transformada en valores numéricos. "
                "Es la información que se utiliza como entrada para el entrenamiento y predicción del modelo."
            )
            st.dataframe(datos_modelo, use_container_width=True)

    except Exception as e:
        st.error("Ocurrió un error durante el procesamiento de la encuesta.")
        st.exception(e)

else:
    st.info("Carga un archivo Excel para iniciar el análisis.")
