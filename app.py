import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO

from procesamiento_datos import procesar_encuesta
from modelo_desercion import entrenar_modelos


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


st.set_page_config(
    page_title="Dashboard de Deserción Estudiantil",
    layout="wide"
)

st.title("Dashboard BI Analytics para el Diagnóstico Temprano del Riesgo de Deserción Estudiantil")

st.markdown("""
Aplicación basada en Machine Learning que permite cargar un archivo Excel con encuestas,
procesar los datos, aplicar un modelo predictivo y visualizar el nivel de riesgo de deserción estudiantil.
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

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Resumen general",
            "Modelo Machine Learning",
            "Variables influyentes",
            "Predicciones por estudiante",
            "Exportación"
        ])

        with tab1:
            st.subheader("Resumen general del análisis")

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

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Riesgo bajo",
                    int(conteo_riesgo[conteo_riesgo["Nivel de riesgo"] == "Bajo"]["Cantidad"].sum())
                )

            with col2:
                st.metric(
                    "Riesgo medio",
                    int(conteo_riesgo[conteo_riesgo["Nivel de riesgo"] == "Medio"]["Cantidad"].sum())
                )

            with col3:
                st.metric(
                    "Riesgo alto",
                    int(conteo_riesgo[conteo_riesgo["Nivel de riesgo"] == "Alto"]["Cantidad"].sum())
                )

            col1, col2 = st.columns(2)

            with col1:
                fig_pie = px.pie(
                    conteo_riesgo,
                    names="Nivel de riesgo",
                    values="Cantidad",
                    hole=0.45,
                    title="Distribución del riesgo de deserción estudiantil"
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                fig_bar = px.bar(
                    conteo_riesgo,
                    x="Nivel de riesgo",
                    y="Cantidad",
                    text="Cantidad",
                    title="Cantidad de estudiantes por nivel de riesgo"
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            st.subheader("Vista previa de la base cargada")
            st.dataframe(df.head(10), use_container_width=True)

            st.subheader("Columnas detectadas para el modelo")
            columnas_df = pd.DataFrame(
                list(columnas_detectadas.items()),
                columns=["Variable del modelo", "Columna detectada en el Excel"]
            )
            st.dataframe(columnas_df, use_container_width=True)

        with tab2:
            st.subheader("Evaluación del modelo predictivo")

            modelo_mayor_desempeno = metricas.iloc[0]["Modelo"]

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

            st.markdown("""
            El sistema compara distintos algoritmos de Machine Learning. En esta ejecución, el modelo con mayor desempeño comparativo fue **KNN**. 
            Sin embargo, se selecciona **Random Forest** como modelo principal debido a que permite clasificar el riesgo de deserción e interpretar la importancia de las variables utilizadas en la predicción.
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
                title="Comparación de algoritmos de Machine Learning"
            )

            st.plotly_chart(fig_metricas, use_container_width=True)

            st.subheader("Métricas de validación por algoritmo")
            st.dataframe(metricas, use_container_width=True)

            st.subheader("Matriz de confusión del modelo principal")
            st.dataframe(matriz_confusion, use_container_width=True)

        with tab3:
            st.subheader("Variables más influyentes en la predicción")

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
                text_auto=".2f",
                title="Importancia de variables en la predicción del riesgo de deserción"
            )

            fig_importancia.update_layout(
                yaxis=dict(autorange="reversed")
            )

            st.plotly_chart(fig_importancia, use_container_width=True)

            st.subheader("Detalle de importancia por variable")
            st.dataframe(importancia_variables, use_container_width=True)

            st.markdown("""
            Esta sección permite identificar los factores que tienen mayor peso dentro del modelo predictivo,
            facilitando la toma de decisiones para acciones preventivas.
            """)

        with tab4:
            st.subheader("Predicciones por estudiante")

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

            st.dataframe(tabla_predicciones, use_container_width=True)

            archivo_filtrado = convertir_excel_filtrado(tabla_predicciones)

            st.download_button(
                label="Descargar predicciones filtradas",
                data=archivo_filtrado,
                file_name="predicciones_filtradas_desercion.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        with tab5:
            st.subheader("Exportación de resultados")

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
            st.dataframe(datos_modelo, use_container_width=True)

    except Exception as e:
        st.error("Ocurrió un error durante el procesamiento de la encuesta.")
        st.exception(e)

else:
    st.info("Carga un archivo Excel para iniciar el análisis.")