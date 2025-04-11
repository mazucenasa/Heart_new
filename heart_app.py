
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

# --- CARGA DE DATOS ---
@st.cache_data
def load_data():
    url = "https://github.com/mazucenasa/Heart_new/blob/main/heart.csv"
    df = pd.read_csv(url)
    return df

st.title("Predicción de Enfermedad Cardíaca")
df = load_data()
X = df.drop("target", axis=1)
y = df["target"]

# --- ENTRENAMIENTO DEL MODELO ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# --- ENTRADA MANUAL INDIVIDUAL ---
st.sidebar.header("Datos del paciente (modo manual)")

def user_input_features():
    age = st.sidebar.slider("Edad", 29, 77, 55)
    sex = st.sidebar.selectbox("Sexo", ["Mujer", "Hombre"])
    sex = 1 if sex == "Hombre" else 0
    cp = st.sidebar.selectbox("Tipo de dolor torácico (0-3)", [0, 1, 2, 3])
    trestbps = st.sidebar.slider("Presión arterial en reposo", 90, 200, 120)
    chol = st.sidebar.slider("Colesterol", 100, 600, 240)
    fbs = st.sidebar.selectbox("Glucosa en ayunas > 120 mg/dl", ["No", "Sí"])
    fbs = 1 if fbs == "Sí" else 0
    restecg = st.sidebar.selectbox("ECG en reposo", [0, 1, 2])
    thalach = st.sidebar.slider("FC máxima alcanzada", 70, 210, 150)
    exang = st.sidebar.selectbox("Angina inducida por ejercicio", ["No", "Sí"])
    exang = 1 if exang == "Sí" else 0
    oldpeak = st.sidebar.slider("Depresión ST", 0.0, 6.0, 1.0)
    slope = st.sidebar.selectbox("Pendiente ST", [0, 1, 2])
    ca = st.sidebar.selectbox("Vasos coloreados (0-4)", [0, 1, 2, 3, 4])
    thal = st.sidebar.selectbox("Thal (0-3)", [0, 1, 2, 3])

    return pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                          exang, oldpeak, slope, ca, thal]], columns=X.columns)

# --- PREDICCIÓN INDIVIDUAL ---
st.subheader("Predicción individual")
input_df = user_input_features()
prediction = model.predict(input_df)[0]
proba = model.predict_proba(input_df)[0][1]

st.write(f"**{'Enfermedad cardíaca detectada' if prediction == 1 else 'Sin enfermedad cardíaca'}**")
st.write(f"Probabilidad de enfermedad: **{proba:.2f}**")

# --- IMPORTANCIA DE VARIABLES ---
st.subheader("Importancia de características")
importances = model.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
fig, ax = plt.subplots()
sns.barplot(x=feat_imp.values, y=feat_imp.index, palette="viridis", ax=ax)
st.pyplot(fig)

# --- PCA ---
st.subheader("Visualización PCA (2D)")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
pca_df["target"] = y.values
fig2, ax2 = plt.subplots()
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="target", palette="Set1", ax=ax2)
st.pyplot(fig2)

# --- PREDICCIÓN POR LOTE DESDE CSV ---
st.subheader("Predicciones por lote (archivo CSV)")

uploaded_file = st.file_uploader("Sube un archivo CSV con datos de pacientes", type=["csv"])
if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)
    st.write("Datos cargados:")
    st.dataframe(batch_df.head())

    try:
        batch_pred = model.predict(batch_df)
        batch_proba = model.predict_proba(batch_df)[:, 1]
        batch_df["predicción"] = batch_pred
        batch_df["probabilidad"] = batch_proba
        batch_df["resultado"] = batch_df["predicción"].apply(lambda x: "Con enfermedad" if x == 1 else "Sin enfermedad")

        st.write("Resultados:")
        st.dataframe(batch_df[["predicción", "probabilidad", "resultado"]])

        # --- DESCARGA DEL RESULTADO EN EXCEL ---
        def convert_df_to_excel(df):
            output = BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                df.to_excel(writer, index=False, sheet_name="Resultados")
            return output.getvalue()

        excel_data = convert_df_to_excel(batch_df)

        st.download_button(
            label="Descargar resultados en Excel",
            data=excel_data,
            file_name="predicciones_cardio.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        st.error("Error en la predicción. Asegúrate de que el CSV tenga las mismas columnas que el dataset original.")
