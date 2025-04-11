
# Heart Disease Prediction App

Esta aplicación predice la probabilidad de que un paciente tenga enfermedad cardíaca usando un modelo de aprendizaje automático entrenado con el dataset de Heart Disease.

## Funcionalidades

- Predicción interactiva para un solo paciente.
- Predicción por lotes desde archivo CSV.
- Visualización con PCA.
- Descarga de resultados en Excel y PDF.

## Cómo ejecutar localmente

1. Clona el repositorio:
   ```
   git clone https://github.com/tu-usuario/heart-app.git
   cd heart-app
   ```

2. Crea un entorno virtual y activa:
   ```
   python -m venv venv
   source venv/bin/activate  # o venv\Scripts\activate en Windows
   ```

3. Instala dependencias:
   ```
   pip install -r requirements.txt
   ```

4. Ejecuta la app:
   ```
   streamlit run heart_app.py
   ```

## Despliegue

Puedes subir este repositorio directamente a [Streamlit Cloud](https://streamlit.io/cloud) y publicar tu app en la web en minutos.
