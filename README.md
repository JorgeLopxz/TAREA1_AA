# TAREA1_AA - Prediccion de suscripcion a deposito

Proyecto de Aprendizaje Automatico para predecir si un cliente se suscribira a un deposito bancario (`yes` o `no`).

## 1. Estructura del proyecto

- `EDA.ipynb`: analisis, preprocesado, comparativa de modelos, HPO, seleccion final y evaluacion en test.
- `model_utils.py`: funciones compartidas para transformaciones de features (ejemplo: tratamiento de `pdays`).
- `modelo_final.ipynb`: carga del modelo entrenado y generacion de predicciones para competicion.
- `modelo_final.joblib`: pipeline final serializado.
- `mystreamlit.py`: app Streamlit para inferencia interactiva.
- `predicciones.csv`: salida final para el conjunto de competicion.
- `bank_13.pkl`: dataset de entrenamiento.
- `bank_competition_13.pkl`: dataset de competicion.

## 2. Entorno y dependencias

Este proyecto se ha ejecutado con Python en entorno virtual (`.venv`).

Dependencias principales:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib
- streamlit

Si no tienes entorno creado, puedes instalar rapido con:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Si prefieres instalar librerias manualmente, las dependencias principales son:

- pandas
- numpy
- scikit-learn
- scipy
- matplotlib
- seaborn
- joblib
- streamlit

## 3. Flujo recomendado de ejecucion

### Paso A: Entrenamiento y seleccion del modelo

1. Abrir `EDA.ipynb`.
2. Ejecutar celdas en orden.
3. Verificar salida de:
   - ranking de modelos por F1 en validacion
   - F1 en test
   - matriz de confusion
4. Confirmar exportacion de artefactos:
   - `modelo_final.joblib`
   - `predicciones.csv`

### Paso B: Uso del modelo final en notebook de inferencia

1. Abrir `modelo_final.ipynb`.
2. Ejecutar celda de carga/prediccion.
3. Comprobar que se regenera `predicciones.csv` sin errores.

### Paso C: Demo web con Streamlit

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run mystreamlit.py
```

La app crea un formulario dinamico segun el esquema de features y devuelve prediccion (y probabilidad si el modelo la expone).

## 4. Salidas de la practica

- Modelo final serializado: `modelo_final.joblib`
- Predicciones de competicion: `predicciones.csv`

Formato esperado de `predicciones.csv`:

- 1 columna: `deposit`
- 162 filas de prediccion
- valores permitidos: `yes`, `no`

## 5. Estado actual (resumen)

- Pipeline entrenado y evaluado en test.
- Notebook de inferencia operativo.
- App Streamlit funcional.
- Artefactos de entrega ya generados.

## 6. Notas de mantenimiento

- Evitar commitear `__pycache__/` o archivos temporales.
- Si se reentrena el modelo, regenerar y versionar `modelo_final.joblib` y `predicciones.csv` de forma consistente.
