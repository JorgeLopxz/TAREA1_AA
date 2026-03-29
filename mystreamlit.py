import joblib
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Predicción de depósito", page_icon="🏦", layout="centered")
st.title("Predicción de suscripción a depósito bancario")


@st.cache_resource
def load_model():
	return joblib.load("modelo_final.joblib")


@st.cache_data
def load_reference_data():
	df = pd.read_pickle("bank_13.pkl")
	return df.drop(columns=["deposit"])


model = load_model()
X_ref = load_reference_data()

st.write("Introduce los datos del cliente para obtener una predicción del modelo final.")

input_data = {}
for col in X_ref.columns:
	if X_ref[col].dtype == "object":
		options = sorted(X_ref[col].dropna().unique().tolist())
		if options:
			input_data[col] = st.selectbox(col, options=options, index=0)
		else:
			input_data[col] = st.text_input(col, value="")
	else:
		col_min = float(X_ref[col].min())
		col_max = float(X_ref[col].max())
		col_median = float(X_ref[col].median())
		input_data[col] = st.number_input(
			col,
			min_value=col_min,
			max_value=col_max,
			value=col_median,
			step=1.0,
		)


if st.button("Predecir"):
	input_df = pd.DataFrame([input_data])
	pred = model.predict(input_df)[0]
	proba_text = ""

	if hasattr(model, "predict_proba"):
		classes = list(model.classes_)
		probas = model.predict_proba(input_df)[0]
		yes_index = classes.index("yes") if "yes" in classes else 1
		proba_text = f" (probabilidad de suscripción: {probas[yes_index]:.2%})"

	if pred == "yes":
		st.success(f"Predicción: el cliente SI se suscribirá{proba_text}.")
	else:
		st.warning(f"Predicción: el cliente NO se suscribirá{proba_text}.")
