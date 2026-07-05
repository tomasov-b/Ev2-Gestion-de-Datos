from __future__ import annotations

"""Entrena y evalua un modelo de regresion logistica para incumplimiento de prestamos.

El script puede leer desde el CSV limpio local o desde Supabase, genera artefactos
para BI, registra tiempos y memoria, y guarda metricas en JSON.
"""

import argparse
import base64
import json
import logging
import os
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
	accuracy_score,
	classification_report,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
	roc_auc_score,
	roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
	from imblearn.over_sampling import SMOTE
	from imblearn.pipeline import Pipeline as ImbPipeline
except Exception:  # pragma: no cover - optional dependency
	ImbPipeline = None
	SMOTE = None

try:
	from supabase import create_client
except Exception:  # pragma: no cover - optional when running local only
	create_client = None


EXPECTED_COLUMNS = [
	"person_age",
	"person_gender",
	"person_education",
	"person_income",
	"person_emp_exp",
	"person_home_ownership",
	"loan_amnt",
	"loan_intent",
	"loan_int_rate",
	"loan_percent_income",
	"cb_person_cred_hist_length",
	"credit_score",
	"previous_loan_defaults_on_file",
	"loan_status",
]

NUMERIC_COLUMNS = [
	"person_age",
	"person_income",
	"person_emp_exp",
	"loan_amnt",
	"loan_int_rate",
	"loan_percent_income",
	"cb_person_cred_hist_length",
	"credit_score",
]

CATEGORICAL_COLUMNS = [
	"person_gender",
	"person_education",
	"person_home_ownership",
	"loan_intent",
	"previous_loan_defaults_on_file",
]

TARGET_COLUMN = "loan_status"


@dataclass
class ResourceSnapshot:
	seconds: float
	rss_mb: float
	cpu_percent: float


def configurar_logger(log_file: Path) -> logging.Logger:
	log_file.parent.mkdir(parents=True, exist_ok=True)
	logger = logging.getLogger("loan_model")
	logger.setLevel(logging.INFO)
	logger.handlers.clear()
	logger.propagate = False
	formato = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

	handler_console = logging.StreamHandler()
	handler_console.setFormatter(formato)
	logger.addHandler(handler_console)

	handler_file = logging.FileHandler(log_file, mode="w", encoding="utf-8")
	handler_file.setFormatter(formato)
	logger.addHandler(handler_file)
	return logger


def captura_recursos() -> ResourceSnapshot:
	process = psutil.Process(os.getpid())
	rss_mb = process.memory_info().rss / (1024 ** 2)
	cpu_percent = process.cpu_percent(interval=0.1)
	return ResourceSnapshot(seconds=time.perf_counter(), rss_mb=rss_mb, cpu_percent=cpu_percent)


def registrar_recursos(logger: logging.Logger, etiqueta: str, inicio: ResourceSnapshot) -> ResourceSnapshot:
	actual = captura_recursos()
	elapsed = actual.seconds - inicio.seconds
	logger.info("%s | tiempo=%.2fs | rss=%.1f MB | cpu=%.1f%%", etiqueta, elapsed, actual.rss_mb, actual.cpu_percent)
	return actual


def hash_value(valor: Any) -> str | None:
	texto = str(valor).strip()
	if not texto:
		return None
	return sha256(texto.encode("utf-8")).hexdigest()


def pseudonimizar_columnas(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
	for columna in columnas:
		if columna in df.columns:
			df[columna] = df[columna].map(hash_value)
	return df


def normalizar_booleano(valor: Any) -> int | None:
	if pd.isna(valor):
		return None
	texto = str(valor).strip().lower()
	if texto in {"1", "true", "yes", "y", "t"}:
		return 1
	if texto in {"0", "false", "no", "n", "f"}:
		return 0
	if texto == "":
		return None
	return int(bool(valor))


def cargar_datos_local(ruta: Path) -> pd.DataFrame:
	return pd.read_csv(ruta)


def cargar_datos_supabase(tabla: str) -> pd.DataFrame:
	if create_client is None:
		raise RuntimeError("La libreria supabase no esta disponible en el entorno.")
	supabase_url = os.getenv("SUPABASE_URL")
	supabase_key = os.getenv("SUPABASE_KEY")
	if not supabase_url or not supabase_key:
		raise EnvironmentError("Faltan SUPABASE_URL y/o SUPABASE_KEY para leer desde Supabase.")
	cliente = create_client(supabase_url, supabase_key)
	respuesta = cliente.table(tabla).select("*").execute()
	datos = respuesta.data if hasattr(respuesta, "data") else respuesta.get("data", [])
	return pd.DataFrame(datos)


def validar_estructura(df: pd.DataFrame, logger: logging.Logger) -> None:
	faltantes = [columna for columna in EXPECTED_COLUMNS if columna not in df.columns]
	if faltantes:
		raise ValueError(f"Faltan columnas obligatorias: {faltantes}")
	extras = sorted(set(df.columns).difference(EXPECTED_COLUMNS))
	if extras:
		logger.warning("Columnas adicionales ignoradas: %s", extras)


def preparar_dataframe(df: pd.DataFrame, logger: logging.Logger, sensitive_columns: list[str]) -> pd.DataFrame:
	df = df.copy()
	validar_estructura(df, logger)
	df = pseudonimizar_columnas(df, sensitive_columns)

	for columna in NUMERIC_COLUMNS + [TARGET_COLUMN]:
		df[columna] = pd.to_numeric(df[columna], errors="coerce")

	df["previous_loan_defaults_on_file"] = df["previous_loan_defaults_on_file"].map(normalizar_booleano)
	for columna in CATEGORICAL_COLUMNS:
		if columna != "previous_loan_defaults_on_file":
			df[columna] = df[columna].astype(str).str.strip()
	df["previous_loan_defaults_on_file"] = df["previous_loan_defaults_on_file"].map({1: "Yes", 0: "No"})

	df = df.dropna(subset=NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)
	df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
	return df


def generar_univariado(df: pd.DataFrame, output_dir: Path, logger: logging.Logger) -> dict[str, Any]:
	output_dir.mkdir(parents=True, exist_ok=True)
	distribucion = df[TARGET_COLUMN].value_counts().sort_index()
	porcentajes = (distribucion / distribucion.sum() * 100).round(2)

	fig, ax = plt.subplots(figsize=(7, 4))
	ax.bar(distribucion.index.astype(str), distribucion.values, color=["#2F6690", "#E07A5F"])
	ax.set_title("Distribucion de la variable objetivo")
	ax.set_xlabel("loan_status")
	ax.set_ylabel("Cantidad")
	for idx, value in enumerate(distribucion.values):
		ax.text(idx, value, str(value), ha="center", va="bottom")
	fig.tight_layout()
	fig.savefig(output_dir / "univariate_target_distribution.png", dpi=160)
	plt.close(fig)
	logger.info("Analisis univariado generado")
	return {"counts": distribucion.to_dict(), "percentages": porcentajes.to_dict()}


def generar_bivariado(df: pd.DataFrame, output_dir: Path, logger: logging.Logger) -> list[str]:
	output_dir.mkdir(parents=True, exist_ok=True)
	archivos = []
	for columna in ["person_age", "person_income"]:
		fig, ax = plt.subplots(figsize=(8, 4))
		sns.boxplot(
			data=df,
			x=TARGET_COLUMN,
			y=columna,
			ax=ax,
			hue=TARGET_COLUMN,
			palette={0: "#81B29A", 1: "#F2CC8F"},
			dodge=False,
			legend=False,
		)
		ax.set_title(f"{columna} vs {TARGET_COLUMN}")
		fig.tight_layout()
		archivo = output_dir / f"boxplot_{columna}_vs_target.png"
		fig.savefig(archivo, dpi=160)
		plt.close(fig)
		archivos.append(str(archivo))

	numeric_df = df[NUMERIC_COLUMNS + [TARGET_COLUMN]].copy()
	fig, ax = plt.subplots(figsize=(10, 7))
	sns.heatmap(numeric_df.corr(numeric_only=True), annot=True, cmap="RdBu_r", center=0, ax=ax)
	ax.set_title("Correlacion entre variables numericas")
	fig.tight_layout()
	archivo = output_dir / "correlation_heatmap.png"
	fig.savefig(archivo, dpi=160)
	plt.close(fig)
	archivos.append(str(archivo))
	logger.info("Analisis bivariado generado")
	return archivos


def crear_one_hot_encoder() -> OneHotEncoder:
	try:
		return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
	except TypeError:
		return OneHotEncoder(handle_unknown="ignore", sparse=False)


def crear_pipeline(use_smote: bool, random_state: int):
	preprocesador = ColumnTransformer(
		transformers=[
			(
				"numericas",
				Pipeline([
					("imputer", SimpleImputer(strategy="median")),
					("scaler", StandardScaler()),
				]),
				NUMERIC_COLUMNS,
			),
			(
				"categoricas",
				Pipeline([
					("imputer", SimpleImputer(strategy="most_frequent")),
					("encoder", crear_one_hot_encoder()),
				]),
				CATEGORICAL_COLUMNS,
			),
		],
		remainder="drop",
	)

	modelo = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs", random_state=random_state)

	if use_smote:
		if SMOTE is None or ImbPipeline is None:
			raise RuntimeError("Se solicito SMOTE, pero imbalanced-learn no esta instalado.")
		return ImbPipeline([
			("preprocesado", preprocesador),
			("smote", SMOTE(random_state=random_state)),
			("modelo", modelo),
		])

	return Pipeline([
		("preprocesado", preprocesador),
		("modelo", modelo),
	])


def obtener_nombres_features(pipeline: Pipeline) -> list[str]:
	preprocesado = pipeline.named_steps["preprocesado"]
	encoder = preprocesado.named_transformers_["categoricas"].named_steps["encoder"]
	categoricas = list(encoder.get_feature_names_out(CATEGORICAL_COLUMNS))
	return NUMERIC_COLUMNS + categoricas


def evaluar_modelo(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
	y_pred = (y_prob >= threshold).astype(int)
	auc = roc_auc_score(y_true, y_prob)
	cm = confusion_matrix(y_true, y_pred)
	return {
		"auc": float(auc),
		"gini": float(2 * auc - 1),
		"threshold": float(threshold),
		"accuracy": float(accuracy_score(y_true, y_pred)),
		"precision": float(precision_score(y_true, y_pred, zero_division=0)),
		"recall": float(recall_score(y_true, y_pred, zero_division=0)),
		"f1": float(f1_score(y_true, y_pred, zero_division=0)),
		"confusion_matrix": cm.tolist(),
		"classification_report": classification_report(y_true, y_pred, zero_division=0, output_dict=True),
		"predictions": y_pred.tolist(),
	}


def guardar_graficos_resultados(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray, output_dir: Path, feature_names: list[str], coefficients: np.ndarray) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)

	cm = confusion_matrix(y_true, y_pred)
	fig, ax = plt.subplots(figsize=(6, 5))
	sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
	ax.set_xlabel("Prediccion")
	ax.set_ylabel("Real")
	ax.set_title("Matriz de confusion")
	fig.tight_layout()
	fig.savefig(output_dir / "confusion_matrix.png", dpi=180)
	plt.close(fig)

	fpr, tpr, _ = roc_curve(y_true, y_prob)
	fig, ax = plt.subplots(figsize=(6, 5))
	ax.plot(fpr, tpr, label="ROC")
	ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
	ax.set_xlabel("False Positive Rate")
	ax.set_ylabel("True Positive Rate")
	ax.set_title("Curva ROC")
	ax.legend()
	fig.tight_layout()
	fig.savefig(output_dir / "roc_curve.png", dpi=180)
	plt.close(fig)

	importancias = pd.DataFrame(
		{
			"feature": feature_names,
			"coefficient": coefficients,
			"abs_coefficient": np.abs(coefficients),
		}
	).sort_values("abs_coefficient", ascending=False).head(15)
	fig, ax = plt.subplots(figsize=(10, 6))
	sns.barplot(data=importancias, x="abs_coefficient", y="feature", ax=ax, color="#2F6690")
	ax.set_title("Importancia de variables por coeficiente absoluto")
	ax.set_xlabel("|coeficiente|")
	ax.set_ylabel("Variable")
	fig.tight_layout()
	fig.savefig(output_dir / "feature_importance.png", dpi=180)
	plt.close(fig)


def guardar_metadatos_bi(df: pd.DataFrame, y_pred: np.ndarray, output_dir: Path) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)
	y_pred = np.asarray(y_pred)
	bi_df = df[["person_gender", "person_age", TARGET_COLUMN]].copy()
	bi_df["approval"] = (y_pred == 0).astype(int)
	bi_df["rejection"] = 1 - bi_df["approval"]
	bi_df["age_group"] = pd.cut(
		bi_df["person_age"],
		bins=[17, 25, 35, 45, 55, 100],
		labels=["18-25", "26-35", "36-45", "46-55", "56+"],
	)
	agregado_genero = bi_df.groupby("person_gender", dropna=False)[["approval", "rejection"]].mean().reset_index()
	agregado_edad = bi_df.groupby("age_group", dropna=False)[["approval", "rejection"]].mean().reset_index()
	agregado_genero.to_csv(output_dir / "fairness_by_gender.csv", index=False)
	agregado_edad.to_csv(output_dir / "fairness_by_age_group.csv", index=False)


def guardar_metricas_json(metricas: dict[str, Any], output_dir: Path) -> Path:
	output_dir.mkdir(parents=True, exist_ok=True)
	archivo = output_dir / "model_metrics.json"
	with archivo.open("w", encoding="utf-8") as f:
		json.dump(metricas, f, indent=2, ensure_ascii=True)
	return archivo


def es_clave_service_role(clave: str | None) -> bool:
	if not clave:
		return False
	partes = clave.split(".")
	if len(partes) < 2:
		return False
	payload = partes[1]
	padding = "=" * (-len(payload) % 4)
	try:
		contenido = base64.urlsafe_b64decode(payload + padding).decode("utf-8")
		return json.loads(contenido).get("role") == "service_role"
	except Exception:
		return False


def guardar_perfil_base(df: pd.DataFrame, output_dir: Path) -> Path:
	output_dir.mkdir(parents=True, exist_ok=True)
	perfil = {
		"numeric": {},
		"categorical": {},
	}

	for columna in NUMERIC_COLUMNS:
		perfil["numeric"][columna] = {
			"mean": float(df[columna].mean()),
			"std": float(df[columna].std(ddof=0) if df[columna].std(ddof=0) and not np.isnan(df[columna].std(ddof=0)) else 0.0),
			"min": float(df[columna].min()),
			"max": float(df[columna].max()),
		}

	for columna in CATEGORICAL_COLUMNS:
		frecuencias = df[columna].astype(str).value_counts(normalize=True, dropna=False).to_dict()
		perfil["categorical"][columna] = {str(clave): float(valor) for clave, valor in frecuencias.items()}

	archivo = output_dir / "baseline_profile.json"
	with archivo.open("w", encoding="utf-8") as f:
		json.dump(perfil, f, indent=2, ensure_ascii=True)
	return archivo


def construir_payload_supabase(metricas: dict[str, Any], feature_names: list[str], coefficients: np.ndarray, run_name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
	run_payload = {
		"run_name": run_name,
		"source": metricas.get("source"),
		"threshold": metricas.get("threshold"),
		"train_rows": metricas.get("train_rows"),
		"test_rows": metricas.get("test_rows"),
		"auc": metricas.get("auc"),
		"gini": metricas.get("gini"),
		"accuracy": metricas.get("accuracy"),
		"precision": metricas.get("precision"),
		"recall": metricas.get("recall"),
		"f1": metricas.get("f1"),
		"tn": metricas.get("confusion_matrix", [[None, None], [None, None]])[0][0],
		"fp": metricas.get("confusion_matrix", [[None, None], [None, None]])[0][1],
		"fn": metricas.get("confusion_matrix", [[None, None], [None, None]])[1][0],
		"tp": metricas.get("confusion_matrix", [[None, None], [None, None]])[1][1],
		"training_seconds": metricas.get("training_seconds"),
		"memory_rss_mb": metricas.get("memory_rss_mb"),
	}

	feature_payload = []
	for orden, (feature, coeficiente) in enumerate(zip(feature_names, coefficients, strict=True), start=1):
		feature_payload.append(
			{
				"run_name": run_name,
				"feature_name": feature,
				"coefficient": float(coeficiente),
				"abs_coefficient": float(abs(coeficiente)),
				"feature_rank": orden,
			}
		)
	return run_payload, feature_payload


def subir_resultados_a_supabase(metricas: dict[str, Any], feature_names: list[str], coefficients: np.ndarray, run_name: str) -> None:
	if create_client is None:
		return
	supabase_url = os.getenv("SUPABASE_URL")
	supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
	if not supabase_url or not supabase_key or not es_clave_service_role(supabase_key):
		return
	cliente = create_client(supabase_url, supabase_key)
	run_payload, feature_payload = construir_payload_supabase(metricas, feature_names, coefficients, run_name)
	cliente.table("loan_model_runs").insert(run_payload).execute()
	cliente.table("loan_model_feature_importance").insert(feature_payload).execute()


def main() -> None:
	parser = argparse.ArgumentParser(description="Entrena regresion logistica para riesgo de incumplimiento")
	parser.add_argument("--source", choices=["local", "supabase"], default="local")
	parser.add_argument("--input", type=Path, default=Path("data/processed/02_loan_data_clean.csv"))
	parser.add_argument("--supabase-table", type=str, default="loan_data_clean")
	parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
	parser.add_argument("--model-output", type=Path, default=Path("artifacts/models/logistic_model.joblib"))
	parser.add_argument("--test-size", type=float, default=0.2)
	parser.add_argument("--threshold", type=float, default=0.4)
	parser.add_argument("--random-state", type=int, default=42)
	parser.add_argument("--use-smote", action="store_true")
	parser.add_argument("--sensitive-columns", nargs="*", default=[])
	parser.add_argument("--sync-supabase", action="store_true")
	parser.add_argument("--log-file", type=Path, default=Path("artifacts/logs/training.log"))
	args = parser.parse_args()

	logger = configurar_logger(args.log_file)
	logger.info("Inicio del entrenamiento")
	inicio_total = captura_recursos()

	if args.source == "supabase":
		df = cargar_datos_supabase(args.supabase_table)
		logger.info("Datos cargados desde Supabase: %s filas", len(df))
	else:
		df = cargar_datos_local(args.input)
		logger.info("Datos cargados desde archivo: %s filas", len(df))

	df = preparar_dataframe(df, logger, args.sensitive_columns)
	resumen_uni = generar_univariado(df, args.artifacts_dir / "plots", logger)
	_ = generar_bivariado(df, args.artifacts_dir / "plots", logger)

	X = df.drop(columns=[TARGET_COLUMN])
	y = df[TARGET_COLUMN].astype(int)
	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=args.test_size,
		stratify=y,
		random_state=args.random_state,
	)

	modelo = crear_pipeline(args.use_smote, args.random_state)
	logger.info("Iniciando entrenamiento | class_weight=balanced | threshold=%.2f", args.threshold)
	inicio_entrenamiento = captura_recursos()
	modelo.fit(X_train, y_train)
	fin_entrenamiento = registrar_recursos(logger, "Entrenamiento completado", inicio_entrenamiento)

	probabilidades = modelo.predict_proba(X_test)[:, 1]
	evaluacion = evaluar_modelo(y_test.to_numpy(), probabilidades, args.threshold)
	evaluacion["source"] = args.source
	evaluacion["train_rows"] = int(len(X_train))
	evaluacion["test_rows"] = int(len(X_test))
	evaluacion["positive_rate"] = float(y.mean())
	evaluacion["target_distribution"] = resumen_uni
	evaluacion["training_seconds"] = float(fin_entrenamiento.seconds - inicio_entrenamiento.seconds)
	evaluacion["memory_rss_mb"] = float(fin_entrenamiento.rss_mb)

	coeficientes = modelo.named_steps["modelo"].coef_[0]
	feature_names = obtener_nombres_features(modelo)
	guardar_metricas_json(evaluacion, args.artifacts_dir / "metrics")
	guardar_perfil_base(df, args.artifacts_dir / "metrics")
	guardar_graficos_resultados(y_test.to_numpy(), probabilidades, evaluacion["predictions"], args.artifacts_dir / "plots", feature_names, coeficientes)
	guardar_metadatos_bi(df.loc[y_test.index], evaluacion["predictions"], args.artifacts_dir / "bi")
	args.model_output.parent.mkdir(parents=True, exist_ok=True)
	joblib.dump(modelo, args.model_output)

	if args.sync_supabase:
		subir_resultados_a_supabase(evaluacion, feature_names, coeficientes, run_name=args.model_output.stem)

	fin_total = registrar_recursos(logger, "Proceso completo", inicio_total)
	logger.info("AUC=%.4f | Gini=%.4f | FN=%s | FP=%s", evaluacion["auc"], evaluacion["gini"], evaluacion["confusion_matrix"][1][0], evaluacion["confusion_matrix"][0][1])
	logger.info("Artefactos generados en: %s", args.artifacts_dir)
	logger.info("Modelo guardado en: %s", args.model_output)
	logger.info("Uso final | rss=%.1f MB | cpu=%.1f%%", fin_total.rss_mb, fin_total.cpu_percent)


if __name__ == "__main__":
	main()
