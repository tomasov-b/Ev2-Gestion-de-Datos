from __future__ import annotations

"""Aplica el modelo entrenado para scoring de incumplimiento y monitoreo de drift.

- Carga el modelo serializado generado por `train_loan_default_model.py`.
- Lee datos desde archivo local o Supabase.
- Genera predicciones, calculos de drift y, si hay etiquetas reales, AUC/Gini.
- Puede persistir predicciones y eventos de drift en Supabase para BI y auditoria.
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
import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import roc_auc_score

try:
	from supabase import create_client
except Exception:  # pragma: no cover - opcional para ejecucion local
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
	logger = logging.getLogger("loan_inference")
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


def validar_estructura(df: pd.DataFrame, logger: logging.Logger, require_target: bool) -> None:
	expected = EXPECTED_COLUMNS if require_target else [col for col in EXPECTED_COLUMNS if col != TARGET_COLUMN]
	faltantes = [columna for columna in expected if columna not in df.columns]
	if faltantes:
		raise ValueError(f"Faltan columnas obligatorias: {faltantes}")
	extras = sorted(set(df.columns).difference(expected))
	if extras:
		logger.warning("Columnas adicionales ignoradas: %s", extras)


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


def preparar_dataframe(df: pd.DataFrame, logger: logging.Logger, require_target: bool) -> pd.DataFrame:
	df = df.copy()
	validar_estructura(df, logger, require_target=require_target)
	for columna in NUMERIC_COLUMNS:
		df[columna] = pd.to_numeric(df[columna], errors="coerce")
	if require_target and TARGET_COLUMN in df.columns:
		df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")

	df["previous_loan_defaults_on_file"] = df["previous_loan_defaults_on_file"].map(normalizar_booleano)
	for columna in CATEGORICAL_COLUMNS:
		if columna != "previous_loan_defaults_on_file":
			df[columna] = df[columna].astype(str).str.strip()
	df["previous_loan_defaults_on_file"] = df["previous_loan_defaults_on_file"].map({1: "Yes", 0: "No"})
	df = df.dropna(subset=NUMERIC_COLUMNS + CATEGORICAL_COLUMNS).reset_index(drop=True)
	if require_target:
		df = df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
		df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
	return df


def cargar_artifactos(model_path: Path, metrics_path: Path, baseline_path: Path) -> tuple[Any, dict[str, Any], dict[str, Any]]:
	modelo = joblib.load(model_path)
	with metrics_path.open("r", encoding="utf-8") as f:
		metricas = json.load(f)
	with baseline_path.open("r", encoding="utf-8") as f:
		baseline = json.load(f)
	return modelo, metricas, baseline


def compute_category_psi(expected_distribution: dict[str, float], actual_series: pd.Series) -> float:
	expected = pd.Series(expected_distribution, dtype=float)
	actual = actual_series.astype(str).value_counts(normalize=True, dropna=False).reindex(expected.index, fill_value=0.0)
	expected = expected.replace(0, 1e-6)
	actual = actual.replace(0, 1e-6)
	psi = ((actual - expected) * np.log(actual / expected)).sum()
	return float(psi)


def construir_firma_fila(df: pd.DataFrame, exclude_columns: list[str] | None = None) -> pd.Series:
	exclude_columns = exclude_columns or []
	columnas = [col for col in df.columns if col not in {TARGET_COLUMN, *exclude_columns}]
	return df[columnas].astype(str).agg("|".join, axis=1).map(lambda texto: sha256(texto.encode("utf-8")).hexdigest())


def etiquetar_grupos(df: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()
	df["age_group"] = pd.cut(
		df["person_age"],
		bins=[17, 25, 35, 45, 55, 100],
		labels=["18-25", "26-35", "36-45", "46-55", "56+"],
	)
	return df


def preparar_registros_salida(df: pd.DataFrame, probabilities: np.ndarray, threshold: float, run_name: str) -> pd.DataFrame:
	output = df.copy()
	output["source_record_hash"] = construir_firma_fila(output)
	output["run_name"] = run_name
	output["prediction_score"] = probabilities
	output["prediction_label"] = (probabilities >= threshold).astype(int)
	output = etiquetar_grupos(output)
	return output


def evaluar_drift(df: pd.DataFrame, baseline: dict[str, Any]) -> dict[str, float]:
	drift: dict[str, float] = {}
	for columna, stats in baseline.get("numeric", {}).items():
		if columna in df.columns:
			baseline_mean = float(stats.get("mean", 0.0))
			baseline_std = max(float(stats.get("std", 0.0)), 1e-6)
			current_mean = float(df[columna].astype(float).mean())
			drift[f"shift_{columna}"] = abs(current_mean - baseline_mean) / baseline_std
	for columna, dist in baseline.get("categorical", {}).items():
		if columna in df.columns:
			drift[f"psi_{columna}"] = compute_category_psi(dist, df[columna])
	return drift


def guardar_metricas(metricas: dict[str, Any], output_dir: Path) -> Path:
	output_dir.mkdir(parents=True, exist_ok=True)
	archivo = output_dir / "inference_metrics.json"
	with archivo.open("w", encoding="utf-8") as f:
		json.dump(metricas, f, indent=2, ensure_ascii=True)
	return archivo


def guardar_drift(drift: dict[str, float], output_dir: Path) -> Path:
	output_dir.mkdir(parents=True, exist_ok=True)
	archivo = output_dir / "drift_metrics.json"
	with archivo.open("w", encoding="utf-8") as f:
		json.dump(drift, f, indent=2, ensure_ascii=True)
	return archivo


def guardar_predicciones(predictions: pd.DataFrame, output_dir: Path) -> Path:
	output_dir.mkdir(parents=True, exist_ok=True)
	archivo = output_dir / "predictions.csv"
	predictions.to_csv(archivo, index=False)
	return archivo


def registrar_en_supabase(predictions: pd.DataFrame, metricas: dict[str, Any], drift: dict[str, float]) -> None:
	if create_client is None:
		return
	supabase_url = os.getenv("SUPABASE_URL")
	supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
	if not supabase_url or not supabase_key or not es_clave_service_role(supabase_key):
		return
	cliente = create_client(supabase_url, supabase_key)
	actual_columna = "loan_status_actual" if "loan_status_actual" in predictions.columns else None
	prediction_rows = predictions[[
		"run_name",
		"source_record_hash",
		"person_age",
		"person_gender",
		"age_group",
		"prediction_score",
		"prediction_label",
	] + ([actual_columna] if actual_columna else [])]
	prediction_payload = prediction_rows.to_dict(orient="records")
	cliente.table("loan_model_predictions").insert(prediction_payload).execute()

	drift_payload = [
		{
			"run_name": metricas.get("run_name"),
			"metric_name": metric_name,
			"metric_value": float(value),
			"baseline_value": float(metricas.get("baseline_gini", 0.0)) if metric_name == "gini" else None,
			"delta": float(metricas.get("gini_drop", 0.0)) if metric_name == "gini" else (float(value) if metric_name.startswith("shift_") else None),
			"alert_level": "high" if ((metric_name == "gini" and float(value) < 0.4) or value >= 1.0 or (metric_name.startswith("psi_") and value >= 0.25) or (metric_name.startswith("shift_") and value >= 2.0)) else "medium" if ((metric_name.startswith("psi_") and value >= 0.1) or (metric_name.startswith("shift_") and value >= 1.0)) else "low",
		}
		for metric_name, value in drift.items()
	]
	if drift_payload:
		cliente.table("loan_model_drift_events").insert(drift_payload).execute()


def fairness_summary(predictions: pd.DataFrame, output_dir: Path) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)
	predictions["approval"] = (predictions["prediction_label"] == 0).astype(int)
	predictions["rejection"] = 1 - predictions["approval"]
	gender_summary = predictions.groupby("person_gender", dropna=False)[["approval", "rejection"]].mean().reset_index()
	age_summary = predictions.groupby("age_group", dropna=False)[["approval", "rejection"]].mean().reset_index()
	gender_summary.to_csv(output_dir / "fairness_by_gender.csv", index=False)
	age_summary.to_csv(output_dir / "fairness_by_age_group.csv", index=False)


def main() -> None:
	parser = argparse.ArgumentParser(description="Scoring de incumplimiento y monitoreo de drift")
	parser.add_argument("--source", choices=["local", "supabase"], default="local")
	parser.add_argument("--input", type=Path, default=Path("data/processed/02_loan_data_clean.csv"))
	parser.add_argument("--supabase-table", type=str, default="loan_data_clean")
	parser.add_argument("--model-path", type=Path, default=Path("artifacts/models/logistic_model.joblib"))
	parser.add_argument("--metrics-path", type=Path, default=Path("artifacts/metrics/model_metrics.json"))
	parser.add_argument("--baseline-path", type=Path, default=Path("artifacts/metrics/baseline_profile.json"))
	parser.add_argument("--output-dir", type=Path, default=Path("artifacts/inference"))
	parser.add_argument("--threshold", type=float, default=None)
	parser.add_argument("--log-file", type=Path, default=Path("artifacts/logs/inference.log"))
	parser.add_argument("--sync-supabase", action="store_true")
	args = parser.parse_args()

	logger = configurar_logger(args.log_file)
	inicio_total = captura_recursos()
	logger.info("Inicio de inferencia")

	modelo, metricas_base, baseline = cargar_artifactos(args.model_path, args.metrics_path, args.baseline_path)
	threshold = args.threshold if args.threshold is not None else float(metricas_base.get("threshold", 0.4))
	run_name = args.model_path.stem

	if args.source == "supabase":
		df = cargar_datos_supabase(args.supabase_table)
		logger.info("Datos leidos desde Supabase: %s filas", len(df))
	else:
		df = cargar_datos_local(args.input)
		logger.info("Datos leidos desde archivo: %s filas", len(df))

	require_target = TARGET_COLUMN in df.columns
	df = preparar_dataframe(df, logger, require_target=require_target)
	inicio_scoring = captura_recursos()
	probabilities = modelo.predict_proba(df.drop(columns=[TARGET_COLUMN], errors="ignore"))[:, 1]
	fin_scoring = registrar_recursos(logger, "Scoring completado", inicio_scoring)

	predictions = preparar_registros_salida(df, probabilities, threshold=threshold, run_name=run_name)
	drift = evaluar_drift(predictions.drop(columns=[TARGET_COLUMN], errors="ignore"), baseline)
	drift["mean_prediction_score"] = float(predictions["prediction_score"].mean())
	drift["default_rate_predicted"] = float(predictions["prediction_label"].mean())

	metricas: dict[str, Any] = {
		"run_name": run_name,
		"threshold": threshold,
		"rows": int(len(predictions)),
		"mean_prediction_score": drift["mean_prediction_score"],
		"predicted_default_rate": drift["default_rate_predicted"],
		"scoring_seconds": float(fin_scoring.seconds - inicio_scoring.seconds),
		"memory_rss_mb": float(fin_scoring.rss_mb),
	}

	if require_target:
		y_true = df[TARGET_COLUMN].astype(int).to_numpy()
		auc = roc_auc_score(y_true, probabilities)
		gini = float(2 * auc - 1)
		metricas["auc"] = float(auc)
		metricas["gini"] = gini
		metricas["baseline_gini"] = float(metricas_base.get("gini", 0.0))
		metricas["gini_drop"] = float(metricas["baseline_gini"] - gini)
		predictions["loan_status_actual"] = y_true
		logger.info("AUC=%.4f | Gini=%.4f | Gini baseline=%.4f | delta=%.4f", auc, gini, metricas["baseline_gini"], metricas["gini_drop"])
		if gini < 0.4 or metricas["gini_drop"] >= 0.1:
			logger.warning("Alerta: degradacion de Gini detectada en produccion")

	guardar_metricas(metricas, args.output_dir / "metrics")
	guardar_drift(drift, args.output_dir / "metrics")
	guardar_predicciones(predictions, args.output_dir / "predictions")
	fairness_summary(predictions, args.output_dir / "bi")

	if args.sync_supabase:
		registrar_en_supabase(predictions, metricas, drift)

	fin_total = registrar_recursos(logger, "Inferencia completa", inicio_total)
	logger.info("Uso final | rss=%.1f MB | cpu=%.1f%%", fin_total.rss_mb, fin_total.cpu_percent)
	logger.info("Resultados guardados en: %s", args.output_dir)


if __name__ == "__main__":
	main()
