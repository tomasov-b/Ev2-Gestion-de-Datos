from __future__ import annotations

"""Sube un CSV limpio a Supabase con validación, reintentos y logging.

Requiere `SUPABASE_URL` y `SUPABASE_KEY` en el entorno.
Uso:
	python script_subir_data.py --entrada data/processed/02_loan_data_clean.csv --tabla loan_data_clean
"""

import argparse
import csv
import hashlib
import logging
import os
import base64
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from supabase import create_client

try:
	from dotenv import load_dotenv
except ImportError:  # pragma: no cover - entorno sin python-dotenv
	def load_dotenv(*args, **kwargs):
		return False


def normalizar_valor_entorno(nombre: str, valor: str | None) -> str | None:
	if valor is None:
		return None
	texto = valor.strip().strip('"').strip("'")
	if nombre == "SUPABASE_URL":
		return texto.rstrip("/")
	if nombre == "SUPABASE_KEY":
		if texto.startswith("eyJ"):
			return texto
		indice = texto.find("eyJ")
		if indice != -1:
			return texto[indice:]
		return texto
	return texto


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


def cargar_env_local(logger: logging.Logger | None = None) -> bool:
	candidatos = [Path.cwd() / ".env", Path(__file__).resolve().parent / ".env"]
	for archivo in candidatos:
		if not archivo.exists():
			continue
		cargado = False
		try:
			cargado = bool(load_dotenv(archivo))
		except TypeError:
			cargado = bool(load_dotenv())

		if cargado:
			if logger:
				logger.info("Variables cargadas desde: %s", archivo)
			return True

		with archivo.open(encoding="utf-8") as f:
			for linea in f:
				linea = linea.strip()
				if not linea or linea.startswith("#") or "=" not in linea:
					continue
				clave, valor = linea.split("=", 1)
				clave = clave.strip()
				valor = valor.strip()
				if clave in {"SUPABASE_URL", "SUPABASE_KEY"}:
					os.environ[clave] = normalizar_valor_entorno(clave, valor) or ""
			if logger:
				logger.info("Variables cargadas manualmente desde: %s", archivo)
			return True
	return False


COLUMNAS_ESPERADAS = [
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


def configurar_logger() -> logging.Logger:
	logger = logging.getLogger("subida_supabase")
	logger.setLevel(logging.INFO)
	logger.handlers.clear()
	logger.propagate = False

	formato = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
	handler = logging.StreamHandler()
	handler.setFormatter(formato)
	logger.addHandler(handler)
	return logger


def to_int(valor: Any) -> int | None:
	if valor is None or str(valor).strip() == "":
		return None
	try:
		return int(round(float(str(valor))))
	except Exception:
		return None


def to_float(valor: Any) -> float | None:
	if valor is None or str(valor).strip() == "":
		return None
	try:
		return float(str(valor))
	except Exception:
		return None


def to_bool_yesno(valor: Any) -> bool:
	texto = str(valor).strip().lower()
	return texto in {"yes", "y", "true", "1"}


def hash_value(valor: Any) -> str | None:
	texto = str(valor).strip()
	if not texto:
		return None
	return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def transformar_registro(reg: Dict[str, Any], hash_columns: set[str] | None = None) -> Dict[str, Any]:
	hash_columns = hash_columns or set()
	transformado = {
		"person_age": to_int(reg.get("person_age")),
		"person_gender": (reg.get("person_gender") or "").strip(),
		"person_education": (reg.get("person_education") or "").strip(),
		"person_income": to_int(reg.get("person_income")),
		"person_emp_exp": to_int(reg.get("person_emp_exp")),
		"person_home_ownership": (reg.get("person_home_ownership") or "").strip(),
		"loan_amnt": to_int(reg.get("loan_amnt")),
		"loan_intent": (reg.get("loan_intent") or "").strip(),
		"loan_int_rate": to_float(reg.get("loan_int_rate")),
		"loan_percent_income": to_float(reg.get("loan_percent_income")),
		"cb_person_cred_hist_length": to_int(reg.get("cb_person_cred_hist_length")),
		"credit_score": to_int(reg.get("credit_score")),
		"previous_loan_defaults_on_file": to_bool_yesno(reg.get("previous_loan_defaults_on_file")),
		"loan_status": to_int(reg.get("loan_status")),
	}

	for columna in hash_columns:
		if columna in reg:
			transformado[columna] = hash_value(reg.get(columna))

	return transformado


def leer_csv(ruta: Path) -> List[Dict[str, Any]]:
	with ruta.open(newline="", encoding="utf-8") as f:
		lector = csv.DictReader(f)
		return [row for row in lector]


def validar_columnas(registros: List[Dict[str, Any]], logger: logging.Logger) -> None:
	if not registros:
		raise ValueError("El archivo de entrada no contiene filas.")

	columnas = set(registros[0].keys())
	faltantes = [columna for columna in COLUMNAS_ESPERADAS if columna not in columnas]
	if faltantes:
		raise ValueError(f"Faltan columnas obligatorias en el CSV: {faltantes}")

	columnas_extra = sorted(columnas.difference(COLUMNAS_ESPERADAS))
	if columnas_extra:
		logger.warning("Se detectaron columnas extra en el CSV: %s", columnas_extra)


def crear_cliente_supabase():
	logger = logging.getLogger("subida_supabase")
	cargar_env_local(logger)
	supabase_url = normalizar_valor_entorno("SUPABASE_URL", os.getenv("SUPABASE_URL"))
	supabase_write_key = normalizar_valor_entorno("SUPABASE_SERVICE_ROLE_KEY", os.getenv("SUPABASE_SERVICE_ROLE_KEY"))
	if not supabase_write_key:
		supabase_write_key = normalizar_valor_entorno("SUPABASE_KEY", os.getenv("SUPABASE_KEY"))
	if not supabase_url or not supabase_write_key:
		raise EnvironmentError(
			"Faltan las variables SUPABASE_URL y/o SUPABASE_SERVICE_ROLE_KEY en el entorno. "
			"Para escribir en tablas con RLS necesitas una clave de servicio, no la anon key."
		)
	if not es_clave_service_role(supabase_write_key):
		raise EnvironmentError(
			"La clave encontrada no parece ser service_role. "
			"Con RLS habilitado, la insercion requiere SUPABASE_SERVICE_ROLE_KEY. "
			"No uses la anon key para este script de carga."
		)
	os.environ["SUPABASE_URL"] = supabase_url
	os.environ["SUPABASE_SERVICE_ROLE_KEY"] = supabase_write_key
	return create_client(supabase_url, supabase_write_key)


def insertar_lotes(supabase, tabla: str, filas: List[Dict[str, Any]], lote: int = 500, reintentos: int = 3, logger: logging.Logger | None = None) -> None:
	logger = logger or logging.getLogger("subida_supabase")
	total = len(filas)
	for i in range(0, total, lote):
		batch = filas[i : i + lote]
		err = None
		for intento in range(1, reintentos + 1):
			try:
				resp = supabase.table(tabla).insert(batch).execute()
				if hasattr(resp, "error") and resp.error:
					err = resp.error
				elif isinstance(resp, dict) and resp.get("error"):
					err = resp.get("error")
				else:
					logger.info("Insertado lote %s-%s (%s filas)", i, i + len(batch), len(batch))
					break
			except Exception as exc:
				err = exc

			if intento < reintentos:
				espera = 2 ** (intento - 1)
				logger.warning(
					"Fallo lote %s-%s en intento %s/%s: %s. Reintentando en %s s.",
					i,
					i + len(batch),
					intento,
					reintentos,
					err,
					espera,
				)
				time.sleep(espera)
			else:
				logger.error("No se pudo insertar el lote %s-%s: %s", i, i + len(batch), err)


def main() -> None:
	parser = argparse.ArgumentParser(description="Sube CSV limpio a Supabase")
	parser.add_argument("--entrada", type=Path, required=False, default=Path("data/processed/02_loan_data_clean.csv"))
	parser.add_argument("--tabla", type=str, required=False, default="loan_data_clean")
	parser.add_argument("--batch", type=int, required=False, default=500)
	parser.add_argument("--hash-columns", nargs="*", default=[], help="Columnas opcionales a seudonimizar antes de subir")
	args = parser.parse_args()

	logger = configurar_logger()
	logger.info("Leyendo CSV desde: %s", args.entrada)
	registros = leer_csv(args.entrada)
	logger.info("Filas leídas: %s", len(registros))
	validar_columnas(registros, logger)

	registros_transformados = [transformar_registro(r, set(args.hash_columns)) for r in registros]
	supabase = crear_cliente_supabase()

	logger.info("Subiendo a tabla: %s en Supabase (batch=%s)", args.tabla, args.batch)
	insertar_lotes(supabase, args.tabla, registros_transformados, lote=args.batch, logger=logger)
	logger.info("Proceso completado.")


if __name__ == "__main__":
	main()

