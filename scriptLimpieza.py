from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any


COLUMNAS_ORDEN = [
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

CAMPOS_ENTEROS = {
	"person_age",
	"person_income",
	"person_emp_exp",
	"loan_amnt",
	"cb_person_cred_hist_length",
	"credit_score",
	"loan_status",
}

CAMPOS_FLOTANTES = {
	"loan_int_rate",
	"loan_percent_income",
}

MAPA_GENERO = {
	"female": "female",
	"male": "male",
}

MAPA_EDUCACION = {
	"high school": "High School",
	"associate": "Associate",
	"bachelor": "Bachelor",
	"master": "Master",
	"doctorate": "Doctorate",
}

MAPA_VIVIENDA = {
	"rent": "RENT",
	"own": "OWN",
	"mortgage": "MORTGAGE",
	"other": "OTHER",
}

MAPA_PROPOSITO = {
	"personal": "PERSONAL",
	"education": "EDUCATION",
	"medical": "MEDICAL",
	"venture": "VENTURE",
	"homeimprovement": "HOMEIMPROVEMENT",
	"debtconsolidation": "DEBTCONSOLIDATION",
}

MAPA_DEFAULTS = {
	"yes": "Yes",
	"no": "No",
}


def normalizar_texto(valor: Any) -> str:
	if valor is None:
		return ""
	return " ".join(str(valor).strip().split())


def convertir_flotante(valor: Any) -> float | None:
	texto = normalizar_texto(valor)
	if not texto:
		return None

	try:
		return float(texto)
	except ValueError:
		return None


def convertir_entero(valor: Any) -> int | None:
	numero = convertir_flotante(valor)
	if numero is None:
		return None
	return int(round(numero))


def normalizar_categoria(valor: Any, mapa: dict[str, str]) -> str | None:
	texto = normalizar_texto(valor)
	if not texto:
		return None

	return mapa.get(texto.casefold(), texto)


def normalizar_tasa_interes(valor: Any) -> float | None:
	tasa = convertir_flotante(valor)
	if tasa is None:
		return None

	if tasa > 100:
		tasa = tasa / 100

	return round(tasa, 2)


def normalizar_proporcion_ingreso(valor: Any) -> float | None:
	proporcion = convertir_flotante(valor)
	if proporcion is None:
		return None

	if proporcion > 1:
		proporcion = proporcion / 100

	return round(proporcion, 4)


def normalizar_registro(registro: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
	limpio: dict[str, Any] = {}

	limpio["person_age"] = convertir_entero(registro.get("person_age"))
	limpio["person_gender"] = normalizar_categoria(registro.get("person_gender"), MAPA_GENERO)
	limpio["person_education"] = normalizar_categoria(registro.get("person_education"), MAPA_EDUCACION)
	limpio["person_income"] = convertir_entero(registro.get("person_income"))
	limpio["person_emp_exp"] = convertir_entero(registro.get("person_emp_exp"))
	limpio["person_home_ownership"] = normalizar_categoria(registro.get("person_home_ownership"), MAPA_VIVIENDA)
	limpio["loan_amnt"] = convertir_entero(registro.get("loan_amnt"))
	limpio["loan_intent"] = normalizar_categoria(registro.get("loan_intent"), MAPA_PROPOSITO)
	limpio["loan_int_rate"] = normalizar_tasa_interes(registro.get("loan_int_rate"))
	limpio["loan_percent_income"] = normalizar_proporcion_ingreso(registro.get("loan_percent_income"))
	limpio["cb_person_cred_hist_length"] = convertir_entero(registro.get("cb_person_cred_hist_length"))
	limpio["credit_score"] = convertir_entero(registro.get("credit_score"))
	limpio["previous_loan_defaults_on_file"] = normalizar_categoria(
		registro.get("previous_loan_defaults_on_file"),
		MAPA_DEFAULTS,
	)
	limpio["loan_status"] = convertir_entero(registro.get("loan_status"))

	for columna in COLUMNAS_ORDEN:
		if limpio[columna] is None:
			return None, f"valor_nulo_o_invalido_en_{columna}"

	if not 18 <= limpio["person_age"] <= 100:
		return None, "edad_fuera_de_rango"

	if limpio["person_emp_exp"] < 0:
		return None, "experiencia_negativa"

	if limpio["person_emp_exp"] > limpio["person_age"]:
		return None, "experiencia_mayor_que_edad"

	if not 300 <= limpio["credit_score"] <= 850:
		return None, "credit_score_fuera_de_rango"

	if limpio["person_income"] <= 0:
		return None, "ingreso_no_positivo"

	if limpio["loan_amnt"] <= 0:
		return None, "monto_no_positivo"

	if not 0 <= limpio["loan_int_rate"] <= 100:
		return None, "tasa_fuera_de_rango"

	if not 0 <= limpio["loan_percent_income"] <= 1:
		return None, "proporcion_fuera_de_rango"

	if limpio["loan_status"] not in {0, 1}:
		return None, "target_invalido"

	return limpio, None


def leer_datos_crudos(ruta_entrada: Path) -> list[dict[str, Any]]:
	with ruta_entrada.open(newline="", encoding="utf-8") as archivo:
		lector = csv.DictReader(archivo)
		return list(lector)


def limpiar_datos(registros: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
	registros_limpios: list[dict[str, Any]] = []
	motivos = Counter()
	vistos: set[tuple[Any, ...]] = set()

	for registro in registros:
		limpio, motivo = normalizar_registro(registro)
		if limpio is None:
			motivos[motivo or "registro_invalido"] += 1
			continue

		firma = tuple(limpio[columna] for columna in COLUMNAS_ORDEN)
		if firma in vistos:
			motivos["duplicado"] += 1
			continue

		vistos.add(firma)
		registros_limpios.append(limpio)

	return registros_limpios, dict(motivos)


def guardar_datos_limpios(ruta_salida: Path, registros: list[dict[str, Any]]) -> None:
	ruta_salida.parent.mkdir(parents=True, exist_ok=True)

	with ruta_salida.open("w", newline="", encoding="utf-8") as archivo:
		escritor = csv.DictWriter(archivo, fieldnames=COLUMNAS_ORDEN)
		escritor.writeheader()
		escritor.writerows(registros)


def procesar_archivo(ruta_entrada: Path, ruta_salida: Path) -> dict[str, Any]:
	registros = leer_datos_crudos(ruta_entrada)
	registros_limpios, motivos = limpiar_datos(registros)
	guardar_datos_limpios(ruta_salida, registros_limpios)

	resumen = {
		"filas_entrada": len(registros),
		"filas_salida": len(registros_limpios),
		"filas_eliminadas": len(registros) - len(registros_limpios),
		"motivos_eliminacion": motivos,
		"ruta_entrada": str(ruta_entrada),
		"ruta_salida": str(ruta_salida),
	}
	return resumen


def construir_rutas_por_defecto() -> tuple[Path, Path]:
	base = Path(__file__).resolve().parent
	ruta_entrada = base / "data" / "raw" / "02_loan_data.csv"
	ruta_salida = base / "data" / "processed" / "02_loan_data_clean.csv"
	return ruta_entrada, ruta_salida


def parsear_argumentos() -> argparse.Namespace:
	ruta_entrada_defecto, ruta_salida_defecto = construir_rutas_por_defecto()

	parser = argparse.ArgumentParser(
		description="Limpia y transforma el dataset de incumplimiento de prestamos."
	)
	parser.add_argument(
		"--entrada",
		type=Path,
		default=ruta_entrada_defecto,
		help="Ruta del archivo CSV crudo.",
	)
	parser.add_argument(
		"--salida",
		type=Path,
		default=ruta_salida_defecto,
		help="Ruta del archivo CSV limpio.",
	)
	return parser.parse_args()


def main() -> None:
	argumentos = parsear_argumentos()
	resumen = procesar_archivo(argumentos.entrada, argumentos.salida)

	print("Proceso completado")
	print(f"Filas de entrada: {resumen['filas_entrada']}")
	print(f"Filas de salida: {resumen['filas_salida']}")
	print(f"Filas eliminadas: {resumen['filas_eliminadas']}")
	if resumen["motivos_eliminacion"]:
		print("Motivos de eliminacion:")
		for motivo, cantidad in sorted(resumen["motivos_eliminacion"].items()):
			print(f"- {motivo}: {cantidad}")
	print(f"Archivo limpio guardado en: {resumen['ruta_salida']}")


if __name__ == "__main__":
	main()
