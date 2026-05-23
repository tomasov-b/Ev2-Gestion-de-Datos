from __future__ import annotations

"""Sube un CSV limpio a Supabase.

Requiere las variables de entorno `SUPABASE_URL` y `SUPABASE_KEY`.
Uso:
	python script_subir_data.py --entrada data/processed/02_loan_data_clean.csv --tabla loan_data_clean

El script convierte tipos básicos y hace inserciones en lotes.
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Any, Dict, List

from supabase import create_client


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


def transformar_registro(reg: Dict[str, Any]) -> Dict[str, Any]:
	return {
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


def leer_csv(ruta: Path) -> List[Dict[str, Any]]:
	with ruta.open(newline="", encoding="utf-8") as f:
		lector = csv.DictReader(f)
		return [row for row in lector]


def insertar_lotes(supabase, tabla: str, filas: List[Dict[str, Any]], lote: int = 500) -> None:
	total = len(filas)
	for i in range(0, total, lote):
		batch = filas[i : i + lote]
		try:
			resp = supabase.table(tabla).insert(batch).execute()
		except Exception as e:
			print(f"ERROR al insertar lote {i}-{i+len(batch)}: {e}")
			continue

		# supabase-py puede devolver un objeto o dict con 'error'
		if hasattr(resp, "error") and resp.error:
			print(f"Error en respuesta del lote {i}-{i+len(batch)}: {resp.error}")
		elif isinstance(resp, dict) and resp.get("error"):
			print(f"Error en respuesta del lote {i}-{i+len(batch)}: {resp.get('error')}")
		else:
			print(f"Insertado lote {i}-{i+len(batch)} ({len(batch)} filas)")


def main() -> None:
	parser = argparse.ArgumentParser(description="Sube CSV a Supabase")
	parser.add_argument("--entrada", type=Path, required=False, default=Path("data/processed/02_loan_data_clean.csv"))
	parser.add_argument("--tabla", type=str, required=False, default="loan_data_clean")
	parser.add_argument("--batch", type=int, required=False, default=500)
	args = parser.parse_args()

	supabase_url = "https://kvgkwvhijkwfhqdvbztw.supabase.co"
	supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imt2Z2t3dmhpamt3ZmhxZHZienR3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3Nzk1NjgxMDAsImV4cCI6MjA5NTE0NDEwMH0.VE2J6EuP3S_LQgxzyO_duc_JcdeDMRwzP7GUsQXT-oc"

	print(f"Leyendo CSV desde: {args.entrada}")
	registros = leer_csv(args.entrada)
	print(f"Filas leídas: {len(registros)}")

	registros_transformados = [transformar_registro(r) for r in registros]

	supabase = create_client(supabase_url, supabase_key)

	print(f"Subiendo a tabla: {args.tabla} en Supabase (batch={args.batch})")
	insertar_lotes(supabase, args.tabla, registros_transformados, lote=args.batch)

	print("Proceso completado.")


if __name__ == "__main__":
	main()

