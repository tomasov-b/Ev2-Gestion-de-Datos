# Caso 4: Predicción de Incumplimiento de Préstamos

## Objetivo
Preparar el dataset crudo de prestamos para analisis, modelado y una posible carga posterior en PostgreSQL / Supabase.

## Archivos principales
- `scriptLimpieza.py`: limpia y normaliza el dataset crudo (`data/raw/02_loan_data.csv`) y escribe el CSV limpio en `data/processed/02_loan_data_clean.csv`.
- `script_subir_data.py`: lee el CSV limpio y lo sube por lotes a una tabla PostgreSQL/Supabase (`loan_data_clean`).
- `train_loan_default_model.py`: entrena y evalua una regresion logistica para prediccion de incumplimiento, genera ROC, matriz de confusion, Gini, importancias y salidas para BI.
- `predict_loan_default.py`: aplica el modelo entrenado, genera scoring, calcula drift y registra predicciones/eventos en Supabase.
- `database.sql`: script SQL para crear la tabla `loan_data_clean` (tipos, restricciones y índices) antes de la carga.
- `Dockerfile`: contenedor para ejecutar el pipeline de entrenamiento de forma reproducible.
- `requirements.txt`: dependencias para ejecutar los scripts (`supabase`, opcionalmente `python-dotenv`).
- `data/raw/02_loan_data.csv`: entrada cruda.
- `data/processed/02_loan_data_clean.csv`: salida limpia generada por `scriptLimpieza.py`.
- `contexto.txt`: contexto funcional y definicion de variables.

## Qué hacen los scripts

- `scriptLimpieza.py`:
	- Lee el CSV crudo en `data/raw/02_loan_data.csv`.
	- Normaliza tipos numéricos y categorías, corrige escalados (por ejemplo tasas o porcentajes) y redondea enteros.
	- Valida reglas (edad entre 18-100, credit_score 300-850, proporciones entre 0 y 1, etc.).
	- Elimina registros inválidos y duplicados exactos después de la normalización.
	- Exporta el CSV limpio a `data/processed/02_loan_data_clean.csv`.

- `script_subir_data.py`:
	- Lee `data/processed/02_loan_data_clean.csv`.
	- Convierte los campos a los tipos adecuados (ints, floats, booleanos).
	- Valida la estructura del CSV antes de insertar.
	- Inserta las filas en la tabla `loan_data_clean` en Supabase/Postgres en lotes (por defecto 500 filas).
	- Usa `SUPABASE_URL` y `SUPABASE_SERVICE_ROLE_KEY` para escritura; la `SUPABASE_KEY` anónima no sirve cuando RLS está activa.
	- Permite seudonimizar columnas antes de la carga con `--hash-columns`.

- `train_loan_default_model.py`:
	- Lee el dataset limpio desde archivo local o desde Supabase.
	- Aplica `StandardScaler` a numéricas y `OneHotEncoder` a categóricas.
	- Usa `class_weight='balanced'` y permite `SMOTE` si se quiere sobremuestreo.
	- Divide en entrenamiento/prueba con estratificación.
	- Genera análisis univariado, boxplots, correlación, matriz de confusión, curva ROC, AUC y Gini.
	- Registra tiempos de ejecución, memoria RAM y métricas en logs y JSON.
	- Exporta importancias de variables, fairness por género y grupos de edad, perfil base para drift y el modelo serializado.

- `predict_loan_default.py`:
	- Carga el modelo entrenado y el perfil base.
	- Genera predicciones y, si hay etiquetas reales, calcula AUC y Gini en producción.
	- Evalua deriva por variables numericas y categoricas frente al perfil base.
	- Guarda predicciones, fairness y drift en archivos locales y, opcionalmente, en Supabase.

## Regla de limpieza principal
- Las columnas originales se mantienen en ingles.
- `person_gender`, `person_education`, `person_home_ownership`, `loan_intent` y `previous_loan_defaults_on_file` se estandarizan a valores canonicos.
- `person_age`, `person_income`, `person_emp_exp`, `loan_amnt`, `cb_person_cred_hist_length`, `credit_score` y `loan_status` se convierten a enteros.
- `loan_int_rate` y `loan_percent_income` se convierten a flotantes y se corrigen heurísticamente si vienen escalados.

## Funciones del script
- `normalizar_texto(valor)`: limpia espacios, convierte el valor a texto y devuelve una cadena vacia si recibe `None`.
- `convertir_flotante(valor)`: intenta transformar el valor a `float`; si no puede hacerlo, devuelve `None`.
- `convertir_entero(valor)`: reutiliza `convertir_flotante()` y redondea el resultado a `int`.
- `normalizar_categoria(valor, mapa)`: estandariza valores categoricos segun un diccionario de equivalencias.
- `normalizar_tasa_interes(valor)`: convierte la tasa a flotante y corrige casos en los que viene escalada en puntos basicos.
- `normalizar_proporcion_ingreso(valor)`: convierte el porcentaje del ingreso a flotante y corrige casos en los que viene expresado como porcentaje.
- `normalizar_registro(registro)`: transforma una fila completa, valida rangos y devuelve el registro limpio o el motivo de rechazo.
- `leer_datos_crudos(ruta_entrada)`: abre el CSV original y carga todas las filas en memoria.
- `limpiar_datos(registros)`: recorre todas las filas, aplica la normalizacion y elimina registros invalidos o duplicados.
- `guardar_datos_limpios(ruta_salida, registros)`: crea la carpeta de salida si hace falta y escribe el CSV limpio.
- `procesar_archivo(ruta_entrada, ruta_salida)`: coordina lectura, limpieza, guardado y construye un resumen del proceso.
- `construir_rutas_por_defecto()`: define las rutas estandar de entrada y salida dentro del proyecto.
- `parsear_argumentos()`: lee los argumentos `--entrada` y `--salida` desde la linea de comandos.
- `main()`: punto de entrada del script; ejecuta todo el flujo y muestra el resumen final.

## Sentencias if del script
- `if valor is None:` en `normalizar_texto()`: evita errores cuando llega un valor nulo.
- `if not texto:` en `convertir_flotante()` y `normalizar_categoria()`: evita convertir cadenas vacias o en blanco.
- `except ValueError:` en `convertir_flotante()`: controla el caso en que el texto no puede convertirse a numero.
- `if numero is None:` en `convertir_entero()`: evita continuar cuando la conversion numerica fallo.
- `if tasa is None:` en `normalizar_tasa_interes()`: corta el proceso si la tasa no es interpretable.
- `if tasa > 100:` en `normalizar_tasa_interes()`: corrige tasas que vienen escaladas como 1602 en lugar de 16.02.
- `if proporcion is None:` en `normalizar_proporcion_ingreso()`: corta el proceso si el valor no es interpretable.
- `if proporcion > 1:` en `normalizar_proporcion_ingreso()`: corrige porcentajes que vienen como 49 en lugar de 0.49.
- `for columna in COLUMNAS_ORDEN:` y `if limpio[columna] is None:` en `normalizar_registro()`: comprueba que ninguna variable esencial quede vacia despues de normalizar.
- `if not 18 <= limpio["person_age"] <= 100:` en `normalizar_registro()`: elimina edades fuera de rango.
- `if limpio["person_emp_exp"] < 0:` en `normalizar_registro()`: elimina experiencias laborales negativas.
- `if limpio["person_emp_exp"] > limpio["person_age"]:` en `normalizar_registro()`: elimina filas incoherentes donde la experiencia supera la edad.
- `if not 300 <= limpio["credit_score"] <= 850:` en `normalizar_registro()`: elimina puntajes crediticios fuera del rango esperado.
- `if limpio["person_income"] <= 0:` en `normalizar_registro()`: elimina ingresos no positivos.
- `if limpio["loan_amnt"] <= 0:` en `normalizar_registro()`: elimina montos de prestamo no positivos.
- `if not 0 <= limpio["loan_int_rate"] <= 100:` en `normalizar_registro()`: valida que la tasa quede en un rango razonable.
- `if not 0 <= limpio["loan_percent_income"] <= 1:` en `normalizar_registro()`: valida que la proporcion prestamo/ingreso quede entre 0 y 1.
- `if limpio["loan_status"] not in {0, 1}:` en `normalizar_registro()`: asegura que el target siga siendo binario.
- `if limpio is None:` en `limpiar_datos()`: cuenta las filas rechazadas y no las agrega al resultado final.
- `if firma in vistos:` en `limpiar_datos()`: elimina duplicados exactos despues de la normalizacion.
- `if resumen["motivos_eliminacion"]:` en `main()`: solo imprime el detalle de motivos si hubo rechazos.

## Ejecución

1) Instalar dependencias

```bash
pip install -r requirements.txt
```

2) Generar el CSV limpio

```bash
python scriptLimpieza.py
```

3) Crear la tabla en PostgreSQL / Supabase

Pega el contenido de database.sql en el editor SQL de Supabase y ejecútalo allí.


4) Subir el CSV a Supabase

```bash
python script_subir_data.py --entrada data/processed/02_loan_data_clean.csv --tabla loan_data_clean
```

5) Entrenar el modelo

```bash
python train_loan_default_model.py --source local --input data/processed/02_loan_data_clean.csv
```

Para leer desde Supabase:

```bash
set SUPABASE_URL=tu_url
set SUPABASE_SERVICE_ROLE_KEY=tu_clave_de_servicio
python train_loan_default_model.py --source supabase --supabase-table loan_data_clean --sync-supabase
```

6) Ejecutar con Docker

```bash
docker build -t loan-risk-model .
docker run --rm -e SUPABASE_URL=tu_url -e SUPABASE_KEY=tu_clave loan-risk-model
```

7) Ejecutar scoring e inferencia

```bash
python predict_loan_default.py --source local --input data/processed/02_loan_data_clean.csv
```

Con Supabase:

```bash
python predict_loan_default.py --source supabase --supabase-table loan_data_clean --sync-supabase
```

## Salidas del entrenamiento

- `artifacts/metrics/model_metrics.json`: AUC, Gini, precisión, recall, F1 y matriz de confusión.
- `artifacts/plots/roc_curve.png`: curva ROC.
- `artifacts/plots/confusion_matrix.png`: matriz de confusión.
- `artifacts/plots/feature_importance.png`: importancia por coeficiente absoluto.
- `artifacts/bi/fairness_by_gender.csv` y `artifacts/bi/fairness_by_age_group.csv`: insumos para auditoría de equidad en BI.
- `artifacts/logs/training.log`: trazabilidad de tiempos, memoria y métricas.
- `artifacts/metrics/baseline_profile.json`: perfil de entrenamiento para monitoreo de drift.
- `artifacts/inference/predictions/predictions.csv`: scoring por registro.
- `artifacts/inference/metrics/drift_metrics.json`: indicadores de deriva por variable.
- `artifacts/logs/inference.log`: trazabilidad del scoring y alertas operativas.

## Seguridad y BI

- Las columnas sensibles pueden seudonimizarse antes de cargar o entrenar usando `--hash-columns` o `--sensitive-columns`.
- `database.sql` habilita RLS en las tablas operativas y de métricas para permitir lectura BI controlada.
- Las escrituras a tablas con RLS deben hacerse con `SUPABASE_SERVICE_ROLE_KEY`; la clave anónima queda para lectura BI o clientes de solo lectura.
- Metabase puede conectarse a PostgreSQL/Supabase y consumir `loan_data_clean`, `loan_model_runs`, `loan_model_feature_importance`, `loan_model_predictions` y las vistas `vw_loan_model_dashboard`, `vw_loan_model_confusion_matrix`, `vw_loan_model_fairness_by_gender`, `vw_loan_model_fairness_by_age_group` y `vw_loan_model_prediction_summary` para construir dashboards de rendimiento, interpretabilidad, equidad y seguimiento operativo.
