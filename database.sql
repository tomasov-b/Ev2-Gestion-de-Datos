-- esquema SQL para crear tabla de datos limpios de préstamos

DROP TABLE IF EXISTS loan_data_clean;

CREATE TABLE loan_data_clean (
	id SERIAL PRIMARY KEY,
	person_age INT NOT NULL CHECK (person_age BETWEEN 18 AND 100),
	person_gender TEXT NOT NULL,
	person_education TEXT NOT NULL,
	person_income BIGINT NOT NULL CHECK (person_income > 0),
	person_emp_exp INT NOT NULL CHECK (person_emp_exp >= 0),
	person_home_ownership TEXT NOT NULL,
	loan_amnt BIGINT NOT NULL CHECK (loan_amnt > 0),
	loan_intent TEXT NOT NULL,
	loan_int_rate NUMERIC(6,2) NOT NULL CHECK (loan_int_rate >= 0 AND loan_int_rate <= 100),
	loan_percent_income NUMERIC(8,4) NOT NULL CHECK (loan_percent_income >= 0 AND loan_percent_income <= 1),
	cb_person_cred_hist_length INT NOT NULL CHECK (cb_person_cred_hist_length >= 0),
	credit_score INT NOT NULL CHECK (credit_score BETWEEN 300 AND 850),
	previous_loan_defaults_on_file BOOLEAN NOT NULL,
	loan_status SMALLINT NOT NULL CHECK (loan_status IN (0,1)),
	created_at TIMESTAMPTZ DEFAULT now()
);

-- Index para mejorar el rendimiento de consultas
CREATE INDEX IF NOT EXISTS idx_loan_data_clean_credit_score ON loan_data_clean (credit_score);
CREATE INDEX IF NOT EXISTS idx_loan_data_clean_status ON loan_data_clean (loan_status);

-- Tablas de trazabilidad del modelo para BI y auditoria
CREATE TABLE IF NOT EXISTS loan_model_runs (
	id BIGSERIAL PRIMARY KEY,
	run_name TEXT NOT NULL,
	source TEXT NOT NULL,
	threshold NUMERIC(5,4) NOT NULL,
	train_rows INT NOT NULL,
	test_rows INT NOT NULL,
	auc NUMERIC(8,6) NOT NULL,
	gini NUMERIC(8,6) NOT NULL,
	accuracy NUMERIC(8,6),
	precision NUMERIC(8,6),
	recall NUMERIC(8,6),
	f1 NUMERIC(8,6),
	tn INT,
	fp INT,
	fn INT,
	tp INT,
	training_seconds NUMERIC(12,4),
	memory_rss_mb NUMERIC(12,4),
	created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS loan_model_feature_importance (
	id BIGSERIAL PRIMARY KEY,
	run_name TEXT NOT NULL,
	feature_name TEXT NOT NULL,
	coefficient NUMERIC(18,8) NOT NULL,
	abs_coefficient NUMERIC(18,8) NOT NULL,
	feature_rank INT NOT NULL,
	created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_loan_model_runs_created_at ON loan_model_runs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_loan_model_feature_importance_run ON loan_model_feature_importance (run_name, feature_rank);

CREATE TABLE IF NOT EXISTS loan_model_predictions (
	id BIGSERIAL PRIMARY KEY,
	run_name TEXT NOT NULL,
	source_record_hash TEXT NOT NULL,
	person_age INT,
	person_gender TEXT,
	age_group TEXT,
	prediction_score NUMERIC(8,6) NOT NULL,
	prediction_label SMALLINT NOT NULL CHECK (prediction_label IN (0,1)),
	loan_status_actual SMALLINT CHECK (loan_status_actual IN (0,1)),
	created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS loan_model_drift_events (
	id BIGSERIAL PRIMARY KEY,
	run_name TEXT NOT NULL,
	metric_name TEXT NOT NULL,
	metric_value NUMERIC(18,8) NOT NULL,
	baseline_value NUMERIC(18,8),
	delta NUMERIC(18,8),
	alert_level TEXT NOT NULL,
	created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_loan_model_predictions_run ON loan_model_predictions (run_name, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_loan_model_drift_events_run ON loan_model_drift_events (run_name, created_at DESC);

CREATE OR REPLACE VIEW vw_loan_model_latest_run AS
SELECT *
FROM loan_model_runs
ORDER BY created_at DESC
LIMIT 1;

CREATE OR REPLACE VIEW vw_loan_model_feature_importance_latest AS
SELECT f.*
FROM loan_model_feature_importance f
JOIN vw_loan_model_latest_run r
	ON r.run_name = f.run_name
ORDER BY f.feature_rank;

CREATE OR REPLACE VIEW vw_loan_model_confusion_matrix AS
SELECT
	run_name,
	tn,
	fp,
	fn,
	tp,
	CASE
		WHEN (tp + fn) > 0 THEN ROUND((tp::NUMERIC / (tp + fn))::NUMERIC, 6)
		ELSE NULL
	END AS recall_default_class,
	CASE
		WHEN (fp + tn) > 0 THEN ROUND((tn::NUMERIC / (fp + tn))::NUMERIC, 6)
		ELSE NULL
	END AS specificity
FROM loan_model_runs;

CREATE OR REPLACE VIEW vw_loan_model_dashboard AS
SELECT
	run_name,
	source,
	threshold,
	train_rows,
	test_rows,
	auc,
	gini,
	accuracy,
	precision,
	recall,
	f1,
	tn,
	fp,
	fn,
	tp,
	training_seconds,
	memory_rss_mb,
	created_at
FROM loan_model_runs
ORDER BY created_at DESC;

CREATE OR REPLACE VIEW vw_loan_model_fairness_by_gender AS
SELECT
	person_gender,
	AVG(CASE WHEN prediction_label = 0 THEN 1.0 ELSE 0.0 END) AS approval_rate,
	AVG(CASE WHEN prediction_label = 1 THEN 1.0 ELSE 0.0 END) AS rejection_rate,
	COUNT(*) AS total_predictions
FROM loan_model_predictions
GROUP BY person_gender;

CREATE OR REPLACE VIEW vw_loan_model_fairness_by_age_group AS
SELECT
	age_group,
	AVG(CASE WHEN prediction_label = 0 THEN 1.0 ELSE 0.0 END) AS approval_rate,
	AVG(CASE WHEN prediction_label = 1 THEN 1.0 ELSE 0.0 END) AS rejection_rate,
	COUNT(*) AS total_predictions
FROM loan_model_predictions
GROUP BY age_group;

CREATE OR REPLACE VIEW vw_loan_model_prediction_summary AS
SELECT
	run_name,
	COUNT(*) AS total_predictions,
	AVG(prediction_score) AS avg_prediction_score,
	AVG(CASE WHEN prediction_label = 1 THEN 1.0 ELSE 0.0 END) AS predicted_default_rate,
	SUM(CASE WHEN loan_status_actual = 1 THEN 1 ELSE 0 END) AS actual_defaults,
	SUM(CASE WHEN loan_status_actual = 0 THEN 1 ELSE 0 END) AS actual_non_defaults
FROM loan_model_predictions
GROUP BY run_name;

-- RLS: BI puede leer, pero la escritura queda restringida al servicio con permisos elevados
ALTER TABLE loan_data_clean ENABLE ROW LEVEL SECURITY;
ALTER TABLE loan_model_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE loan_model_feature_importance ENABLE ROW LEVEL SECURITY;
ALTER TABLE loan_model_predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE loan_model_drift_events ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS loan_data_clean_read_all ON loan_data_clean;
CREATE POLICY loan_data_clean_read_all ON loan_data_clean
	FOR SELECT
	USING (true);

DROP POLICY IF EXISTS loan_model_runs_read_all ON loan_model_runs;
CREATE POLICY loan_model_runs_read_all ON loan_model_runs
	FOR SELECT
	USING (true);

DROP POLICY IF EXISTS loan_model_feature_importance_read_all ON loan_model_feature_importance;
CREATE POLICY loan_model_feature_importance_read_all ON loan_model_feature_importance
	FOR SELECT
	USING (true);

DROP POLICY IF EXISTS loan_model_predictions_read_all ON loan_model_predictions;
CREATE POLICY loan_model_predictions_read_all ON loan_model_predictions
	FOR SELECT
	USING (true);

DROP POLICY IF EXISTS loan_model_drift_events_read_all ON loan_model_drift_events;
CREATE POLICY loan_model_drift_events_read_all ON loan_model_drift_events
	FOR SELECT
	USING (true);

