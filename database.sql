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

