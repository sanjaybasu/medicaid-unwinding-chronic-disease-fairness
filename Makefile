PYTHON := python3
RSCRIPT := Rscript
CODE := code

.PHONY: env data analyze figures tables manuscript test clean

env:
	$(PYTHON) -m venv .venv
	. .venv/bin/activate && pip install -r env/requirements.txt
	$(RSCRIPT) env/R_setup.R

data:
	$(PYTHON) $(CODE)/data_kff_unwinding.py
	$(PYTHON) $(CODE)/data_cdc_wonder.py
	$(PYTHON) $(CODE)/data_hcupnet.py
	$(PYTHON) $(CODE)/data_brfss.py
	$(PYTHON) $(CODE)/data_meps_hc.py
	$(PYTHON) $(CODE)/data_state_covariates.py
	$(PYTHON) $(CODE)/data_models.py

analyze: aim1 aim2 aim3

aim1:
	$(PYTHON) $(CODE)/aim1_did_python.py
	$(RSCRIPT) $(CODE)/aim1_did_R.R

aim2:
	$(PYTHON) $(CODE)/aim2_fairness_audit.py
	$(PYTHON) $(CODE)/aim2_microsim.py

aim3:
	$(PYTHON) $(CODE)/aim3_synthesis.py

figures:
	$(PYTHON) $(CODE)/figures.py

tables:
	$(PYTHON) $(CODE)/tables.py

manuscript: figures tables
	@echo "Manuscript artifacts in notebooks/medicaid-unwinding-chronic-disease-fairness/"

test:
	$(PYTHON) -m pytest tests/ -v

clean:
	rm -rf data/clean/*
	rm -rf figures/*
	rm -rf tables/*
