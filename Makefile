PYTHON := python
DATA_OUT := backend/data/processed/master_town_data.parquet

.PHONY: help data cluster backend frontend dev clean

help: ## Show this help.
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "};{printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

data: ## Build the master dataset 
	cd backend && $(PYTHON) -m src.data_pipeline --output ../../$(DATA_OUT)

cluster: data ## Run KMeans clustering to add labels 
	cd backend && $(PYTHON) -m src.clustering --input ../../$(DATA_OUT) --output ../../$(DATA_OUT)

backend: ## Start Flask API dev mode.
	cd backend && FLASK_APP=app.py FLASK_ENV=development flask run

frontend: ## Start React dev server.
	cd frontend && npm start

dev: ## reminder to run backend & frontend in separate terminals.
	@echo "Use two terminals: 'make backend' and 'make frontend'" 

clean: ## Remove processed data and model artifacts.
	rm -rf backend/data/processed/* 