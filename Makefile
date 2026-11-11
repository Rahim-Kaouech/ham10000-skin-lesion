.PHONY: help data prepare train baseline evaluate pipeline samples test lint format up down logs mlflow

help:            ## Affiche cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

data:            ## Télécharge HAM10000 depuis Kaggle (nécessite ~/.kaggle/kaggle.json)
	kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p data/raw --unzip

prepare:         ## Split train/val/test groupé par lesion_id
	python -m src.data.prepare

baseline:        ## Baseline ResNet50 + entropie croisée simple, sans MixUp/CutMix
	python -m src.models.train --run-name baseline-resnet50 --set model.name=resnet50 \
	  --set train.loss=ce --set train.mixup_alpha=0 --set train.cutmix_alpha=0 --set train.randaugment=false

train:           ## Modèle avancé (params.yaml)
	python -m src.models.train

evaluate:        ## Test + export ONNX FP32/FP16/INT8 + Model Registry
	python -m src.models.evaluate

pipeline:        ## Pipeline DVC complet (prepare -> train -> evaluate)
	dvc repro

samples:         ## Copie des images de test dans la galerie du frontend
	python -m src.data.export_samples --per-class 2

test:            ## Tous les tests
	pytest tests -v
	cd backend && pytest -v

lint:            ## Linting Python + JS
	black --check src backend tests
	flake8 src backend tests
	cd frontend && npm run lint

format:          ## Formatage automatique
	black src backend tests

up:              ## Lance toute l'application
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f backend

mlflow:          ## UI MLflow locale sur ./mlruns (sans Docker)
	mlflow ui --port 5000
