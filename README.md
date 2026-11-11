# DermaLens · Classification de lésions cutanées (HAM10000)

Projet de module **Deep Learning Avancé** (Cycle Ingénieur IA & Data Science), **Axe 1 : CNN avancé & Transfer Learning + Grad-CAM**.

Le projet classe 7 types de lésions pigmentées à partir d'images dermatoscopiques du dataset HAM10000. Le modèle est un CNN pré-entraîné (EfficientNetV2 / ConvNeXt) fine-tuné, exporté en ONNX et servi par une API FastAPI. L'interface React affiche des cartes **Grad-CAM** pour l'explicabilité. Le cycle de vie complet est outillé : DVC, MLflow, Docker et GitHub Actions.

> ⚠️ Outil pédagogique. Il ne fournit aucun diagnostic médical.

```
Navigateur ──► Frontend React (nginx :3000) ──/api──► FastAPI (:8000) ──► ONNX Runtime (logits + Grad-CAM)
                                                        ▲
              DVC (data/, models/) ──► MLflow Tracking & Model Registry (:5000)
```

---

## 1. Démarrage rapide (1 commande)

Prérequis : Docker, et les artefacts du modèle dans `models/` (`dvc pull`, ou entraînement, voir §3).

```bash
dvc pull                       # récupère data/ et models/ depuis le remote DVC
docker compose up --build      # (ou docker-compose up --build)
```

| Service | URL |
|---|---|
| Application web | http://localhost:3000 |
| API + Swagger | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |

Sans modèle, l'API démarre en mode **dégradé** : `/health` renvoie `"degraded"` et les routes d'inférence renvoient `503`.

Pour servir le modèle quantifié INT8 : `MODEL_VARIANT=int8 docker compose up`.

---

## 2. Le dataset HAM10000 et les choix anti-fuite

10 015 images dermatoscopiques réparties en 7 classes très déséquilibrées :

| Code | Classe | Maligne |
|---|---|---|
| `nv` | Nævus mélanocytaire (~67 %) | |
| `mel` | Mélanome | ✔ |
| `bkl` | Lésion kératosique bénigne | |
| `bcc` | Carcinome basocellulaire | ✔ |
| `akiec` | Kératose actinique / Bowen | ✔ (pré-maligne) |
| `vasc` | Lésion vasculaire | |
| `df` | Dermatofibrome | |

**Les points méthodologiques à défendre en soutenance :**

1. **Split groupé par `lesion_id`.** Une même lésion peut avoir plusieurs photos. Le script `src/data/prepare.py` utilise `StratifiedGroupKFold` (≈ 70/15/15) et vérifie par une assertion qu'aucune lésion n'est partagée entre train, val et test. C'est testé dans `tests/test_prepare.py`.
2. **Pas de fuite par le prétraitement.** La normalisation utilise les statistiques ImageNet (fixes). L'augmentation n'est appliquée qu'au train.
3. **Métriques adaptées au déséquilibre.** Prédire toujours `nv` donne déjà ~67 % d'accuracy, ce qui la rend trompeuse. On suit donc :
   - la **balanced accuracy** (métrique officielle ISIC 2018), qui sert aussi à sélectionner le modèle ;
   - le **F1 macro** ;
   - le **rappel par classe**, en particulier `recall_mel` ;
   - l'**AUC one-vs-rest** et la matrice de confusion.
4. Le jeu de test officiel ISIC n'est pas public. On évalue donc sur notre propre test groupé, qui n'est touché qu'une seule fois, par `evaluate.py`.

### Téléchargement

```bash
pip install kaggle   # puis placer ~/.kaggle/kaggle.json
make data            # kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p data/raw --unzip
```

---

## 3. Pipeline Deep Learning & MLOps

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
dvc repro            # prepare -> train -> evaluate  (ou : make prepare train evaluate)
```

| Étape DVC | Script | Sorties |
|---|---|---|
| `prepare` | `src/data/prepare.py` | `data/processed/{images,splits.csv,stats.json}` |
| `train` | `src/models/train.py` | `models/best.pt`, `train_metrics.json`, run MLflow |
| `evaluate` | `src/models/evaluate.py` | `model.onnx`, `model_fp16.onnx`, `model_int8.onnx`, `metadata.json`, `test_metrics.json`, version du Model Registry |

`dvc metrics show`, `dvc params diff` et `dvc exp run -S train.loss=ce` permettent de comparer les expériences.

### Stratégie d'entraînement (`params.yaml`)

- **Backbones timm pré-entraînés ImageNet :**
  - `resnet50` sert de baseline ;
  - `efficientnetv2_rw_s` et `convnext_tiny` sont les modèles avancés.
- **Fine-tuning en 2 phases :**
  - Phase 1 (`freeze_epochs`) : la tête seule.
  - Phase 2 : tout le réseau, avec des **LR discriminants** (backbone = `lr × 0.1`), un warmup puis une décroissance cosinus, AdamW et l'AMP.
- **Déséquilibre :** au choix
  - Focal Loss pondérée par classe ;
  - entropie croisée pondérée ;
  - `WeightedRandomSampler`.
  Il ne faut pas cumuler sampler et pondération : ce serait une double correction.
- **Augmentation (train uniquement) :**
  - RandomResizedCrop, flips H/V et rotation 90° (les lésions n'ont pas d'orientation) ;
  - ColorJitter et RandAugment ;
  - MixUp/CutMix (`timm.data.Mixup`), avec des pertes compatibles avec les cibles souples.
- **Early stopping** sur la balanced accuracy de validation.

### Plan d'expériences suggéré (à tracer dans MLflow)

```bash
make baseline                                                   # ResNet50 + CE
python -m src.models.train --set train.loss=weighted_ce         # effet de la pondération
python -m src.models.train --set train.loss=focal               # Focal Loss
python -m src.models.train --set train.mixup_alpha=0 --set train.cutmix_alpha=0   # ablation MixUp/CutMix
python -m src.models.train --set model.name=convnext_tiny       # autre backbone
python -m src.models.train --set train.img_size=224             # effet de la résolution
```

Ces runs fournissent directement le tableau « Baseline vs modèle avancé » du rapport.

### Entraîner sur Kaggle / Colab

```bash
!git clone https://github.com/<vous>/<repo>.git && cd <repo>
!pip install -q -r requirements.txt
# Tracking distant (ex. DagsHub, qui offre MLflow + remote DVC gratuits) :
%env MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow
%env MLFLOW_TRACKING_USERNAME=<user>
%env MLFLOW_TRACKING_PASSWORD=<token>
!dvc pull && dvc repro && dvc push
```

Sans serveur distant, MLflow écrit dans `./mlruns`. Il suffit alors de télécharger ce dossier et de lancer `mlflow ui`.

### MLflow

Chaque run journalise :
- **les paramètres** : `params.yaml` aplati ;
- **les métriques par époque** : loss, balanced accuracy, F1, rappels par classe, LR ;
- **les artefacts** :
  - les matrices de confusion val et test et le rapport de classification ;
  - la répartition des classes ;
  - le checkpoint ;
  - les modèles ONNX et le benchmark de latence (taille, p50/p95 et accuracy de chaque variante).

Le modèle est enregistré dans le **Model Registry** (`ham10000-classifier`) :
- l'alias `staging` est posé à chaque nouvelle version ;
- l'alias `production` n'est posé que si la balanced accuracy de test bat celle de la version en production. On peut aussi le forcer avec `--promote`.

Les aliases remplacent les « stages », dépréciés depuis MLflow 2.9.

### ONNX & Grad-CAM exportable

ONNX ne contient que le forward. Le Grad-CAM est donc calculé **analytiquement dans le graphe** (`src/utils/gradcam.py`). Pour une tête `pool moyen → [LayerNorm] → Linear`, les poids Grad-CAM se calculent sans autograd :
- tête linéaire : $\alpha_k \propto \partial y_c/\partial p_k = W_{c,k}$ ;
- tête LayerNorm + linéaire (ConvNeXt) : la même dérivée, passée par la LayerNorm (formule en forme fermée).

Le modèle ONNX a donc deux sorties, `logits` et `cam`, et le serveur n'a besoin ni de PyTorch ni de gradients. `tests/test_model.py` vérifie l'**égalité numérique avec Grad-CAM par autograd** pour ResNet, EfficientNet et ConvNeXt.

Variantes exportées :
- **FP32** : écart de logits avec PyTorch vérifié, de l'ordre de 1e-7 ;
- **FP16** : pour GPU ;
- **INT8** : quantification statique QDQ par canal, calibrée sur la validation. Elle ne porte que sur Conv/Gemm : la branche Grad-CAM reste en FP32.

---

## 4. API FastAPI

| Méthode | Route | Description |
|---|---|---|
| GET | `/health` | état, GPU, version et variante du modèle |
| GET | `/model/metadata` | architecture, classes, métriques val/test, benchmark ONNX |
| POST | `/predict` | `UploadFile` → probabilités des 7 classes (triées), alerte de risque, latences |
| POST | `/predict/visualize` | PNG annoté par Grad-CAM (`?format=png`, par défaut) ou JSON avec image, heatmap et overlay en base64 (`?format=json&alpha=0.45`) |
| GET | `/monitoring/stats` | latence moyenne/p95, confiance moyenne, part des prédictions peu confiantes, distribution des classes prédites (signal de dérive) |

Points de conception :
- les réponses sont validées par des **schémas Pydantic stricts** (`extra="forbid"`) ;
- l'inférence s'exécute dans un threadpool pour ne pas bloquer la boucle asyncio ;
- le prétraitement reproduit exactement celui de la validation : resize sur le côté court, center-crop, normalisation ImageNet.

Codes d'erreur :

| Code | Cas |
|---|---|
| `400` | image invalide ou trop petite, mauvais type MIME, fichier vide |
| `413` | fichier de plus de 10 Mo |
| `422` | champ `file` manquant |
| `503` | modèle non chargé |

```bash
curl -F file=@lesion.jpg http://localhost:8000/predict
curl -F file=@lesion.jpg "http://localhost:8000/predict/visualize" -o gradcam.png
```

---

## 5. Frontend React

Stack : Vite, React 18 et Tailwind CSS v4.
- **3 modes d'entrée :**
  - glisser-déposer ;
  - galerie d'exemples du jeu de test (`make samples`), avec affichage de la vérité terrain ;
  - **webcam**, en capture unique ou en **inférence continue**.
- **Résultats :**
  - **barres de probabilités animées**, rouges pour les classes malignes ;
  - bandeau d'alerte si la classe prédite est maligne ;
  - **visualiseur Grad-CAM** avec curseur d'opacité et mode comparaison.
- **Latence :** client (aller-retour), serveur total, inférence ONNX et prétraitement, avec un historique des dernières requêtes.

```bash
cd frontend && npm install && npm run dev     # http://localhost:5173 (proxy /api -> :8000)
```

---

## 6. Tests & CI/CD

```bash
make test     # pytest tests/ (ML) + backend/tests (API)
make lint     # black, flake8, eslint
```

- `.github/workflows/ci-tests.yml` :
  - black et flake8 ;
  - pytest pour le backend (mini-modèle ONNX réel construit dans les fixtures, pas de poids nécessaires) ;
  - pytest pour le ML (split sans fuite, Focal Loss, Grad-CAM exportable égal à l'autograd) ;
  - ESLint et build Vite.
- `.github/workflows/docker-build.yml` :
  - build des images multi-stage backend et frontend, avec cache GHA ;
  - push sur GHCR depuis `main` ou un tag ;
  - smoke test `docker compose` : l'API démarre, `/health` répond et `/predict` renvoie bien 503 sans modèle.

---

## 7. Arborescence

```
.github/workflows/   ci-tests.yml, docker-build.yml
data/                raw/ + processed/   (DVC)
src/
  config.py          classes, constantes, chargement params
  data/              prepare.py (split par lésion), dataset.py, export_samples.py
  models/            architecture.py, losses.py, train.py, evaluate.py
  utils/             gradcam.py, metrics.py
tests/               tests du code ML
backend/             FastAPI : app/{main,schemas,inference,visualization,config}.py, tests/, Dockerfile
frontend/            React : src/{components,services/api.js,App.jsx}, Dockerfile, nginx.conf
models/              artefacts (DVC) : best.pt, model*.onnx, metadata.json
dvc.yaml, params.yaml, docker-compose.yml, Makefile
```

## 8. Répartition des rôles (trinôme)

| Rôle | Responsabilités principales |
|---|---|
| Lead Deep Learning | `src/data`, `src/models`, ablations, Grad-CAM, quantification, analyse des erreurs |
| Lead MLOps / DevOps | DVC + remote, MLflow + Registry, Dockerfiles, docker-compose, GitHub Actions, monitoring |
| Lead Full-Stack | `backend/` (API, schémas, tests), `frontend/` (UX, webcam, Grad-CAM, latence) |

## 9. Checklist soutenance (démo live)

- [ ] `docker compose up --build` sur une machine propre
- [ ] Inférence par upload, galerie et webcam, avec Grad-CAM et latences affichées
- [ ] Swagger `/docs` : montrer un 400 (fichier texte) et le schéma Pydantic
- [ ] MLflow : comparaison baseline vs modèle avancé, artefacts, alias `production`
- [ ] DVC : `dvc dag`, `dvc metrics show`, `dvc params diff`
- [ ] GitHub Actions au vert
- [ ] Savoir expliquer :
  - le split par `lesion_id` ;
  - le choix de la balanced accuracy ;
  - la Focal Loss ;
  - le Grad-CAM analytique ;
  - le compromis FP32/INT8.

## Références

- Tschandl P., Rosendahl C., Kittler H. *The HAM10000 dataset*. Sci. Data 5, 180161 (2018).
- Codella N. et al. *Skin Lesion Analysis Toward Melanoma Detection 2018 (ISIC)*. arXiv:1902.03368.
- Selvaraju R. et al. *Grad-CAM*. ICCV 2017. Lin T.-Y. et al. *Focal Loss*. ICCV 2017.
