# Realized Volatility Timing

Projet de timing de volatilité réalisée pour une stratégie de carry options.

Le projet estime une variance latente avec un modèle de Heston filtré par Unscented Kalman Filter (UKF), construit un spread entre volatilité implicite et volatilité estimée, puis utilise ce spread pour ajuster dynamiquement l'exposition d'une stratégie short volatility.

Le code d'exécution est autonome : les briques utiles du cours ont été copiées dans `investment_lab/`. Le dossier `prof_cours/` sert seulement de référence locale.

## Contenu du dépôt

### Code principal

- `realized_vol_timing/heston_ukf.py` : modèle de Heston, UKF, log-vraisemblance, calibration rolling
- `realized_vol_timing/data.py` : construction du panel marché à partir des données options
- `realized_vol_timing/signals.py` : spread et allocation dynamique
- `realized_vol_timing/benchmarks.py` : benchmarks `RV_21d` et `RV_63d`
- `realized_vol_timing/strategy.py` : allocation des trades et backtest
- `realized_vol_timing/experiments.py` : orchestration des expériences SPY / AAPL
- `realized_vol_timing/reporting.py` : export des figures et tables

### Dossiers utiles

- `investment_lab/` : copie locale des outils du cours
- `notebooks/` : notebook de rendu
- `scripts/` : scripts d'exécution
- `tests/` : tests
- `data/` : données locales
- `outputs/` : résultats exportés

## Notebook principal

Le notebook à rendre est :

- `notebooks/Project_Report.ipynb`

Il recalcule les expériences depuis les données brutes et contient :

- la formulation du modèle de Heston en espace d'état
- l'estimation UKF
- la calibration rolling par maximum de vraisemblance
- le benchmark de volatilité réalisée `RV_21d`
- une étude courte pour montrer la démarche
- une étude full sample sur `SPY`
- une étude full sample sur `AAPL`

## Données attendues

Le projet utilise les fichiers présents dans `data/`. Les fichiers utilisés actuellement sont :

- `data/spy_2020_2022.parquet`
- `data/aapl_2016_2023.parquet`
- `data/optiondb_2016_2023.parquet`
- `data/par-yield-curve-rates-2020-2023.csv`

Tickers pris en charge dans le pipeline réel :

- `SPY`
- `AAPL`

## Installation

Depuis la racine du projet :

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[notebook,dev]"
```
Sans les extra

```bash
python -m pip install -e .
```

## Lancement

### Démo synthétique

```bash
python scripts/run_synthetic_demo.py
```

### Expérience réelle

Run rapide :

```bash
python scripts/run_dynamic_carry.py --ticker SPY
python scripts/run_dynamic_carry.py --ticker AAPL
```

Run full sample :

```bash
python scripts/run_dynamic_carry.py --ticker SPY --full-sample
python scripts/run_dynamic_carry.py --ticker AAPL --full-sample
```

Run sur plage personnalisée :

```bash
python scripts/run_dynamic_carry.py --ticker SPY --start-date 2020-01-02 --end-date 2020-09-30
```

Compatibilité avec l'ancien script :

```bash
python scripts/run_spy_dynamic_carry.py --full-sample
```

## Sorties

Les runs exportent leurs résultats dans `outputs/`, principalement dans :

- `outputs/dynamic_carry/`

Chaque run crée un sous-dossier avec le ticker et les dates. On y trouve en général :

- `figures/`
- `tables/`
- `run_summary.txt`
- `run_metadata.json`

## Tests

```bash
pytest -q
```

## Remarques

- Les runs full sample sont plus longs que le sample court, car les paramètres de Heston sont recalibrés en rolling.
- Sur certaines machines, un run long peut prendre plusieurs dizaines de minutes.
- Le backtest utilise une stratégie short 1-week strangle avec allocation dynamique.
- Le signal ne retourne pas la position : l'allocation varie entre une exposition nulle et une exposition renforcée, mais ne passe pas long volatility.
