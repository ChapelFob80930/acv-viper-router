# ACV Viper Router and International Floorplan

Demo concept: a Viper drive-through scan plus service history decides what happens to a used car, and export-bound cars continue into AI-assisted dealer floorplan financing for Canada and Mexico.

**All data is synthetic. Prices, fees, import rules, thresholds and financing terms are illustrative placeholders, not legal, credit or import advice.**

## Live site

One page with three views (routing diagram, 3D journey, dealer floorplan flow): the `docs/` folder, served by GitHub Pages.

## What is in here

| Part | What it does |
| --- | --- |
| `core/rules.py` | Layer 1: hard gates (frame, flood, salvage title, airbag, odometer) run before any model |
| `core/model.py`, `train.py` | Layer 2: LightGBM net value per route + P(sell), SHAP, quantile-based confidence score |
| `core/llm.py` | Layer 3: LLM parses messy service records and explains decisions. It never decides. Uses Claude if `ANTHROPIC_API_KEY` is set, otherwise an offline parser and templates |
| `core/pipeline.py` | Orchestrates the layers. Human validation if confidence < 55% or the top two routes are within 10% |
| `retrain.py` | Automated learning loop: new outcomes, drift check, candidate model, promoted only if better |
| `core/floorplan.py`, `core/floorplan_demo.py`, `build_floorplan.py` | Dealer floorplan: import eligibility (rules), landed cost, vehicle resale / recovery / days-to-sell models, dealer probability-of-default model (LightGBM + SHAP), credit-line rule, transaction decision, synthetic datasets in `data/` |
| `app.py`, `static/` | FastAPI demo with a live decision UI |
| `presentation.html`, `journey3d.fragment.html`, `finance.fragment.html` | Page sources for the diagram, 3D scene and floorplan flowchart |
| `build_site.py` | Builds the static site into `docs/` |

## Run locally

```
python -m venv venv
venv\Scripts\pip install -r requirements.txt     # Windows; use venv/bin/pip elsewhere
venv\Scripts\python train.py                      # train the routing models (writes models/)
venv\Scripts\python build_floorplan.py            # rebuild floorplan datasets and demo JSON
venv\Scripts\python -m uvicorn app:app --port 8000
```

Then open http://localhost:8000 (live demo) or rebuild the static site with `python build_site.py`.

Optional automatic retraining while the server runs: set `VIPER_AUTO_RETRAIN_MIN=60`.

## Caveats

- Fleet-level results come from the simulator's own ground truth. They show the pipeline works, not real-world performance.
- Real training needs counterfactual handling: historical outcomes only show the route actually taken.
- Dealer default labels, import rules, uplifts and advance rates are assumptions to be replaced with real ACV data and legal review.
- The routing demo vehicles are scored by the real models. The floorplan examples are computed by `build_floorplan.py` and embedded in the 3D scene.
