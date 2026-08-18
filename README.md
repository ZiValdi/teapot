# Thermochemical TEA Dashboard MVP

Small Streamlit dashboard for pyrolysis / thermochemical process-data exploration.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Parameters

The app expects these CSV columns:

- `Cellulose(wt%)`
- `Hemicellulose(wt%)`
- `Lignin(wt%)`
- `Pyrolysis temperature (°C)`
- `HeatingRate(°C/min)`
- `N2 flow rate (mL/min)`
- `ParticleSize(mm)`
- `ParticleSize(μm)`
- `bio-liquid yield(wt%)`

Internally, particle size is normalized to micrometers as `particle_size_um`.

## Economics

The dashboard includes a preliminary TEA layer with editable euro-denominated assumptions, CAPEX/OPEX estimates, NPV, IRR, payback, minimum selling price, cash flow, and a lightweight Monte Carlo uncertainty/sensitivity panel inspired by BioSTEAM-style TEA workflows.
