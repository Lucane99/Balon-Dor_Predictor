# Ballon d'Or Predictor

A machine learning project that estimates how a player's season would rank in the Ballon d'Or vote, with an interactive Streamlit app.

**Live app:** _coming soon_

## What it does

- **The Simulator:** edit the stats of real contenders and see how the model re-ranks them.
- **Custom Player Creator:** build your own season (goals, assists, shots, defensive stats, silverware) and see where it would have placed in each year from 2018 to 2024.
- **The Data Story:** what the data says about silverware, positions and how far the model can be trusted.

## How it works

- **Data:** league stats for top-5 European leagues (FBref, via Kaggle) merged with Ballon d'Or rankings (Kaggle + Wikipedia). Trophy data (World Cup, Euro/Copa América, Champions League, league titles) was scraped from Wikipedia. 172 nominees across six award years (2018, 2019, 2021-2024; there was no 2020 award).
- **Features:** goals, assists, shots, expected goals and over-performance, goal contributions per 90, defensive stats, position, and a weighted `trophy_score`.
- **Model:** Random Forest regressor (200 trees, max depth 6) predicting a nominee's final rank.
- **Validation:** leave-one-year-out. Each year is scored by a model that never saw it, so the results below are honest, not memorised.

## Results

| Measure | Result |
|---|---|
| Real winners placed in the model's top 3 | **4 of 6** (random picking would get about 0.6) |
| Average miss | **6.1 ranks** vs 7.5 for always guessing the middle |
| Ranking skill (Spearman, per year) | **0.50** |

## What I found

- Silverware matters, but it does not decide the vote on its own. Many nominees with big trophy hauls finished far down the list.
- Defenders and goalkeepers make up 23% of nominees and have won none.
- Adding a club-prestige feature gave no measurable gain once trophies were included.

## Limitations

- It is a contender finder, not an exact rank predictor. The biggest miss was Rodri (2024), placed 21st.
- It only sees season stats and trophies, not narrative, big-match moments or voting campaigns.
- Only 172 rows from six seasons, and top-5 league stats only.

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit app |
| `Balondor_Analysis.ipynb` | Data cleaning, modelling and validation |
| `ballondor_predictor_model.pkl`, `features_list.pkl` | Trained model and its feature list |
| `ballondor_history.csv` | Out-of-fold scores for past nominees, used by the app |
| `ballon_dor_fbref_dataset_v2.csv` | Merged dataset used in the notebook |

## Data sources

Player stats from FBref (via Kaggle); Ballon d'Or rankings from Kaggle and Wikipedia; trophy history from Wikipedia. This project is not affiliated with FBref, France Football or the Ballon d'Or.