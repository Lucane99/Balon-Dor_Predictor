import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page Configuration
st.set_page_config(page_title="Ballon d'Or Predictor", page_icon="Balondor_Predictor", layout="wide")
st.title("Ballon d'Or Machine Learning Predictor")
st.markdown("Predicting football's highest individual honor from league stats and silverware, using a Random Forest trained on the 2018-2024 nominees.")

# Load the pre-trained model and features
@st.cache_resource # This caches the model so it doesn't reload on every click
def load_model():
    model = joblib.load('ballondor_predictor_model.pkl')
    features = joblib.load('features_list.pkl')
    return model, features

rf_model, features_master = load_model()

# Past nominees' honest scores (from the notebook), used to put a score in context
@st.cache_data
def load_history():
    return pd.read_csv('ballondor_history.csv')

try:
    history = load_history()
except FileNotFoundError:
    st.error("ballondor_history.csv not found. Run the export cell in the notebook and put the file next to app.py.")
    st.stop()

past_scores = history['model_score'].to_numpy()
winner_scores = history.loc[history['rank'] == 1, 'model_score']


def percent_beaten(score):
    """Share of 2018-2024 nominees whose honest (out-of-fold) model score is worse than this one."""
    return float((past_scores > score).mean() * 100)

POSITIONS = ["Forward", "Midfielder", "Defender", "Goalkeeper"]

# Typical values per position (from the nominees in the training data); used when a stat is left blank
TYPICAL = {
    "Forward":    {"tackles_won": 9.0,  "interceptions": 6.0,  "clearances": 12.0, "clean_sheets": 0.0},
    "Midfielder": {"tackles_won": 25.0, "interceptions": 21.0, "clearances": 14.0, "clean_sheets": 0.0},
    "Defender":   {"tackles_won": 26.0, "interceptions": 33.5, "clearances": 83.0, "clean_sheets": 0.0},
    "Goalkeeper": {"tackles_won": 0.0,  "interceptions": 1.0,  "clearances": 11.5, "clean_sheets": 9.5},
}

# Expected goals per shot, measured from the nominees (it differs a lot by position)
XG_PER_SHOT = {"Forward": 0.166, "Midfielder": 0.096, "Defender": 0.100, "Goalkeeper": 0.100}

DEFAULT_90S = 29.0   # typical 90s played by a nominee (median 28.8)


def _num(df, col):
    """Column as numbers; a missing column or a blank cell becomes NaN."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def prepare_features(input_df):
    """Turn what the user typed into the 14 features the model was trained on."""
    df = input_df.copy()

    # 1. Position: tidy it up; blank or unknown counts as Forward
    if "position" in df.columns:
        pos = df["position"].astype(str).str.strip().str.title()
    else:
        pos = pd.Series("Forward", index=df.index)
    df["position"] = pos.where(pos.isin(POSITIONS), "Forward")
    for p in POSITIONS:
        df[f"is_{p}"] = (df["position"] == p).astype(float)

    # 2. Numbers the user types: blank counts as 0
    for col in ["goals", "assists", "total_shots", "trophy_score"]:
        df[col] = _num(df, col).fillna(0.0).clip(lower=0)

    # 3. Defensive stats: blank = typical for the position. Clean sheets only exist for keepers.
    for col in ["tackles_won", "interceptions", "clearances", "clean_sheets"]:
        typical = df["position"].map(lambda p: TYPICAL[p][col])
        df[col] = _num(df, col).fillna(typical).clip(lower=0)
    df.loc[df["position"] != "Goalkeeper", "clean_sheets"] = 0.0

    # 4. The three derived features
    nineties = _num(df, "nineties")
    nineties = nineties.where(nineties > 0, DEFAULT_90S)
    df["expected_goals"] = df["total_shots"] * df["position"].map(XG_PER_SHOT)
    df["goal_contributions_p_90"] = (df["goals"] + df["assists"]) / nineties
    df["xg_overperformance"] = df["goals"] - df["expected_goals"]

    return df

# Building the Dashboard Structure
tab1, tab2, tab3 = st.tabs([" The Simulator", "Custom Player Creator", "The Data Story"])

with tab1:
    st.header("The 'What-If' Simulator")
    st.write("Adjust the stats of the contenders (or add new rows) and see how the model re-ranks them.")
    st.caption("Use **league-only** numbers, because that is what the model was trained on. "
               "Leave a cell blank to use a typical value for that position. "
               "The starting numbers are placeholders, so edit them.")

    # Base data for the contenders. Blank (NaN) cells = typical value for the position.
    default_contenders = pd.DataFrame([
        {'player': 'Vinícius Júnior', 'club': 'Real Madrid', 'position': 'Forward', 'goals': 25, 'assists': 15, 'total_shots': 90, 'tackles_won': 20, 'interceptions': np.nan, 'clearances': np.nan, 'clean_sheets': np.nan, 'trophy_score': 25.0, 'nineties': np.nan},
        {'player': 'Kylian Mbappé', 'club': 'Real Madrid', 'position': 'Forward', 'goals': 35, 'assists': 10, 'total_shots': 130, 'tackles_won': 15, 'interceptions': np.nan, 'clearances': np.nan, 'clean_sheets': np.nan, 'trophy_score': 25.0, 'nineties': np.nan},
        {'player': 'Erling Haaland', 'club': 'Manchester City', 'position': 'Forward', 'goals': 40, 'assists': 5, 'total_shots': 140, 'tackles_won': 10, 'interceptions': np.nan, 'clearances': np.nan, 'clean_sheets': np.nan, 'trophy_score': 10.0, 'nineties': np.nan},
        {'player': 'Lamine Yamal', 'club': 'Barcelona', 'position': 'Forward', 'goals': 15, 'assists': 22, 'total_shots': 65, 'tackles_won': 25, 'interceptions': np.nan, 'clearances': np.nan, 'clean_sheets': np.nan, 'trophy_score': 12.0, 'nineties': np.nan},
        {'player': 'Rodri', 'club': 'Manchester City', 'position': 'Midfielder', 'goals': 8, 'assists': 10, 'total_shots': 40, 'tackles_won': 70, 'interceptions': np.nan, 'clearances': np.nan, 'clean_sheets': np.nan, 'trophy_score': 22.0, 'nineties': np.nan},
    ])

    # Displaying the editable dataframe
    edited_df = st.data_editor(
        default_contenders,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "player": st.column_config.TextColumn("Player"),
            "club": st.column_config.TextColumn("Club (display only)"),
            "position": st.column_config.SelectboxColumn(
                "Position",
                options=["Forward", "Midfielder", "Defender", "Goalkeeper"],
                required=True
            ),
            "goals": st.column_config.NumberColumn("League goals", min_value=0),
            "assists": st.column_config.NumberColumn("League assists", min_value=0),
            "total_shots": st.column_config.NumberColumn("Shots", min_value=0),
            "tackles_won": st.column_config.NumberColumn("Tackles won", min_value=0, help="Blank = typical for the position"),
            "interceptions": st.column_config.NumberColumn("Interceptions", min_value=0, help="Blank = typical for the position"),
            "clearances": st.column_config.NumberColumn("Clearances", min_value=0, help="Blank = typical for the position"),
            "clean_sheets": st.column_config.NumberColumn("Clean sheets", min_value=0, help="Goalkeepers only. Ignored for other positions."),
            "trophy_score": st.column_config.NumberColumn(
                "Trophy Score",
                help="World Cup = 20 (final lost 10) | UCL = 15 (final lost 8) | Euro/Copa = 12 (final lost 6) | League Title = 6 to 10",
                min_value=0.0,
                max_value=60.0
            ),
            "nineties": st.column_config.NumberColumn(
                "90s played",
                help="Minutes played divided by 90. Blank = 29, a typical nominee.",
                min_value=1.0,
                max_value=40.0
            ),
        }
    )

    # Prediction Engine
    if st.button("Predict Ballon d'Or", type="primary", key="tab1_predict_button"):
        pool = edited_df.dropna(how="all").reset_index(drop=True)   # ignore completely empty rows

        if pool.empty:
            st.warning("Add at least one player to the table first.")
        else:
            with st.spinner("Calculating full metrics and ranking players..."):
                pool["player"] = pool["player"].fillna("Unnamed player")
                pool["club"] = pool["club"].fillna("")

                # 1. Turn the typed inputs into the 14 model features, 2. predict, 3. rank (lowest score = best)
                X_app = prepare_features(pool)[features_master]
                pool["model_score"] = rf_model.predict(X_app)
                pool["predicted_rank"] = pool["model_score"].rank(method="min").astype(int)

                sorted_df = pool.sort_values("predicted_rank").reset_index(drop=True)

            st.divider()
            st.subheader("Final Ranking")

            # Display winner banner
            winner_name = sorted_df.iloc[0]['player']
            winner_club = sorted_df.iloc[0]['club']
            club_text = f" ({winner_club})" if winner_club else ""
            st.success(f"**Predicted Winner: {winner_name}{club_text}**")

            # Prepare display dataframe
            display_cols = {
                'predicted_rank': 'Rank', 'player': 'Player', 'club': 'Club', 'position': 'Position',
                'goals': 'Goals', 'assists': 'Assists', 'trophy_score': 'Trophy score', 'model_score': 'Model score'
            }
            for col in ['goals', 'assists', 'trophy_score']:
                sorted_df[col] = pd.to_numeric(sorted_df[col], errors='coerce').fillna(0)
            final_display = sorted_df[list(display_cols)].rename(columns=display_cols).set_index('Rank')

            # Highlight ONLY the rank 1 row in gold
            def highlight_top_rank(row):
                if row.name == 1:  # Index 1 corresponds to rank 1
                    return ['background-color: #FFF2B2; color: black; font-weight: bold'] * len(row)
                return [''] * len(row)

            # Display styled table
            st.dataframe(
                final_display.style.apply(highlight_top_rank, axis=1).format(
                    {'Goals': '{:.0f}', 'Assists': '{:.0f}', 'Trophy score': '{:.0f}', 'Model score': '{:.2f}'},
                    na_rep='-'
                ),
                use_container_width=True
            )
            st.caption("Model score is a predicted rank, so lower is better. Scores are pulled toward the middle "
                       "(nobody gets exactly 1.0), so compare players with each other rather than reading the number as a placing.")
            st.balloons()

# Silverware points, same weights the model was trained with
TOURNAMENT_POINTS = {"None": 0, "World Cup winner": 20, "World Cup runner-up": 10,
                     "Euro / Copa América winner": 12, "Euro / Copa América runner-up": 6}
UCL_POINTS = {"None": 0, "Champions League winner": 15, "Champions League runner-up": 8}
LEAGUE_POINTS = {"None / other league": 0, "Premier League champion": 10, "La Liga champion": 10,
                 "Serie A champion": 8, "Bundesliga champion": 8, "Ligue 1 champion": 6}

with tab2:
    st.header("Custom Player Creator")
    st.write("Build a season profile and see how it compares with every Ballon d'Or nominee from 2018 to 2024.")
    st.caption("Use **league-only** stats (top-5 leagues), because that is what the model was trained on.")

    col1, col2 = st.columns(2)

    with col1:
        player_name = st.text_input("Player Name", value="Custom Star", key="tab2_player_name")
        position = st.selectbox("Position", ["Forward", "Midfielder", "Defender", "Goalkeeper"], key="tab2_position")

        st.markdown("**Silverware this season**")
        intl = st.selectbox("International tournament", list(TOURNAMENT_POINTS), key="tab2_intl")
        ucl = st.selectbox("Champions League", list(UCL_POINTS), key="tab2_ucl")
        league = st.selectbox("Domestic league", list(LEAGUE_POINTS), key="tab2_league")
        trophy_score = float(TOURNAMENT_POINTS[intl] + UCL_POINTS[ucl] + LEAGUE_POINTS[league])
        st.metric("Trophy score", f"{trophy_score:.0f}")

    with col2:
        goals = st.slider("League goals", 0, 50, 20, key="tab2_goals")
        assists = st.slider("League assists", 0, 25, 8, key="tab2_assists")
        total_shots = st.slider("Shots", 0, 200, 90, key="tab2_shots")
        nineties = st.slider("90s played", 10, 38, 29, key="tab2_nineties", help="Minutes played divided by 90.")

        st.markdown("**Defensive stats** (leave blank for a typical value for the position)")
        d1, d2 = st.columns(2)
        tackles = d1.number_input("Tackles won", min_value=0, step=1, value=None, placeholder="typical", key="tab2_tackles")
        interceptions = d2.number_input("Interceptions", min_value=0, step=1, value=None, placeholder="typical", key="tab2_interceptions")
        clearances = d1.number_input("Clearances", min_value=0, step=1, value=None, placeholder="typical", key="tab2_clearances")
        clean_sheets = d2.number_input("Clean sheets (goalkeepers)", min_value=0, step=1, value=None, placeholder="typical", key="tab2_clean_sheets")

    st.markdown("<br>", unsafe_allow_html=True)

    # Dedicated button for Tab 2
    if st.button("Predict Custom Player Rank", type="primary", key="tab2_predict_button"):
        # 1. Build initial player dictionary
        input_dict = {
            'player': player_name,
            'position': position,
            'goals': goals,
            'assists': assists,
            'total_shots': total_shots,
            'tackles_won': tackles,
            'interceptions': interceptions,
            'clearances': clearances,
            'clean_sheets': clean_sheets,
            'trophy_score': trophy_score,
            'nineties': nineties,
        }
        
        # 2. Convert to DataFrame and pass through prepare_features (fills defaults, derived metrics)
        custom_df = pd.DataFrame([input_dict])
        processed_custom_df = prepare_features(custom_df)

        # 3. Predict raw rank score using features_master
        score = rf_model.predict(processed_custom_df[features_master])[0]

        # 4. Put the score in context: how many past nominees does it beat? (lower score = better)
        pct = percent_beaten(score)
        median_winner = float(np.median(winner_scores))

        st.divider()
        st.subheader("Evaluation Results")

        m1, m2, m3 = st.columns(3)
        m1.metric(f"Model score for {player_name}", f"{score:.2f}", help="Predicted rank. Lower is better.")
        m2.metric("Better than past nominees", f"{pct:.0f}%", help="Share of the 172 nominees (2018-2024) with a worse score.")
        m3.metric("Median past winner", f"{median_winner:.2f}")

        if pct >= 90:
            st.success(f"**Winner-level season.** {player_name} scores better than {pct:.0f}% of past nominees.")
        elif pct >= 75:
            st.info(f"**Strong contender.** {player_name} scores better than {pct:.0f}% of past nominees.")
        elif pct >= 50:
            st.info(f"**Solid nominee-level season.** {player_name} beats {pct:.0f}% of past nominees, but is not close to winner level.")
        else:
            st.warning(f"**Below a typical nominee.** {player_name} beats only {pct:.0f}% of past nominees.")

        
        # 5. Where would this score have placed in each past year?
        st.markdown("**Where this season would have placed**")
        rows = []
        for year, g in history.groupby("year"):
            place = int((g["model_score"] < score).sum()) + 1
            winner = g.loc[g["rank"] == 1, "player"].iloc[0]
            rows.append({"Year": int(year), "Would have placed": f"#{place} of {len(g) + 1}", "Actual winner": winner})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        
        st.caption("Estimates from the model's honest out-of-fold scores. Real voting also depends on things the model can't see.")

        # 6. Silverware effect: same player, no trophies
        no_trophy_df = prepare_features(pd.DataFrame([{**input_dict, "trophy_score": 0.0}]))
        score_no_trophy = rf_model.predict(no_trophy_df[features_master])[0]
        pct_no_trophy = percent_beaten(score_no_trophy)

        st.markdown("**Silverware effect**")
        s1, s2 = st.columns(2)
        s1.metric("With this season's trophies", f"{score:.2f}", f"beats {pct:.0f}% of nominees", delta_color="off")
        s2.metric("Same stats, no trophies", f"{score_no_trophy:.2f}", f"beats {pct_no_trophy:.0f}% of nominees", delta_color="off")

        # 7. Position note from the history file
        winners_pos = history.loc[(history["rank"] == 1) & (history["position_name"] == position), "player"]
        nominees_pos = int((history["position_name"] == position).sum())
        if len(winners_pos) == 0:
            st.info(f"No {position.lower()} won the Ballon d'Or between 2018 and 2024 ({nominees_pos} {position.lower()} nominees). The model learned that pattern too.")
        else:
            st.info(f"{len(winners_pos)} of the 6 winners from 2018 to 2024 were {position.lower()}s: {', '.join(winners_pos)}.")

import altair as alt

with tab3:
    st.header("The Data Story")
    st.write("What the model learned from the 2018-2024 nominees, and how far to trust it.")

    # Section 1: silverware vs final rank
    plot_df = history.copy()

    # Winners: nudge ties apart slightly so both dots show (e.g. Modric and Benzema, both at 25)
    w = (plot_df[plot_df["rank"] == 1][["player", "year", "club", "trophy_score", "rank"]]
         .sort_values(["trophy_score", "year"]).reset_index(drop=True))
    size = w.groupby("trophy_score")["year"].transform("size")
    pos = w.groupby("trophy_score").cumcount()
    w["x_plot"] = w["trophy_score"] + (pos - (size - 1) / 2) * 0.7

    # Year labels above the dots, spread out so they don't collide
    xs = []
    for x in w["x_plot"]:
        xs.append(x if not xs else max(x, xs[-1] + 2.4))
    w["label_x"] = xs
    w["label_y"] = 1 - 2.5 - 2.5 * (w.index % 2)
    w["label"] = w["year"].astype(int).astype(str)

    yenc = alt.Y("rank:Q", title="Final Ballon d'Or rank (1 = winner)",
                 scale=alt.Scale(reverse=True, domain=[-7, 52]),
                 axis=alt.Axis(values=[1, 5, 10, 15, 20, 25, 30, 40, 50]))
    tip = ["player", "year", "club", "rank", "trophy_score"]

    others = (alt.Chart(plot_df[plot_df["rank"] > 1])
              .mark_circle(size=55, opacity=0.4, color="#9aa0a6")
              .encode(x=alt.X("trophy_score:Q", title="Trophy score"), y=yenc, tooltip=tip))
    winners = (alt.Chart(w)
               .mark_point(size=180, filled=True, color="#e8453c", stroke="white", strokeWidth=1.5, opacity=0.95)
               .encode(x="x_plot:Q", y=yenc, tooltip=tip))
    lines = alt.Chart(w).mark_rule(color="#e8453c", opacity=0.5).encode(
        x="x_plot:Q", y="rank:Q", x2="label_x:Q", y2="label_y:Q")
    labels = alt.Chart(w).mark_text(fontSize=12, fontWeight="bold", color="#e8453c", dy=-6).encode(
        x="label_x:Q", y="label_y:Q", text="label:N")

    st.altair_chart((others + winners + lines + labels).properties(height=420), use_container_width=True)
    st.caption("Each grey dot is one nominee from 2018 to 2024, and the red dots are the winners. "
               "Winners with the same trophy score are nudged slightly apart. Hover over a dot to see the player and season.")

    st.markdown("**The six winners**")
    winners_table = (history.loc[history["rank"] == 1, ["year", "player", "club", "trophy_score", "model_place"]]
                     .rename(columns={"year": "Year", "player": "Winner", "club": "Club",
                                      "trophy_score": "Trophy score", "model_place": "Model's place that year"})
                     .sort_values("Year"))
    st.dataframe(winners_table, hide_index=True)

    
    # Section 2: which positions get nominated and which win
    st.subheader("Which positions win?")

    order = ["Forward", "Midfielder", "Defender", "Goalkeeper"]
    pos_counts = (history.assign(role=np.where(history["rank"] == 1, "Winner", "Other nominee"))
                  .groupby(["position_name", "role"]).size().reset_index(name="count"))
    totals = history.groupby("position_name").size().reindex(order).fillna(0).astype(int)
    wins = history[history["rank"] == 1].groupby("position_name").size().reindex(order).fillna(0).astype(int)
    label_df = pd.DataFrame({"position_name": order, "total": totals.values,
                             "text": [f"{t} nominees, {w} won" for t, w in zip(totals.values, wins.values)]})

    bars = (alt.Chart(pos_counts).mark_bar()
            .encode(y=alt.Y("position_name:N", sort=order, title=None),
                    x=alt.X("count:Q", title="Nominees, 2018-2024",
                            scale=alt.Scale(domain=[0, int(totals.max() * 1.35)])),
                    color=alt.Color("role:N", title=None,
                                    scale=alt.Scale(domain=["Other nominee", "Winner"], range=["#9aa0a6", "#e8453c"])),
                    order=alt.Order("role:N", sort="descending"),
                    tooltip=["position_name", "role", "count"]))
    text = (alt.Chart(label_df).mark_text(align="left", dx=6, fontSize=12)
            .encode(y=alt.Y("position_name:N", sort=order), x="total:Q", text="text:N"))
    st.altair_chart((bars + text).properties(height=200), use_container_width=True)

    no_win = [p for p in order if wins[p] == 0]
    no_win_share = totals[no_win].sum() / totals.sum() * 100
    names = " and ".join(p + "s" for p in no_win)
    st.caption(f"Winners are in red. {names} make up {no_win_share:.0f}% of nominees but have won none. "
               f"With only six winners this is a small sample, so read it as a pattern, not a rule.")

    
    # Section 3: how good is the model, honestly?
    st.subheader("How good is the model, honestly?")
    st.write("Every score below comes from a model that never saw that year, so it is an honest test, not a memorised one.")

    model_error = (history["model_score"] - history["rank"]).abs().mean()
    guess_error = (history["rank"] - history["rank"].median()).abs().mean()
    year_corr = history.groupby("year").apply(lambda g: g["model_score"].corr(g["rank"], method="spearman"))
    win = history[history["rank"] == 1].sort_values("year")
    top3 = int((win["model_place"] <= 3).sum())

    chance = float(sum(3 / (history["year"] == y).sum() for y in win["year"]))

    c1, c2, c3 = st.columns(3)
    c1.metric("Winners in model's top 3", f"{top3} of {len(win)}", f"random picking would get about {chance:.1f}",
              help="For each winner, did the model place him in the top 3 of that year's nominees?")
    c2.metric("Ranking skill", f"{year_corr.mean():.2f}", help="Average Spearman correlation per year. 1 is perfect, 0 is random.")
    c3.metric("Average miss (ranks)", f"{model_error:.1f}", f"{guess_error - model_error:.1f} better than guessing the middle",
              help="Average gap between the model's score and the real rank, compared with always guessing the median rank.")

    worst = win.loc[win["model_place"].idxmax()]
    st.caption(f"Where it struggles: {worst['player']} ({int(worst['year'])}) was placed #{int(worst['model_place'])} by the model. "
               f"It sees goals, assists, defensive stats and silverware, but not the stories, big moments and "
               f"voting campaigns that decide close years. Treat it as an informed estimate, not a forecast.")