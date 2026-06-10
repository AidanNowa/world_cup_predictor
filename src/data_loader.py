import pandas as pd
import numpy as np

from pathlib import Path

#dict of tournament importance weights, higher weights = more influcence on fitted attack/defense parameters

TOURNAMENT_WEIGHTS = {
    "FIFA World Cup":               4.0,
    "FIFA World Cup qualification": 2.5,
    "UEFA Euro":                    3.5,
    "UEFA Euro qualification":      2.0,
    "Copa America":                 3.5,
    "AFC Asian Cup":                3.0,
    "Africa Cup of Nations":        3.0,
    "CONCACAF Gold Cup":            2.5,
    "UEFA Nations League":          2.0,
    "Confederations Cup":           2.5,
    "FIFA Series":                  0.75,
    "Friendly":                     0.5,
} 
DEFAULT_WEIGHT = 1.0 

GLOBAL_MEAN_ELO = 1550.0


def load_raw_data(filepath: str | Path) -> pd.DataFrame:
    df = pd.read_csv(filepath, parse_dates=["date"])
    print(f"[data_loader] Loaded {len(df):,} raw matches.")
    return df 


def filter_by_date(df: pd.DataFrame, years: int=5) -> pd.DataFrame:
    #keep only recent games since those are far more relevant
    cutoff = pd.Timestamp.today() - pd.DateOffset(years=years)
    filtered = df[df["date"] >= cutoff].copy()
    print(f"[data_loader] After date filter ({years}y): {len(filtered):,} matches.")
    return filtered


def filter_tournaments(df: pd.DataFrame, exclude: list[str] | None=None) -> pd.DataFrame:
    #optionally drop tournaments that add noise
    if exclude is None:
        exclude = ["Olympic Games", "Youth", "U-20", "U-23", "Beach Soccer"]

    pattern = "|".join(exclude)
    mask = ~df["tournament"].str.contains(pattern, case=False, na=False)
    filtered = df[mask].copy()
    print(f"[data_loader] After tournament filter: {len(filtered):,} matches.")
    return filtered


def filter_sparse_teams(df: pd.DataFrame, min_matches: int=8) -> pd.DataFrame:
    match_counts = pd.concat([df["home_team"], df["away_team"]]).value_counts()
    valid_teams = match_counts[match_counts >= min_matches].index
    mask = df["home_team"].isin(valid_teams) & df["away_team"].isin(valid_teams)
    filtered = df[mask].copy()
    print(f"[data_loader] After sparse filter {len(filtered):,} matches {filtered['home_team'].nunique()} unique teams.")
    return filtered


def add_recency_weight(df: pd.DataFrame, half_life_days: int=365) -> pd.DataFrame:
    #apply exponential decay so recent matches matter more
    #weight = 0.5^(days_ago/half_life_days)
    today = pd.Timestamp.today()
    days_ago = (today - df["date"]).dt.days.clip(lower=0)
    df = df.copy()
    df["recency_weight"] = 0.5 ** (days_ago / half_life_days)
    return df


def add_tournament_weight(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tournament_weight"] = df["tournament"].map(TOURNAMENT_WEIGHTS).fillna(DEFAULT_WEIGHT)
    return df


def add_elo_weight(df: pd.DataFrame, elo_filepath: str | Path | None = None) -> pd.DataFrame:
    #add an oppent-quality weight based on both teams' elo raitings (static for most recent elo)
    df = df.copy()

    if elo_filepath is None or not Path(elo_filepath).exists():
        print("[data_loader] WARNING: ELO file not found. Setting elo_weight=1.0 for all matches.")
        df["home_elo"] = GLOBAL_MEAN_ELO
        df["away_elo"] = GLOBAL_MEAN_ELO
        df["elo_weight"] = 1.0
        return df

    elo_raw = pd.read_csv(elo_filepath)
    elo_map: dict[str, float] = dict(zip(elo_raw["nation"].str.strip(), elo_raw["elo_rating"].astype(float)))

    df["home_elo"] = df["home_team"].map(elo_map).fillna(GLOBAL_MEAN_ELO)
    df["away_elo"] = df["away_team"].map(elo_map).fillna(GLOBAL_MEAN_ELO)

    #warning for missing teams
    all_teams = pd.unique(df[["home_team", "away_team"]].values.ravel())
    missing = [t for t in all_teams if t not in elo_map]
    if missing:
        print(f"[data_loader] {len(missing)} teams not found in ELO file. Using {GLOBAL_MEAN_ELO:.0f} fallback: {sorted(missing)[:10]}" + ("..." if len(missing) > 10 else ""))

    #attempt to normalize mean ELO of both teams
    df["elo_weight"] = (df["home_elo"] + df["away_elo"]) / (2 * GLOBAL_MEAN_ELO)
    low = df["elo_weight"].quantile(0.05)
    high = df["elo_weight"].quantile(0.95)
    print(f"[data_loader] ELO weight range (5th-95th pct): {low:.2f} - {high:.2f}")

    return df
    


def add_combined_weight(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["weight"] = df["recency_weight"] * df["tournament_weight"] * df["elo_weight"]
    df["weight"] = df["weight"] / df["weight"].mean() #normalize
    return df


def drop_incomplete_rows(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    after = len(df)
    if before != after:
        print(f"[data_loader] Dropped {before - after} rows with missing scores.")
    return df


def get_all_teams(df: pd.DataFrame) -> list[str]:
    teams = pd.unique(df[["home_team", "away_team"]].values.ravel())
    return sorted(teams.tolist())


def load_and_prepare(filepath: str | Path, elo_filepath: str | Path, years: int=5, half_life_days: int=365, exclude_tournaments: list[str] | None=None, min_matches: int=8) -> pd.DataFrame:
    df = load_raw_data(filepath)
    df = filter_by_date(df, years=years)
    df = filter_tournaments(df, exclude=exclude_tournaments)
    df = filter_sparse_teams(df, min_matches=min_matches)
    df = drop_incomplete_rows(df)
    df = add_recency_weight(df, half_life_days=half_life_days)
    df = add_tournament_weight(df)
    df = add_elo_weight(df, elo_filepath=elo_filepath)
    df = add_combined_weight(df)

    print(f"[data_loader] Final dataset: {len(df):,} matches, {df['home_team'].nunique()} unique teams.")
    return df.reset_index(drop=True)


def summarize(df: pd.DataFrame) -> None:
    #print  diagnostic summary of the dataset
    print("\n=== Dataset Summary ===")
    print(f"Date range     : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Total matches  : {len(df):,}")
    print(f"Unique teams   : {df['home_team'].nunique()}")
    print(f"Avg goals/game : {(df['home_score'] + df['away_score']).mean():.2f}")
    print(f"\nTop tournaments by match count:")
    print(df["tournament"].value_counts().head(10).to_string())
    print(f"\nELO weight distribution:")
    print(df["elo_weight"].describe().round(3).to_string())
    print(f"\nHighest ELO matches (most informative):")
    top = df.nlargest(5, "elo_weight")[["date", "home_team", "away_team", "home_elo", "away_elo", "elo_weight"]]
    print(top.to_string(index=False))
    print(f"\nWeight distribution:")
    print(df["weight"].describe().round(3).to_string())
    print("=======================\n")


if __name__ == "__main__":
    DATA_PATH = Path("data/results.csv")
    ELO_PATH = Path("data/elo_ratings.csv")

    df = load_and_prepare(filepath=DATA_PATH, elo_filepath=ELO_PATH, years=5, half_life_days=365)

    summarize(df)
    
    print(df[["date", "home_team", "away_team", "home_score", "away_score", 
            "tournament", "weight", "recency_weight", "tournament_weight"]].tail(10).to_string(index=False))

