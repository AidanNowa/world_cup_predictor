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


def add_combined_weight(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["weight"] = df["recency_weight"] * df["tournament_weight"]
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


def load_and_prepare(filepath: str | Path, years: int=5, half_life_days: int=365, exclude_tournaments: list[str] | None=None) -> pd.DataFrame:
    df = load_raw_data(filepath)
    df = filter_by_date(df, years=years)
    df = filter_tournaments(df, exclude=exclude_tournaments)
    df = drop_incomplete_rows(df)
    df = add_recency_weight(df, half_life_days=half_life_days)
    df = add_tournament_weight(df)
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
    print(f"\nWeight distribution:")
    print(df["weight"].describe().round(3).to_string())
    print("=======================\n")


if __name__ == "__main__":
    DATA_PATH = Path("data/results.csv")

    df = load_and_prepare(filepath=DATA_PATH, years=5, half_life_days=365)

    summarize(df)
    
    print(df[["date", "home_team", "away_team", "home_score", "away_score", 
            "tournament", "weight", "recency_weight", "tournament_weight"]].tail(10).to_string(index=False))

