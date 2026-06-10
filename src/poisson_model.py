import numpy as np
import pandas as pd

from scipy.optimize import minimize
from scipy.stats import poisson
from pathlib import Path
from tqdm import tqdm

from data_loader import load_and_prepare

class PoissonModel:
    def __init__(self):
        self.teams: list[str] = []
        self.params: dict[str, dict[str, float]] = {}
        self.intercept: float = 0.0
        self.rho: float = 0.0 #dixon-coles low-score correction
        self._team_index: dict[str, int] = {} #team name mapping to index in parameter vector

    def fit(self, df: pd.DataFrame) -> "Poisson Model":
        #fit attack and defense parameters for every team based on goals
        self.teams = sorted(pd.unique(df[["home_team", "away_team"]].values.ravel()).tolist())
        n = len(self.teams)
        self._team_index = {team: i for i, team in enumerate(self.teams)}

        print(f"[poisson_model] Fitting model for {n} teams on {len(df):,} matches...")

        #index 0 = intercept (log baseline goals)
        #index 1, ..., n = attack parameters (one per team)
        #index n+1, ..., 2n = defense parameters (one per team)
        #index 2n+1: rho (dixon-coles low-score correction)
        
        #initial guess: intercept = log(1.5 goals), all attack/defense = 0
        x0 = np.zeros(1 + 2*n + 1)
        x0[0] = np.log(1.5)

        #constraint: mean attack = 0 (pins the scale os the model is identifiable)
        #without it attack and intercept would trade off freely 
        constraints = {
            "type": "eq",
            "fun": lambda x: x[1: n+1].mean() #mean attack == 0
        }

        bounds = (
            [(None, None)]
            + [(None, None)] * n
            + [(None, None)] * n
            + [(-0.99, 0.99)] #rho must say in (-1,1) to keep tau correction valid
        )

        self._pbar = tqdm(desc="Fitting model", unit=" evals", colour="green")

        result = minimize(
            fun=self._negative_log_likelihood,
            x0=x0,
            args=(df,),
            method="SLSQP",
            constraints=constraints,
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-9}
        )

        self._pbar.close()

        if not result.success:
            print(f"[poisson_model] WARNING: optimiser did not fully converge.")
            print(f"Message: {result.message}")
    
        self._unpack_params(result.x)
        print(f"[poisson_model] Fitting complete.\nrho (Dixon-Coles) = {self.rho:.4f}")
        return self #allows for model = PoissonModel().fit(df)

    
    def predict_lambda(self, team_a: str, team_b: str) -> tuple[float, float]:
        #return expected goals (lambda) for each team in a match
        self._assert_fitted()
        self._assert_known(team_a, team_b)
        
        log_lambda_a = (self.intercept + self.params[team_a]["attack"] + self.params[team_b]["defense"])
        log_lambda_b = (self.intercept + self.params[team_b]["attack"] + self.params[team_a]["defense"])
        
        return np.exp(log_lambda_a), np.exp(log_lambda_b)


    def predict_outcome_probs(self, team_a: str, team_b: str, max_goals: int=10) -> dict[str, float]:
    #compute p(team_a wins), p(draw), p(team_b wins) by summing over the join poisson probablity matrix. 
        lambda_a, lambda_b = self.predict_lambda(team_a, team_b)
        
        #prob_matrix[i, j] = P(team_a scores i) * P(team_b scores j)
        goals_range = np.arange(0, max_goals + 1)
        prob_a = poisson.pmf(goals_range, lambda_a) #shape (max_goals+1,)
        prob_b = poisson.pmf(goals_range, lambda_b)
        prob_matrix = np.outer(prob_a, prob_b) #shape (G+1, G+1)

        p_win = float(np.tril(prob_matrix, k=-1).sum()) #i>j (where A scores more)
        p_draw = float(np.trace(prob_matrix)) #i==j
        p_lose = float(np.triu(prob_matrix, k=1).sum()) #i<j (where B scores more)

        return {"win": p_win, "draw": p_draw, "lose": p_lose}


    def simulate_match(self, team_a: str, team_b: str) -> tuple[int, int]:
        #sample a single scoreline from the fitted poisson, used in simulator
        lambda_a, lambda_b = self.predict_lambda(team_a, team_b)
        goals_a = np.random.poisson(lambda_a)
        goals_b = np.random.poisson(lambda_b)
        return int(goals_a), int(goals_b)


    def team_rankings(self) -> pd.DataFrame:
        self._assert_fitted()
        rows = []
        for team in self.teams:
            attack = self.params[team]["attack"]
            defense = self.params[team]["defense"]
            rows.append({
                "team": team,
                "attack": round(attack, 4),
                "defense": round(defense, 4),
                #lower defense = concede fewer goals (better defense) 
                "strength": round(attack - defense, 4) #subtract defense so higher is still better 
            })
        return (pd.DataFrame(rows).sort_values("strength", ascending=False).reset_index(drop=True))


    def _negative_log_likelihood(self, x: np.ndarray, df: pd.DataFrame) -> float:
        n = len(self.teams)
        intercept  = x[0]
        attack_vec  = x[1: n + 1]
        defense_vec = x[n + 1:]
 
        #map team names to parameter values via index
        idx_home = df["home_team"].map(self._team_index).values
        idx_away = df["away_team"].map(self._team_index).values
 
        log_lambda_home = intercept + attack_vec[idx_home] + defense_vec[idx_away]
        log_lambda_away = intercept + attack_vec[idx_away] + defense_vec[idx_home]
 
        lambda_home = np.exp(log_lambda_home)
        lambda_away = np.exp(log_lambda_away)
 
        home_goals = df["home_score"].values
        away_goals = df["away_score"].values
        weights    = df["weight"].values
 
        #log P(k | lambda) = k*log(lambda) - lambda - log(k!)
        ll_home = poisson.logpmf(home_goals, lambda_home)
        ll_away = poisson.logpmf(away_goals, lambda_away)
 
        #weighted sum, more important matches contribute more to the fit
        weighted_ll = weights * (ll_home + ll_away)
        nll = -weighted_ll.sum()   #negate because we minimize
        
        self._pbar.set_postfix({"loss": f"{nll:.2f}"})
        self._pbar.update(1)
        return nll        

    
    def _unpack_params(self, x: np.ndarray) -> None:
        n = len(self.teams)
        self.intercept = x[0]
        print(f"[debug] Unpacking {n} teams")
        for i, team in enumerate(self.teams):
            self.params[team] = {
                "attack": x[1+i],
                "defense": x[n+1+i]
            }

    
    def _assert_fitted(self) -> None:
        if not self.params:
            raise RuntimeError("model has not been fitted yet. Call .fit(df) first.")

    
    def _assert_known(self, *teams: str) -> None:
        for team in teams:
            if team not in self.params:
                raise ValueError(f"Unknown team: '{team}'. Check spelling or add more historical data.")


if __name__ == "__main__":
    df = load_and_prepare(Path("data/results.csv"), Path("data/elo_ratings.csv"))

    model = PoissonModel().fit(df)
    print("\nTop 15 teams by model strength:")
    print(model.team_rankings().head(15).to_string(index=False))

    #example matchup: France vs Brazil
    team_a, team_b = "France", "Brazil"
    la, lb = model.predict_lambda(team_a, team_b)
    print(f"\n{team_a} vs {team_b}")
    print(f"    Expected Goals: {team_a} {la:.2f} - {lb:.2f} {team_b}")

    probs = model.predict_outcome_probs(team_a, team_b)
    print(f"  P(win)  = {probs['win']}")
    print(f"  P(draw) = {probs['draw']}")
    print(f"  P(lose) = {probs['lose']}")
    
    #simulate a single scoreline
    goals_a, goals_b = model.simulate_match(team_a, team_b)
    print(f"\n  Simulated scoreline: {team_a} {goals_a} - {goals_b} {team_b}")





