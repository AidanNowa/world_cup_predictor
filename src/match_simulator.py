import numpy as np

from pathlib import Path
from data_loader import load_and_prepare
from dataclasses import dataclass
from poisson_model import PoissonModel



@dataclass
class MatchResult:
    #holds the outcome of a single simulated match
    team_a: str
    team_b: str
    goals_a: int
    goals_b: int
    winner: str | None
    went_to_et: bool = False
    went_to_pens: bool = False
    is_knockout: bool = False
    
    
    def loser(self) -> str | None:
        if self.winner is None:
            return None
        if self.winner == self.team_a:
            return self.team_b
        return self.team_a

    
    def is_draw(self) -> bool:
        if self.winner is None:
            return True
        return False

    
    def summary(self) -> str:
        score = f"{self.team_a} {self.goals_a} - {self.team_b} {self.goals_b}"
        if self.went_to_pens:
            return f"{score} (pens: {self.winner} win)"
        if self.went_to_et:
            return f"{score} (AET: {self.winner} win)"
        if self.is_draw():
            return f"{score} (draw)"
        return f"{score} ({self.winner} win)"


class MatchSimulator:
    #simulates matches using the fitted PoissonModel
    #Add goal influence of ET, 30 mins ET is about 1/3 of 90mins, with fatigure it may reduce scoring slightly
    ET_SCALE = 0.28

    #Penalty shootout odds are about 50/50 (could use team-level stats but this is easier for now)
    PENALTY_WIN_PROB = 0.5

    def __init__(self, model: PoissonModel):
        self.model = model

    def simulate_group_match(self, team_a: str, team_b: str) -> MatchResult:
        #simulate group stage match (90mins only, draws allowed)
        goals_a, goals_b = self.model.simulate_match(team_a, team_b)
        winner = self._ninety_min_winner(team_a, team_b, goals_a, goals_b)

        return MatchResult(
            team_a=team_a,
            team_b=team_b,
            goals_a=goals_a,
            goals_b=goals_b,
            winner=winner,
            is_knockout=False
        )


    def simulate_knockout_match(self, team_a: str, team_b: str) -> MatchResult:
        #simulate knowckout match (90mins, ET if needed, Penalties if needed, no draws)
        goals_a, goals_b = self.model.simulate_match(team_a, team_b)
        winner = self._ninety_min_winner(team_a, team_b, goals_a, goals_b)

        went_to_et = False
        went_to_pens = False

        if winner is None:
            #extra time with reduced scoring rate
            went_to_et = True
            et_a, et_b = self._simulate_extra_time(team_a, team_b)
            goals_a += et_a
            goals_b += et_b
            winner = self._ninety_min_winner(team_a, team_b, goals_a, goals_b)

        if winner is None:
            #penalties
            went_to_pens = True
            winner = self._simulate_penalties(team_a, team_b)

        return MatchResult(
            team_a=team_a,
            team_b=team_b,
            goals_a=goals_a,
            goals_b=goals_b,
            winner=winner,
            went_to_et=went_to_et,
            went_to_pens=went_to_pens,
            is_knockout=True,
        )


    def predict_group_probs(self, team_a: str, team_b: str) -> dict[str, float]:
        #return win/draw/loss probabilities for a group match for reporting without running simulation
        return self.model.predict_outcome_probs(team_a, team_b)


    def _ninety_min_winner(self, team_a: str, team_b: str, goals_a: int, goals_b: int) -> str | None:
        if goals_a > goals_b:
            return team_a
        if goals_b > goals_a:
            return team_b
        return None


    def _simulate_extra_time(self, team_a: str, team_b: str) -> tuple[int, int]:
        #sim ET by scaling lambdas down and return additional goals
        lambda_a, lambda_b = self.model.predict_lambda(team_a, team_b)
        et_goals_a = np.random.poisson(lambda_a * self.ET_SCALE)
        et_goals_b = np.random.poisson(lambda_b * self.ET_SCALE)
        return int(et_goals_a), int(et_goals_b)


    def _simulate_penalties(self, team_a: str, team_b: str) -> str:
        #sim penalty shootout as weighted coin flip
        if np.random.random() < self.PENALTY_WIN_PROB:
            return team_a
        return team_b


if __name__ == "__main__":
    df = load_and_prepare(Path("data/results.csv"))

    model = PoissonModel().fit(df)
    simulator = MatchSimulator(model)

    print("-----Group Stage Match-----")
    result = simulator.simulate_group_match("France", "Brazil")
    print(result.summary())

    print("\n-----Knockout Match------")
    result = simulator.simulate_knockout_match("England", "Spain")    
    print(result.summary())
    if result.went_to_et:
        print("   -> Went to ET")
    if result.went_to_pens:
        print("   -> Decided by penalties")

    print("\n-----Outcome Probabilities-----")
    probs = simulator.predict_group_probs("Germany", "Argentina")
    print(f"Germany vs Argentina:")
    print(f"   P(Germany wins)   = {probs['win']:.1%}")
    print(f"   P(draw)           = {probs['draw']:.1%}")
    print(f"   P(Argentina wins) = {probs['lose']:.1%}")

    print("\n-----10 Simulated Morocco vs Norway Scorelines-----")
    for _ in range(10):
        result = simulator.simulate_group_match("Morocco", "Norway")
        print(f"   {result.summary()}")














