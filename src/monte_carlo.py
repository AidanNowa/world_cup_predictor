import pandas as pd
import time
import os 

from collections import defaultdict
from tqdm import tqdm
from tournament import TournamentSimulator, GROUPS
from pathlib import Path

from data_loader import load_and_prepare
from poisson_model import PoissonModel
from match_simulator import MatchSimulator


class MonteCarloSimulator:
    #run repeated full-tournament sims and get stats
    def __init__(self, tournament: TournamentSimulator, n_simulations: int = 10_000):
        self.tournament = tournament
        self.n_simulations = n_simulations

        #all teams across all groups
        self.all_teams: list[str] = [team for teams in GROUPS.values() for team in teams]

        #accumulators to count across all simulations (reduce memory)
        self._wins: dict[str, int] = defaultdict(int)
        self._finals: dict[str, int] = defaultdict(int)
        self._semis: dict[str, int] = defaultdict(int)
        self._group_advance: dict[str, int] = defaultdict(int)
        self._group_win: dict[str, int] = defaultdict(int)

        self._ran = False #guard to prevent reading results before running

    
    def run(self) -> MonteCarloSimulator:
        print(f"\n[monte_carlo] Running {self.n_simulations:,} simulations...")
        t0 = time.time()
        
        for _ in tqdm(range(self.n_simulations), desc="simulating tournaments", unit="sims"):
            result = self.tournament.simulate()
            self._record(result)

        elapsed = time.time() - t0
        print(f"[monte_carlo] Done in {elapsed:.1f}s ({elapsed / self.n_simulations * 1000:.1f} ms/sim\n")

        self._ran = True
        return self


    def to_dataframe(self) -> pd.DataFrame:
        #return a dataframe for all teams ranked by win probability
        self._assert_ran()
        
        team_to_group = {
            team: group 
            for group, teams in GROUPS.items()
            for team in teams
        }

        n = self.n_simulations
        rows = []
        for team in self.all_teams:
            rows.append({
                "team": team,
                "group": team_to_group.get(team, "?"),
                "win_pct": round(self._wins[team] / n * 100, 2),
                "final_pct": round(self._finals[team] / n * 100, 2),
                "semi_pct": round(self._semis[team] / n * 100, 2),
                "group_advance_pct": round(self._group_advance[team] / n * 100, 2),
                "group_win_pct": round(self._group_win[team] / n * 100, 2)
            })

        return (pd.DataFrame(rows).sort_values("win_pct", ascending=False).reset_index(drop=True))

    
    def print_report(self, top_n: int = 16) -> None:
        self._assert_ran()
        df = self.to_dataframe()

        header = (
            f"\n{'-'*72}\n"
            f" 2026 FIFA World Cup Monte Carlo Results  "
            f"({self.n_simulations:,} simulations)\n"
            f"{'-'*72}\n"
            f"{'Rank':<5} {'Team':<25} {'Grp':>3}  "
            f"{'Win%':>6}  {'Final%':>7}  {'Semi%':>6}  {'Advance%':>9}  {'1st%':>5}\n"
            f"{'-'*72}"
        )
        print(header)

        for rank, row in df.head(top_n).iterrows():
            print(                
                f"{rank+1:<5} {row['team']:<25} {row['group']:>3}  "
                f"{row['win_pct']:>6.2f}  "
                f"{row['final_pct']:>7.2f}  "
                f"{row['semi_pct']:>6.2f}  "
                f"{row['group_advance_pct']:>9.2f}  "
                f"{row['group_win_pct']:>5.2f}"
            )

        print(f"{'-'*72}\n")

    
    def group_stage_report(self) -> None:
        self._assert_ran()
        df = self.to_dataframe()
        
        print(f"\n{'-'*60}")
        print(f"Group Stage Probabilities ({self.n_simulations:,} simulations)")
        print(f"{'-'*60}")

        for group, teams in GROUPS.items():
            print(f"\n   -----Group {group}-----")
            print(f"   {'Team'} {'1st%':>6}  {'Advance%':>9}")
            print(f"   {'-'*44}")
            group_df = df[df["group"] == group].sort_values("group_advance_pct", ascending=False)

            for _, row in group_df.iterrows():
                print(f"   {row['team']} {row['group_win_pct']:>6.2f}  {row['group_advance_pct']:>9.2f}")
        
    
    def save_csv(self, path: str = "results/monte_carlo_results.csv") -> None:
        self._assert_ran()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        print(f"[monte_carlo] Results saved to {path}")


    #internal helper for updating accumulators, called once per sim
    def _record(self, result) -> None:        
        self._wins[result.winner] += 1
        
        self._finals[result.winner] += 1
        self._finals[result.runner_up] += 1

        for team in [result.winner, result.runner_up] + result.third_place:
            self._semis[team] += 1

        for group, standings in result.group_standings.items():
            #1st place
            self._group_advance[standings[0].team] += 1
            self._group_win[standings[0].team] += 1
            #2nd place
            self._group_advance[standings[1].team] += 1
        
        #using the teams in the knockout we can infer 3rd place qualifiers
        first_and_second = {
            standings[0].team for standings in result.group_standings.values()
        } | {
            standings[1].team for standings in result.group_standings.values()
        }
        r32_teams: set[str] = set()
        if result.knockout_results:
            for r in result.knockout_results[:16]:
                r32_teams.add(r.team_a)
                r32_teams.add(r.team_b)

        for team in r32_teams - first_and_second:
            self._group_advance[team] += 1

    def _assert_ran(self) -> None:
        if not self._ran:
            raise RuntimeError("No simulations run yet. Call .run() first.")


if __name__ == "__main__":
    df = load_and_prepare(Path("data/results.csv"))
    model = PoissonModel().fit(df)
    simulator = MatchSimulator(model)
    tournament = TournamentSimulator(simulator)

    mc = MonteCarloSimulator(tournament, n_simulations=10_000)
    mc.run()

    mc.print_report(top_n=20)
    mc.group_stage_report()
    
    mc.save_csv("results/monte_carlo_results.csv")

    #sanity checks
    df_results = mc.to_dataframe()
    
    #win prob should sum to ~100%
    total = df_results["win_pct"].sum()
    print(f"Win% total (should be ~100): {total:.2f}%")

    #advancing from group should be ~66.7% since 32/48 teams advance
    avg_advance = df_results["group_advance_pct"].mean()
    print(f"Avg group advance % (should be ~66.7): {avg_advance:.2f}%")
















