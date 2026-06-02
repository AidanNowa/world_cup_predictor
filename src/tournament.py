import numpy as np
import pandas as pd
import random

from pathlib import Path
from dataclasses import dataclass, field

from data_loader import load_and_prepare
from poisson_model import PoissonModel
from match_simulator import MatchSimulator, MatchResult

#simulate full 2026 WC tournament

GROUPS: dict[str, list[str]] = {
    "A": ["Czech Republic", "Mexico", "South Africa", "South Korea"],
    "B": ["Bosnia and Herzegovina", "Canada", "Qatar", "Switzerland"],
    "C": ["Brazil", "Haiti", "Morocco", "Scotland"],
    "D": ["Australia", "Paraguay", "Turkey", "United States"],
    "E": ["Curaçao", "Ecuador", "Germany", "Ivory Coast"],
    "F": ["Japan", "Netherlands", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Cape Verde", "Saudi Arabia", "Spain", "Uruguay"],
    "I": ["France", "Iraq", "Norway", "Senegal"],
    "J": ["Algeria", "Argentina", "Austria", "Jordan"],
    "K": ["Colombia", "DR Congo", "Portugal", "Uzbekistan"],
    "L": ["Croatia", "England", "Ghana", "Panama"],

}

# Slots are ("position", "group") or ("3rd", ["G1","G2",...]) for third-place
R32_BRACKET_TEMPLATE = [
    #Match 73
    (("2nd", "A"),  ("2nd", "B")),
    #Match 74
    (("1st", "E"),  ("3rd", ["A", "B", "C", "D", "F"])),
    #Match 75
    (("1st", "F"),  ("2nd", "C")),
    #Match 76
    (("1st", "C"),  ("2nd", "F")),
    #Match 77
    (("1st", "I"),  ("3rd", ["C", "D", "F", "G", "H"])),
    #Match 78
    (("2nd", "E"),  ("2nd", "I")),
    #Match 79
    (("1st", "A"),  ("3rd", ["C", "E", "F", "H", "I"])),
    #Match 80
    (("1st", "L"),  ("3rd", ["E", "H", "I", "J", "K"])),
    #Match 81
    (("1st", "D"),  ("3rd", ["B", "E", "F", "I", "J"])),
    #Match 82
    (("1st", "G"),  ("3rd", ["A", "E", "H", "I", "J"])),
    #Match 83
    (("2nd", "K"),  ("2nd", "L")),
    #Match 84
    (("1st", "H"),  ("2nd", "J")),
    #Match 85
    (("1st", "B"),  ("3rd", ["E", "F", "G", "I", "J"])),
    #Match 86
    (("1st", "J"),  ("2nd", "H")),
    #Match 87
    (("1st", "K"),  ("3rd", ["D", "E", "I", "J", "L"])),
    #Match 88
    (("2nd", "D"),  ("2nd", "G")),
]


@dataclass
class TeamStanding:
    #track single team group stage record
    team: str
    played: int=0
    won: int=0
    drawn: int=0
    lost: int=0
    goals_for: int=0
    goals_against: int=0
    points: int=0

    @property
    def goal_difference(self) -> int:
        return self.goals_for - self.goals_against

    def update(self, goals_for: int, goals_against: int) -> None:
        self.played += 1
        self.goals_for += goals_for
        self.goals_against += goals_against
        if goals_for > goals_against:
            self.won += 1
            self.points += 3
        elif goals_for == goals_against:
            self.drawn += 1
            self.points += 1
        else:
            self.lost += 1

@dataclass
class TournamentResult:
    #stores complete coucome of one sim tournament
    winner: str
    runner_up: str
    third_place: list[str]
    group_standings: dict[str, list[TeamStanding]]
    knockout_results: list[MatchResult] = field(default_factory=list)


class TournamentSimulator:
    def __init__(self, simulator: MatchSimulator, groups: dict[str, list[str]] = GROUPS):
        self.simulator = simulator
        self.groups = groups

    #simulate full tournament
    def simulate(self) -> TournamentResult:
        group_standings, group_results = self._simulate_all_groups()
        qualifiers, advancing_thirds = self._determine_qualifiers(group_standings)
        
        bracket = self._build_bracket(qualifiers, advancing_thirds)
        knockout_results, finalist_a, finalist_b, sf_losers = self._simulate_knockout_rounds(bracket)

        final_result = self.simulator.simulate_knockout_match(finalist_a, finalist_b)
        knockout_results.append(final_result)

        return TournamentResult(
            winner=final_result.winner,
            runner_up=final_result.loser(),
            third_place=sf_losers,
            group_standings=group_standings,
            knockout_results=knockout_results
        )


    def predict_tournament(self) -> TournamentResult:
        #build sinle most likely tournament bracket for bracket challenges
        #select most likely winner not winner from single instance
        #award draw if P(draw) is highest
        group_standings, group_results = self._predict_all_groups()

        qualifiers, advancing_thirds = self._determine_qualifiers(group_standings)

        bracket = self._build_bracket(qualifiers, advancing_thirds)
        knockout_results, finalist_a, finalist_b, sf_losers = self._predict_knockout_rounds(bracket)

        final_result = self._predict_knockout_match(finalist_a, finalist_b)
        knockout_results.append(final_result)

        return TournamentResult(
            winner=final_result.winner,
            runner_up=final_result.loser(),
            third_place=sf_losers,
            group_standings=group_standings,
            knockout_results=knockout_results
        )

    #group stage functions
    def _simulate_all_groups(self) -> tuple[dict[str, list[TeamStanding]], list[MatchResult]]:
        all_standings: dict[str, list[TeamStanding]] = {}
        all_results: list[MatchResult] = []

        for group_letter, teams in self.groups.items():
            standings, results = self._simulate_group(teams)
            all_standings[group_letter] = standings
            all_results.extend(results)

        return all_standings, all_results

    
    def _simulate_group(self, teams: list[str]) -> tuple[list[TeamStanding], list[MatchResult]]:
        #simulate all games in a single group
        standings = {team: TeamStanding(team=team) for team in teams}
        results: list[MatchResult] = []

        #all teams play each other once
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                team_a, team_b = teams[i], teams[j]
                result = self.simulator.simulate_group_match(team_a, team_b)
                results.append(result)
                
                standings[team_a].update(result.goals_a, result.goals_b)
                standings[team_b].update(result.goals_b, result.goals_a)

        sorted_standings = self._sort_standings(list(standings.values()), results)
        return sorted_standings, results


    def _sort_standings(self, standings: list[TeamStanding], results: list[MatchResults]) -> list[TeamStanding]:
        #sort group using the fifa tie breaker rules
        #Points->H2H points->H2H GD->H2H GF->Overall GD->Overall GF->Random (lots)
        standings.sort(key=lambda s: s.points, reverse=True)
        standings = self._apply_head_to_head(standings, results)

        return standings


    def _apply_head_to_head(self, standings: list[TeamStanding], results: list[MatchResult]) -> list[TeamStanding]:
        h2h: dict[tuple[str, str], tuple[int, int]] = {}
        for result in results:
            h2h[(result.team_a, result.team_b)] = (result.goals_a, result.goals_b)
            h2h[(result.team_b, result.team_a)] = (result.goals_b, result.goals_a)
    
        def h2h_points(team: str, opponents: list[str]) -> int:
            pts = 0
            for opp in opponents:
                ga, gb = h2h.get((team, opp), (0, 0))
                if ga > gb:
                    pts += 3
                elif ga == gb:
                    pts += 1
            return pts
        
        def h2h_gd(team: str, opponents: list[str]) -> int:
            return sum(h2h.get((team, opp), (0,0))[0] - h2h.get((team, opp), (0, 0))[1] for opp in opponents)

        def h2h_gf(team: str, opponents: list[str]) -> int:
            return sum(h2h.get((team, opp), (0, 0))[0] for opp in opponents)

        #find group of teams tied on points and resort each tied cluster
        i = 0
        result_list = []
        while i < len(standings):
            j = i + 1
            while j < len(standings) and standings[j].points == standings[i].points:
                j += 1

            tied_group = standings[i:j]
            if len(tied_group) > 1:
                opponents = [s.team for s in tied_group]
                tied_group.sort(
                    key=lambda s: (
                        h2h_points(s.team, [o for o in opponents if o != s.team]),
                        h2h_gd(s.team,     [o for o in opponents if o != s.team]),
                        h2h_gf(s.team,     [o for o in opponents if o != s.team]),
                        s.goal_difference,
                        s.goals_for,
                        random.random(), #lots
                    ),
                    reverse=True,
                )
            result_list.extend(tied_group)
            i=j

        return result_list

    
    def _predict_all_groups(self) -> tuple[dict[str, list[TeamStanding]], list[MatchResult]]:
        all_standings: dict[str, list[TeamStanding]] = {}
        all_results: list[MatchResult] = []

        for group_letter, teams in self.groups.items():
            standings, results = self._predict_group(teams)
            all_standings[group_letter] = standings
            all_results.extend(results)
    
        return all_standings, all_results


    def _predict_group(self, teams: list[str]) -> tuple[list[TeamStanding], list[MatchResult]]:
        #predict all 6 group matches by taking the most probable outcome
        standings = {team: TeamStanding(team=team) for team in teams}
        results: list[MatchResult] = []

        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                team_a, team_b = teams[i], teams[j]
                result = self._predict_group_match(team_a, team_b)
                results.append(result)  
                standings[team_a].update(result.goals_a, result.goals_b)
                standings[team_b].update(result.goals_b, result.goals_a)

        sorted_standings = self._sort_standings(list(standings.values()), results)
        return sorted_standings, results


    def _predict_group_match(self, team_a: str, team_b: str) -> MatchResult:
        probs = self.simulator.predict_group_probs(team_a, team_b)
        lambda_a, lambda_b = self.simulator.model.predict_lambda(team_a, team_b)

        goals_a = round(lambda_a)
        goals_b = round(lambda_b)

        #determine most likely outcome and adjust scoreline if needed
        best_outcome = max(probs, key=probs.get)
        
        if best_outcome == "win" and goals_a <= goals_b:
            goals_a = goals_b + 1
        elif best_outcome == "lose" and goals_b <= goals_a:
            goals_b = goals_a + 1
        elif best_outcome == "draw":
            score = min(goals_a, goals_b)
            goals_a = goals_b = score
        
        winner = None #stay None if draw 
        if best_outcome == "win":
            winner = team_a
        elif best_outcome == "lose":
            winner = team_b 
        
        return MatchResult(
            team_a=team_a,
            team_b=team_b,
            goals_a=goals_a,
            goals_b=goals_b,
            winner=winner,
            is_knockout=False
        )

    
    def _predict_knockout_match(self, team_a: str, team_b: str) -> MatchResult:
        probs = self.simulator.predict_group_probs(team_a, team_b)
        lambda_a, lambda_b = self.simulator.model.predict_lambda(team_a, team_b)

        goals_a = round(lambda_a)
        goals_b = round(lambda_b)

        if probs["win"] >= probs["lose"]:
            winner = team_a
            if goals_a <= goals_b:
                goals_a = goals_b + 1
        else:
            winner = team_b
            if goals_b <= goals_a:
                goals_b = goals_a + 1

        return MatchResult(
            team_a=team_a,
            team_b=team_b,
            goals_a=goals_a,
            goals_b=goals_b,
            winner=winner,
            is_knockout=True
        )
        

    def _predict_knockout_rounds(self, bracket: list[tuple[str, str]]) -> tuple[list[MatchResult], str, str, list[str]]:
        all_results: list[MatchResult] = []
        teams = bracket
        round_names = ["Round of 32", "Round of 16", "Quarter-Finals", "Semi-Finals"]
        round_idx = 0
        sf_losers: list[str]=[]

        while len(teams) > 1:
            round_winners = []
            
            for team_a, team_b in teams:
                result = self._predict_knockout_match(team_a, team_b)
                all_results.append(result)
                round_winners.append(result.winner)

                if round_idx < len(round_names) and round_names[round_idx] == "Semi-Finals":
                    sf_losers.append(result.loser())

            teams = [(round_winners[i], round_winners[i + 1]) for i in range(0, len(round_winners) - 1, 2)]
            round_idx += 1
        finalist_a, finalist_b = teams[0]
        return all_results, finalist_a, finalist_b, sf_losers

    def _determine_qualifiers(self, group_standings: dict[str, list[TeamStanding]]) -> tuple[dict[str, dict[str, str]], list[TeamStanding]]:
        #determine 32 advancing teams (1st and 2nd plus 8 best 3rd place)
        qualifiers: dict[str, dict[str, str]] = {}
        third_place_standings: list[TeamStanding] = []

        for group, standings in group_standings.items():
            qualifiers[group] = {
                "1st": standings[0].team,
                "2nd": standings[1].team,
                "3rd": standings[2].team,
            }
            third_place_standings.append(standings[2])
        third_place_standings.sort(key=lambda s: (s.points, s.goal_difference, s.goals_for), reverse=True)
        advancing_thirds = third_place_standings[:8]

        return qualifiers, advancing_thirds


    #knockout bracket functions
    def _build_bracket(self, qualifiers: dict[str, dict[str, str]], advancing_thirds: list[TeamStanding]) -> list[tuple[str, str]]:
        #3rd place slots list eligible groups best ranked advancing third-place team from those groups fills the slot
        advancing_third_by_group: dict[str, str] = {
            s.team: qualifiers_group
            for qualifiers_group, q in qualifiers.items()
            if any(s.team == q["3rd"] for s in advancing_thirds)
            for s in advancing_thirds
            if s.team == q["3rd"]
        }
        advancing_set = {s.team for s in advancing_thirds}
        group_to_third: dict[str, str | None] = {
            group: (q["3rd"] if q["3rd"] in advancing_set else None)
            for group, q in qualifiers.items()
        }

        def resolve_slot(slot: tuple) -> str:
            position, group_or_list = slot
            if position in ("1st", "2nd"):
                return qualifiers[group_or_list][position]
            else:
                #3rd place slot, find best advancing 3rd from eligible group
                eligible_groups: list[str] = group_or_list
                for standing in advancing_thirds:
                    team = standing.team
                    for g, q in qualifiers.items():
                        if q["3rd"] == team and g in eligible_groups:
                            return team
            #fallback that should not happen
            print(f"[ERROR]: Fallback team used for 3rd place selection for slot: {slot}")
            return advancing_thirds[0].team 

        bracket = [(resolve_slot(slot_a), resolve_slot(slot_b)) for slot_a, slot_b in R32_BRACKET_TEMPLATE]
        return bracket


    def _simulate_knockout_rounds(self, bracket: list[tuple[str, str]]) -> tuple[list[MatchResult], str, str, list[str]]:
        #simulate all rounds except final and return those finalists, and 2 semi-final losers
        all_results: list[MatchResults] = []
        teams = bracket

        round_names = ["Round of 32", "Round of 16", "Quarter-Finals", "Semi-Finals"]
        round_idx = 0
        sf_losers: list[str] = []
        while len(teams) > 1: #adjusted from >2 since this skipped the semi-finals
            round_winners = []
        
            for team_a, team_b in teams:
                result = self.simulator.simulate_knockout_match(team_a, team_b)
                all_results.append(result)
                round_winners.append(result.winner)

                if round_names[round_idx] == "Semi-Finals" and round_idx < len(round_names):
                    sf_losers.append(result.loser())
            
            #pair winners for next round
            teams = [(round_winners[i], round_winners[i + 1]) for i in range(0, len(round_winners) - 1, 2)]
            round_idx += 1

        finalist_a, finalist_b = teams[0]
        return all_results, finalist_a, finalist_b, sf_losers

#printing helpers
def print_group_standings(group_standings: dict[str, list[TeamStanding]]) -> None:
    #print all group standings
    for group, standings in group_standings.items():
        print(f"\n--- Group {group} ---")
        print(f"{'Team':<25} {'P':>2} {'W':>2} {'D':>2} {'L':>2} "
              f"{'GF':>3} {'GA':>3} {'GD':>4} {'Pts':>4}")
        print("-" * 55)
        for i, s in enumerate(standings):
            qualifier = "✓" if i < 2 else " "
            print(f"{qualifier} {s.team:<23} {s.played:>2} {s.won:>2} "
                  f"{s.drawn:>2} {s.lost:>2} {s.goals_for:>3} "
                  f"{s.goals_against:>3} {s.goal_difference:>4} {s.points:>4}")
 
 
def print_knockout_results(results: list[MatchResult]) -> None:
    #print knockout round results
    round_sizes = {32: "Round of 32", 16: "Round of 16",
                   8: "Quarter-Finals", 4: "Semi-Finals", 2: "Final"}
    n = len(results)
    print(f"\n-----Knockout Results ({n} matches)-----")
    for r in results:
        print(f"  {r.summary()}")


def print_bracket(result: TournamentResult) -> None:
    results = result.knockout_results
    toal = len(results)

    round_definitions = [
        ("Round of 32", 16),
        ("Round of 16", 8),
        ("Quarter-Finals", 4),
        ("Semi-Finals", 2),
        ("Final", 1)
    ]
    
    idx = 0
    print("\n" + "=" * 55)
    print("   2026 WORLD CUP  (PREDICTED BRACKET)")
    print("=" * 55)

    for round_name, n_matches in round_definitions:
        round_results = results[idx: idx + n_matches]
        if not round_results:
            break
        print("\n" + '-' * 55)
        print(f"   {round_name}")
        print('-' * 55)
        for r in round_results:
            winner = "DRAW"
            if r.winner: 
                win_indicator = f"-> {r.winner}"
            print(f"   {r.team_a} {r.goals_a} - {r.goals_b} { r.team_b}  {win_indicator}")
        idx += n_matches

    print("\n" + "=" * 55)
    print(f"\nWinner: {result.winner}")
    print(f"Runner-up: {result.runner_up}")
    print(f"Third place teams: {', '.join(result.third_place)}") 
    print("=" * 55)

if __name__ == "__main__":
    df = load_and_prepare(Path("data/results.csv"))
    model = PoissonModel().fit(df)
    simulator = MatchSimulator(model)

    tournament = TournamentSimulator(simulator)
    
    print("\n>>> RANDOM SIMULATION (sampled outcome)")
    result = tournament.simulate()

    print(f"\nWinner: {result.winner}")
    print(f"Runner-up: {result.runner_up}")
    print(f"Third place teams: {', '.join(result.third_place)}")

    print_group_standings(result.group_standings)
    print_knockout_results(result.knockout_results)

    print("\n>>> PREDICTED BRACKET")
    prediction = tournament.predict_tournament()
    print_group_standings(prediction.group_standings)
    print_bracket(prediction)




