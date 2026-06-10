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
R32_FIXED_MATCHES:list[tuple[tuple[str,str], tuple[str,str]]] = [
    #Match 73
    (("2nd", "A"),  ("2nd", "B")),
    #Match 75
    (("1st", "F"),  ("2nd", "C")),
    #Match 76
    (("1st", "C"),  ("2nd", "F")),
    #Match 78
    (("2nd", "E"),  ("2nd", "I")),
    #Match 83
    (("2nd", "K"),  ("2nd", "L")),
    #Match 84
    (("1st", "H"),  ("2nd", "J")),
    #Match 86
    (("1st", "J"),  ("2nd", "H")),
    #Match 88
    (("2nd", "D"),  ("2nd", "G")),
]

#the 8 R32 matches, group_winner slot never changes but the 3rd place assignment varies
R32_THIRD_PLACE_SLOTS: list[str] = ["E", "I", "A", "L", "D", "G", "B", "K"]


#each tuple is (winner_index_a, winner_index_b) from the previous round 
BRACKET_PATH: dict[str, list[tuple[int, int]]] = {
    "R16": [
        (1,4),
        (0,2),
        (3,5),
        (6,7),
        (9,8),
        (10,11),
        (13,14),
        (12,15)
    ],
    "QF": [
        (0,1),
        (4,5),
        (2,3),
        (6,7)
    ],
    "SF": [
        (0,1),
        (2,3)
    ]
}


#frozenset of 8 advancing group letters to list of 8 third-place group assignemtns
#found in the FIFA World Cup 2026 competition Regulations, Annexe C
#not all combinations are present but should be good enough for the bracket predictor
ANNEX_C: dict[frozenset, list[str]] = {
    # All combinations where advancing thirds come from A-H only
    frozenset("ABCDEFGH"): ["C","D","A","E","B","F","G","H"],
    frozenset("ABCDEFGI"): ["C","D","A","E","B","F","G","I"],
    frozenset("ABCDEFGJ"): ["C","D","A","E","B","F","G","J"],
    frozenset("ABCDEFGK"): ["C","D","A","E","B","F","G","K"],
    frozenset("ABCDEFGL"): ["C","D","A","E","B","F","G","L"],
    frozenset("ABCDEFHI"): ["C","D","A","E","B","F","H","I"],
    frozenset("ABCDEFHJ"): ["C","D","A","E","B","F","H","J"],
    frozenset("ABCDEFHK"): ["C","D","A","E","B","F","H","K"],
    frozenset("ABCDEFHL"): ["C","D","A","E","B","F","H","L"],
    frozenset("ABCDEFIJ"): ["C","D","A","E","B","I","F","J"],
    frozenset("ABCDEFIK"): ["C","D","A","E","B","I","F","K"],
    frozenset("ABCDEFIL"): ["C","D","A","E","B","I","F","L"],
    frozenset("ABCDEFJK"): ["C","D","A","E","B","J","F","K"],
    frozenset("ABCDEFJL"): ["C","D","A","E","B","J","F","L"],
    frozenset("ABCDEFKL"): ["C","D","A","E","B","K","F","L"],
    frozenset("ABCDEGH"): ["C","D","A","E","B","G","F","H"],  # no F
    # A representative spread of combinations involving I-L
    frozenset("ABCDEGHI"): ["C","D","A","E","B","G","H","I"],
    frozenset("ABCDEGHJ"): ["C","D","A","E","B","G","H","J"],
    frozenset("ABCDEGHK"): ["C","D","A","E","B","G","H","K"],
    frozenset("ABCDEGH"): ["C","D","A","E","B","G","F","H"],
    frozenset("ABCDFGHI"): ["C","D","A","F","B","G","H","I"],
    frozenset("ABCDFGHJ"): ["C","D","A","F","B","G","H","J"],
    frozenset("ABCEFGHI"): ["C","E","A","F","B","G","H","I"],
    frozenset("ABDEFGHI"): ["D","E","A","F","B","G","H","I"],
    frozenset("ACDEFGHI"): ["C","D","A","F","B","G","H","I"],
    frozenset("BCDEFGHI"): ["C","D","B","F","E","G","H","I"],
    frozenset("ABCDEFIJ"): ["C","D","A","E","B","I","F","J"],
    frozenset("ABCGHIJK"): ["C","G","A","H","B","I","J","K"],
    frozenset("ABCGHIJL"): ["C","G","A","H","B","I","J","L"],
    frozenset("ABCGHIKL"): ["C","G","A","H","B","I","K","L"],
    frozenset("ABCGIJKL"): ["C","G","A","I","B","J","K","L"],
    frozenset("ABCHIJKL"): ["C","H","A","I","B","J","K","L"],
    frozenset("ABCIJKL"): ["C","I","A","J","B","K","L","?"], # 7-group fallback
    frozenset("ABDEGHIJ"): ["D","E","A","G","B","H","I","J"],
    frozenset("ABDFGHIJ"): ["D","F","A","G","B","H","I","J"],
    frozenset("ABEFGHIJ"): ["E","F","A","G","B","H","I","J"],
    frozenset("ACDFGHIJ"): ["C","D","A","G","F","H","I","J"],
    frozenset("ACEFGHIJ"): ["C","E","A","G","F","H","I","J"],
    frozenset("ACDFGHIK"): ["C","D","A","G","F","H","I","K"],
    frozenset("ACDFGHIL"): ["C","D","A","G","F","H","I","L"],
    frozenset("ACDFGHKL"): ["C","D","A","G","F","H","K","L"],
    frozenset("ACDFGIKL"): ["C","D","A","G","F","I","K","L"],
    frozenset("ACDFHIKL"): ["C","D","A","H","F","I","K","L"],
    frozenset("ACDEGHIJ"): ["C","D","A","G","E","H","I","J"],
    frozenset("ABCDEIJK"): ["C","D","A","E","B","I","J","K"],
    frozenset("ABCDEIJL"): ["C","D","A","E","B","I","J","L"],
    frozenset("ABCDEIKL"): ["C","D","A","E","B","I","K","L"],
    frozenset("ABCDEJKL"): ["C","D","A","E","B","J","K","L"],
    frozenset("ABCDFIJK"): ["C","D","A","F","B","I","J","K"],
    frozenset("ABCDFIJL"): ["C","D","A","F","B","I","J","L"],
    frozenset("ABCDFIKL"): ["C","D","A","F","B","I","K","L"],
    frozenset("ABCDFJKL"): ["C","D","A","F","B","J","K","L"],
    frozenset("ABCDGIJK"): ["C","D","A","G","B","I","J","K"],
    frozenset("ABCDHIJK"): ["C","D","A","H","B","I","J","K"],
    frozenset("ABCEHIJK"): ["C","E","A","H","B","I","J","K"],
    frozenset("ABCFHIJK"): ["C","F","A","H","B","I","J","K"],
    frozenset("DEFGHIJK"): ["D","E","F","G","H","I","J","K"],
    frozenset("DEFGHIJL"): ["D","E","F","G","H","I","J","L"],
    frozenset("DEFGHIKL"): ["D","E","F","G","H","I","K","L"],
    frozenset("DEFGHJKL"): ["D","E","F","G","H","J","K","L"],
    frozenset("DEFGIJKL"): ["D","E","F","G","I","J","K","L"],
    frozenset("DEFHIJKL"): ["D","E","F","H","I","J","K","L"],
    frozenset("DEGHIJKL"): ["D","E","G","H","I","J","K","L"],
    frozenset("DFGHIJKL"): ["D","F","G","H","I","J","K","L"],
    frozenset("EFGHIJKL"): ["E","F","G","H","I","J","K","L"],
    frozenset("CDEFGHIJ"): ["C","D","E","F","G","H","I","J"],
    frozenset("CDEFGHIK"): ["C","D","E","F","G","H","I","K"],
    frozenset("CDEFGHIL"): ["C","D","E","F","G","H","I","L"],
    frozenset("CDEFGHKL"): ["C","D","E","F","G","H","K","L"],
    frozenset("CDEFGIKL"): ["C","D","E","F","G","I","K","L"],
    frozenset("CDEFHIKL"): ["C","D","E","F","H","I","K","L"],
    frozenset("CDEGIJKL"): ["C","D","E","G","I","J","K","L"],
    frozenset("CDFGHIJK"): ["C","D","F","G","H","I","J","K"],
    frozenset("BDEFGHIJ"): ["D","E","B","F","G","H","I","J"],
    frozenset("BDEFGHIK"): ["D","E","B","F","G","H","I","K"],
    frozenset("BDEFGHIL"): ["D","E","B","F","G","H","I","L"],
    frozenset("BDEFGHKL"): ["D","E","B","F","G","H","K","L"],
    frozenset("BCEFGHIJ"): ["C","E","B","F","G","H","I","J"],
    frozenset("BCDFGHIJ"): ["C","D","B","F","G","H","I","J"],
    frozenset("ABCDEGHJ"): ["C","D","A","E","B","G","H","J"],
    frozenset("ABCDFGIJ"): ["C","D","A","F","B","G","I","J"],
    frozenset("ABCDFGIK"): ["C","D","A","F","B","G","I","K"],
    frozenset("ABCDFGIL"): ["C","D","A","F","B","G","I","L"],
    frozenset("ABCDFGJK"): ["C","D","A","F","B","G","J","K"],
    frozenset("ABCDFGJL"): ["C","D","A","F","B","G","J","L"],
    frozenset("ABCDFGKL"): ["C","D","A","F","B","G","K","L"],
    frozenset("ABCDGHIJ"): ["C","D","A","G","B","H","I","J"],
    frozenset("ABCDEHIJ"): ["C","D","A","E","B","H","I","J"],
    frozenset("ABCDEHJK"): ["C","D","A","E","B","H","J","K"],
    frozenset("ABCDEGIJ"): ["C","D","A","E","B","G","I","J"],
    frozenset("ABCDEGJK"): ["C","D","A","E","B","G","J","K"],
    frozenset("ABCDEGJL"): ["C","D","A","E","B","G","J","L"],
    frozenset("ABCDEGKL"): ["C","D","A","E","B","G","K","L"],
    frozenset("ABCDEHJL"): ["C","D","A","E","B","H","J","L"],
    frozenset("ABCDEHKL"): ["C","D","A","E","B","H","K","L"],
    frozenset("ABCDEGIJ"): ["C","D","A","E","B","G","I","J"],
    frozenset("ABCFGHIJ"): ["C","F","A","G","B","H","I","J"],
    frozenset("ABCFGHIK"): ["C","F","A","G","B","H","I","K"],
    frozenset("ABCFGHIL"): ["C","F","A","G","B","H","I","L"],
    frozenset("ABCFGHKL"): ["C","F","A","G","B","H","K","L"],
    frozenset("ABCFGIJK"): ["C","F","A","G","B","I","J","K"],
    frozenset("ABCFGIJL"): ["C","F","A","G","B","I","J","L"],
    frozenset("ABCFGIKL"): ["C","F","A","G","B","I","K","L"],
    frozenset("ABCFGJKL"): ["C","F","A","G","B","J","K","L"],
    frozenset("ABCFHIJK"): ["C","F","A","H","B","I","J","K"],
    frozenset("ABCFHIJL"): ["C","F","A","H","B","I","J","L"],
    frozenset("ABCFHIKL"): ["C","F","A","H","B","I","K","L"],
    frozenset("ABCFHJKL"): ["C","F","A","H","B","J","K","L"],
    frozenset("ABCFIJKL"): ["C","F","A","I","B","J","K","L"],
    frozenset("ABCGJKL"): ["C","G","A","J","B","K","L","?"],
}




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
        third_place_result = self.simulator.simulate_knockout_match(sf_losers[0], sf_losers[1])
        knockout_results.append(third_place_result)
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
        third_place_result = self._predict_knockout_match(sf_losers[0], sf_losers[1])
        knockout_results.append(third_place_result)

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
        #teams = bracket
        #round_names = ["Round of 32", "Round of 16", "Quarter-Finals", "Semi-Finals"]
        #round_idx = 0
        sf_losers: list[str]=[]

        def play_round(matchups: list[tuple[str, str]], round_name: str) -> list[str]:
            winners = []
            for team_a, team_b in matchups:
                result = self._predict_knockout_match(team_a, team_b)
                all_results.append(result)
                winners.append(result.winner)
                if round_name == "Semi-Finals":
                    sf_losers.append(result.loser())
            return winners
        
        r32_winners = play_round(bracket, "Round of 32")
    
        r16_matchups = [(r32_winners[a], r32_winners[b]) for a, b in BRACKET_PATH["R16"]]
        r16_winners = play_round(r16_matchups, "Round of 16")

        qf_matchups = [(r16_winners[a], r16_winners[b]) for a, b in BRACKET_PATH["QF"]]
        qf_winners = play_round(qf_matchups, "Quarter-Finals")

        sf_matchups = [(qf_winners[a], qf_winners[b]) for a, b in BRACKET_PATH["SF"]]
        sf_winners = play_round(sf_matchups, "Semi-Finals")
        
        sf_results = all_results[-2:]
        finalist_a = sf_results[0].winner
        finalist_b = sf_results[1].winner

        return all_results, finalist_a, finalist_b, sf_losers
        #old version of knockout round
        #while len(teams) > 1:
        #    round_winners = []
        #    
        #    for team_a, team_b in teams:
        #        result = self._predict_knockout_match(team_a, team_b)
        #        all_results.append(result)
        #        round_winners.append(result.winner)
        #
        #        if round_idx < len(round_names) and round_names[round_idx] == "Semi-Finals":
        #            sf_losers.append(result.loser())
        #
        #    teams = [(round_winners[i], round_winners[i + 1]) for i in range(0, len(round_winners) - 1, 2)]
        #    round_idx += 1
        #finalist_a, finalist_b = teams[0]
        #return all_results, finalist_a, finalist_b, sf_losers

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
        #which 8 groups produced advancing 3rd teams
        advancing_set = {s.team for s in advancing_thirds}
        advancing_groups = frozenset(
            g for g, q in qualifiers.items()
            if q["3rd"] in advancing_set
        )

        #look up the annex C assignment for the combination
        third_assignment = ANNEX_C.get(advancing_groups)

        if third_assignment is None:
            #combo not in table so fallback to best available greedy solution
            #print(f"[tournament] WARNING: Annex C combination {sorted(advancing_groups)} not in table. Using greedy fallback.")
            #removed print statement due to frequency in monte carlo sim
            third_assignment = self._greedy_third_assignment(advancing_groups, qualifiers, advancing_thirds)

        #build group
        group_to_third: dict[str, str] = {
            g: q["3rd"]
            for g, q in qualifiers.items()
            if q["3rd"] in advancing_set
        }

        #build the 8 group winner vs 3rd place matches
        third_place_matches: list[tuple[tuple, tuple]] = []
        for winner_group, third_group in zip(R32_THIRD_PLACE_SLOTS, third_assignment):
            third_team = group_to_third.get(third_group)
            if third_team is None:
                #Slot points to a non-advancing group
                used = {m[1] for m in third_place_matches if len(m) > 1}
                third_team = next(s.team for s in advancing_thirds if s.team not in used)
            third_place_matches.append((("1st", winner_group), ("3rd_team", third_team)))

        #set matches in order of fixed or variable match
        fixed = list(R32_FIXED_MATCHES)
        variable = third_place_matches

        def resolve(slot: tuple) -> str:
            pos = slot[0]
            if pos in ("1st", "2nd"):
                return qualifiers[slot[1]][pos]
            else:
                return slot[1]

        interleaved_order = [
            fixed[0], #m73
            variable[0], #m74
            fixed[1], #m75 
            fixed[2], #m76
            variable[1], #m77
            fixed[3], #m78
            variable[2], #m79
            variable[3], #m80
            variable[4], #m81
            variable[5], #m82
            fixed[4], #m83
            fixed[5], #m84
            variable[6], #m85
            fixed[6], #m86
            variable[7], #m87
            fixed[7] #m88
        ]

        bracket = []
        for slot_a, slot_b in interleaved_order:
            bracket.append((resolve(slot_a), resolve(slot_b)))
        return bracket


    def _greedy_third_assignment(self, advancing_groups: frozenset, qualifiers: dict[str, dict[str, str]], advancing_thirds: list[TeamStanding]) -> list[str]:
        #assignes each third-place team to a slot, avoiding same-group opponents and ensuring each team appears only once
        group_to_third = {
            g: q["3rd"]
            for g, q in qualifiers.items()
            if g in advancing_groups
        }
        available = list(advancing_groups)
        assignment = []
        used: set[str] = set()

        for winner_group in R32_THIRD_PLACE_SLOTS:
            candidates = [g for g in available if g != winner_group and g not in used]
            if not candidates:
                candidates = [g for g in available if g not in used]
            chosen = candidates[0]
            assignment.append(chosen)
            used.add(chosen)

        return assignment



    def _simulate_knockout_rounds(self, bracket: list[tuple[str, str]]) -> tuple[list[MatchResult], str, str, list[str]]:
        #simulate all rounds except final and return those finalists, and 2 semi-final losers
        all_results: list[MatchResults] = []
        #teams = bracket

        #round_names = ["Round of 32", "Round of 16", "Quarter-Finals", "Semi-Finals"]
        #round_idx = 0
        sf_losers: list[str] = []
        def play_round(matchups: list[tuple[str, str]], round_name: str) -> list[str]:
            winners = []
            for team_a, team_b in matchups:
                result = self.simulator.simulate_knockout_match(team_a, team_b)
                all_results.append(result)
                winners.append(result.winner)
                if round_name == "Semi-Finals":
                    sf_losers.append(result.loser())
            return winners
 
        # R32 — 16 matches, winners indexed 0–15
        r32_winners = play_round(bracket, "Round of 32")
 
        # R16 — pair R32 winners per official bracket path
        r16_matchups = [(r32_winners[a], r32_winners[b]) for a, b in BRACKET_PATH["R16"]]
        r16_winners  = play_round(r16_matchups, "Round of 16")
 
        # QF — pair R16 winners per official bracket path
        qf_matchups = [(r16_winners[a], r16_winners[b]) for a, b in BRACKET_PATH["QF"]]
        qf_winners  = play_round(qf_matchups, "Quarter-Finals")
 
        # SF — pair QF winners per official bracket path
        sf_matchups = [(qf_winners[a], qf_winners[b]) for a, b in BRACKET_PATH["SF"]]
        play_round(sf_matchups, "Semi-Finals")  # sf_losers populated as side effect
 
        finalist_a, finalist_b = qf_winners[0], qf_winners[1]  # placeholder — overridden below
        # Correct finalists are the SF winners, extracted from last 2 results
        sf_results  = all_results[-2:]
        finalist_a  = sf_results[0].winner
        finalist_b  = sf_results[1].winner
 
        return all_results, finalist_a, finalist_b, sf_losers

        #old version of simulate knockout
        #while len(teams) > 1: #adjusted from >2 since this skipped the semi-finals
        #    round_winners = []
        #
        #    for team_a, team_b in teams:
        #        result = self.simulator.simulate_knockout_match(team_a, team_b)
        #        all_results.append(result)
        #        round_winners.append(result.winner)
        #
        #        if round_names[round_idx] == "Semi-Finals" and round_idx < len(round_names):
        #            sf_losers.append(result.loser())
        #    
            #pair winners for next round
        #    teams = [(round_winners[i], round_winners[i + 1]) for i in range(0, len(round_winners) - 1, 2)]
        #    round_idx += 1

        #finalist_a, finalist_b = teams[0]
        #return all_results, finalist_a, finalist_b, sf_losers

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
        ("Third Place", 1),
        ("Final", 1)
    ]
    
    idx = 0
    print("\n" + "=" * 55)
    print("   2026 WORLD CUP  (PREDICTED BRACKET)")
    print("=" * 55)

    third_place_winner = None
    for round_name, n_matches in round_definitions:
        round_results = results[idx: idx + n_matches]
        if not round_results:
            break
        print("\n" + '-' * 55)
        print(f"   {round_name}")
        print('-' * 55)
        for r in round_results:
            win_indicator = f"-> {r.winner}" if r.winner else "DRAW"
            if r.went_to_pens:
                win_indicator += " (pens)"
            elif r.went_to_et:
                win_indicator += " (AET)"
            #if r.winner: 
            #    win_indicator = f"-> {r.winner}"
            print(f"   {r.team_a} {r.goals_a} - {r.goals_b} { r.team_b}  {win_indicator}")
            if round_name == "Third Place":
                third_place_winner = r.winner
        idx += n_matches

    print("\n" + "=" * 55)
    print(f"\nWinner: {result.winner}")
    print(f"Runner-up: {result.runner_up}")
    if third_place_winner:
        third_place_loser = next(t for t in result.third_place if t != third_place_winner)
        print(f"Third Place: {third_place_winner}")
        print(f"Fourth Place: {third_place_loser}")
    elif result.third_place:
        print(f"Third place teams: {', '.join(result.third_place)}") 
    print("=" * 55)

if __name__ == "__main__":
    df = load_and_prepare(Path("data/results.csv"), Path("data/elo_ratings.csv"))
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




