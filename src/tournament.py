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
    "A": ["Czechia", "Mexico", "South Africa", "South Korea"],
    "B": ["Bosnia and Herzegovina", "Canada", "Qatar", "Switzerland"],
    "C": ["Brazil", "Haiti", "Morocco", "Scotland"],
    "D": ["Australia", "Paraguay", "Turkey", "United States"],
    "E": ["Curacao", "Ecuador", "Germany", "Ivory Coast"],
    "F": ["Japan", "Netherlands", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Cape Verde", "Saudi Arabia", "Spain", "Uruguay"],
    "I": ["France", "Iraq", "Norway", "Senegal"],
    "J": ["Algeria", "Argentina", "Austria", "Jordan"],
    "K": ["Colombia", "DR Congo", "Portugal", "Uzbekistan"],
    "L": ["Croatia", "England", "Ghana", "Panama"],

}
