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
    "A": ["Mexico",         "South Africa",   "Mali",          "New Zealand"],
    "B": ["Argentina",      "Morocco",         "Iraq",          "Ukraine"],
    "C": ["USA",            "Panama",          "Albania",       "Ukraine"],
    "D": ["France",         "Japan",           "Paraguay",      "Saudi Arabia"],
    "E": ["Spain",          "Netherlands",     "Serbia",        "Ecuador"],
    "F": ["England",        "Senegal",         "Slovakia",      "Uzbekistan"],
    "G": ["Brazil",         "Cameroon",        "Colombia",      "Costa Rica"],
    "H": ["Portugal",       "Belgium",         "Nigeria",       "Egypt"],
    "I": ["Germany",        "Australia",       "Venezuela",     "Bahrain"],
    "J": ["Uruguay",        "South Korea",     "Canada",        "Tunisia"],
    "K": ["Poland",         "Austria",         "Guatemala",     "Comoros"],
    "L": ["Croatia",        "Denmark",         "Bolivia",       "Ethiopia"],

}
