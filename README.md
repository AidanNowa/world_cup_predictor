<<<<<<< HEAD
=======
# FIFA World Cup 2026 — Poisson Match Predictor
 
A machine learning project that uses a **Poisson goals model** and **Monte Carlo simulation** to predict match outcomes and tournament winners for the 2026 FIFA World Cup.
 
---
 
## How It Works
 
### 1. The Poisson Goals Model
 
For any match between Team A and Team B, the model treats goals as independent Poisson random variables:
 
```
Goals_A ~ Poisson(λ_A)    where λ_A = exp(intercept + attack_A + defense_B)
Goals_B ~ Poisson(λ_B)    where λ_B = exp(intercept + attack_B + defense_A)
```
 
Every team gets two fitted parameters:
 
| Parameter  | Meaning                                            |
|------------|----------------------------------------------------|
| `attack`   | Tendency to score (higher = more goals scored) |
| `defense`  | Tendency to concede (lower = fewer goals let in) |
 
These are estimated by **maximum likelihood** over ~5,000+ historical international matches, weighted by both **recency** (recent matches matter more) and **match importance** (World Cup > friendly).
 
### 2. Predicting a Match
 
Given fitted λ values, the model can either:
 
- **Analytically** compute P(win), P(draw), P(loss) by summing the joint Poisson probability matrix
- **Sample** a random scoreline via `numpy.random.poisson` for simulation
### 3. Monte Carlo Tournament Simulation
 
A full 2026 World Cup is simulated end-to-end:
 
```
Group Stage (12 groups × 6 matches)
    ↓  Top 2 per group + best 8 third-place teams advance
Round of 32
    ↓ 
Round of 16
    ↓
Quarter-Finals
    ↓
Semi-Finals
    ↓
  Final
```
 
Running **10,000 full simulations** produces win probabilities for every team.
 
### 4. The Predicted Bracket
 
Alongside the simulation, the model produces a single **deterministic bracket** for bracket challenge submissions where every match is decided by whichever team has the higher `P(win)` from the Poisson matrix rather than by random sampling.
 
---
 
## Project Structure
 
```
world_cup_predictor/
├── data/
│   └── results.csv          # ~45k international results (martj42/international_results)
├── src/
│   ├── data_loader.py       # Load, filter & weight historical match data
│   ├── poisson_model.py     # Fit attack/defense parameters via MLE
│   ├── match_simulator.py   # Simulate single matches (group + knockout rules)
│   ├── tournament.py        # Full 2026 tournament logic + official bracket
│   └── monte_carlo.py       # Run N simulations, aggregate probabilities
├── results/
│   └── monte_carlo_results.csv
└── README.md
```
 
---
 
## Sample Output
>>>>>>> d3a4614395789113044278faf221af9aa171074d
