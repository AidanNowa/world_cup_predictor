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
--------------------------------------------------------------------------------------
 2026 FIFA World Cup Monte Carlo Results  (10,000 simulations)
--------------------------------------------------------------------------------------
Rank  Team                      Grp    Win%   Final%   Semi%   Advance%  1st in Group%
--------------------------------------------------------------------------------------
1     Spain                       H   22.76    33.83   46.57      99.72          83.06
2     England                     L   14.49    24.17   37.59      98.80          71.44
3     Argentina                   J    9.90    16.97   29.65      96.85          68.02
4     Germany                     E    6.88    13.79   25.11      97.67          59.06
5     Morocco                     C    5.58    11.25   21.59      97.62          46.52
6     Norway                      I    5.08    10.68   18.94      93.00          45.13
7     Portugal                    K    4.69     9.93   20.62      90.76          51.67
8     France                      I    4.32     9.65   18.20      91.49          41.30
9     Switzerland                 B    3.71     8.40   19.29      98.55          73.68
10    Netherlands                 F    3.29     7.77   16.34      90.44          46.44
11    Ecuador                     E    3.24     7.26   14.97      90.36          28.29
12    Japan                       F    2.56     6.03   12.88      86.48          37.53
13    Brazil                      C    2.47     6.06   13.50      97.14          38.18
14    Colombia                    K    1.88     4.69   11.45      83.64          33.61
15    Croatia                     L    1.34     4.12   10.40      91.10          24.64
16    Uruguay                     H    1.26     3.14    8.85      86.57          14.89
17    Belgium                     G    1.22     3.04    8.39      91.01          53.29
18    Turkey                      D    0.81     2.27    6.69      77.60          34.01
19    Austria                     J    0.64     1.87    5.98      78.09          19.41
20    Senegal                     I    0.62     1.81    4.54      67.63          12.21
--------------------------------------------------------------------------------------


------------------------------------------------------------
Group Stage Probabilities (10,000 simulations)
------------------------------------------------------------

   -------Group A--------
   Team   1st%   Advance%
   --------------------------------------------
   Mexico  33.28      79.39
   South Korea  27.60      73.58
   Czech Republic  26.01      71.16
   South Africa  13.11      50.34

   -------Group B--------
   Team   1st%   Advance%
   --------------------------------------------
   Switzerland  73.68      98.55
   Canada  18.54      83.27
   Bosnia and Herzegovina   6.15      58.28
   Qatar   1.63      23.08

   -------Group C--------
   Team   1st%   Advance%
   --------------------------------------------
   Morocco  46.52      97.62
   Brazil  38.18      97.14
   Scotland  15.20      87.60
   Haiti   0.10       2.76

   -------Group D--------
   Team   1st%   Advance%
   --------------------------------------------
   Turkey  34.01      77.60
   Australia  27.30      73.15
   Paraguay  25.45      72.17
   United States  13.24      50.92

   -------Group E--------
   Team   1st%   Advance%
   --------------------------------------------
   Germany  59.06      97.67
   Ecuador  28.29      90.36
   Ivory Coast  12.13      72.27
   Curaçao   0.52       9.13

   -------Group F--------
   Team   1st%   Advance%
   --------------------------------------------
   Netherlands  46.44      90.44
   Japan  37.53      86.48
   Tunisia   8.95      49.92
   Sweden   7.08      41.17

   -------Group G--------
   Team   1st%   Advance%
   --------------------------------------------
   Belgium  53.29      91.01
   Iran  23.80      75.41
   Egypt  17.63      67.82
   New Zealand   5.28      33.12

   -------Group H--------
   Team   1st%   Advance%
   --------------------------------------------
   Spain  83.06      99.72
   Uruguay  14.89      86.57
   Saudi Arabia   1.29      36.45
   Cape Verde   0.76      23.72

   -------Group I--------
   Team   1st%   Advance%
   --------------------------------------------
   Norway  45.13      93.00
   France  41.30      91.49
   Senegal  12.21      67.63
   Iraq   1.36      16.70

   -------Group J--------
   Team   1st%   Advance%
   --------------------------------------------
   Argentina  68.02      96.85
   Austria  19.41      78.09
   Algeria   8.78      56.03
   Jordan   3.79      34.08

   -------Group K--------
   Team   1st%   Advance%
   --------------------------------------------
   Portugal  51.67      90.76
   Colombia  33.61      83.64
   Uzbekistan   7.81      47.06
   DR Congo   6.91      41.90

   -------Group L--------
   Team   1st%   Advance%
   --------------------------------------------
   England  71.44      98.80
   Croatia  24.64      91.10
   Ghana   3.00      43.04
   Panama   0.92      21.56


>>> PREDICTED BRACKET

--- Group A ---
Team                       P  W  D  L  GF  GA   GD  Pts
-------------------------------------------------------
✓ Mexico                   3  3  0  0   6   3    3    9
✓ South Korea              3  2  0  1   5   4    1    6
  Czech Republic           3  1  0  2   4   5   -1    3
  South Africa             3  0  0  3   3   6   -3    0

--- Group B ---
Team                       P  W  D  L  GF  GA   GD  Pts
-------------------------------------------------------
✓ Switzerland              3  3  0  0   8   3    5    9
✓ Canada                   3  2  0  1   5   4    1    6
  Bosnia and Herzegovina   3  1  0  2   4   5   -1    3
  Qatar                    3  0  0  3   3   8   -5    0

--- Group C ---
Team                       P  W  D  L  GF  GA   GD  Pts
-------------------------------------------------------
✓ Morocco                  3  3  0  0   7   2    5    9
✓ Brazil                   3  2  0  1   7   4    3    6
  Scotland                 3  1  0  2   5   5    0    3
  Haiti                    3  0  0  3   2  10   -8    0

--- Group D ---
Team                       P  W  D  L  GF  GA   GD  Pts
-------------------------------------------------------
✓ Turkey                   3  3  0  0   6   3    3    9
✓ Australia                3  1  1  1   4   4    0    4
  Paraguay                 3  1  1  1   4   4    0    4
  United States            3  0  0  3   3   6   -3    0

--- Group E ---
Team                       P  W  D  L  GF  GA   GD  Pts
-------------------------------------------------------
✓ Germany                  3  2  1  0   7   2    5    7
✓ Ecuador                  3  1  2  0   3   1    2    5
  Ivory Coast              3  1  1  1   2   2    0    4
  Curaçao                  3  0  0  3   0   7   -7    0

--- Group F ---
Team                       P  W  D  L  GF  GA   GD  Pts
-------------------------------------------------------
✓ Netherlands              3  3  0  0   7   3    4    9
✓ Japan                    3  2  0  1   5   4    1    6
  Tunisia                  3  1  0  2   4   5   -1    3
  Sweden                   3  0  0  3   3   7   -4    0

--- Group G ---
Team                       P  W  D  L  GF  GA   GD  Pts
-------------------------------------------------------
✓ Belgium                  3  3  0  0   6   3    3    9
✓ Iran                     3  2  0  1   5   4    1    6
  Egypt                    3  1  0  2   4   5   -1    3
  New Zealand              3  0  0  3   3   6   -3    0

--- Group H ---
Team                       P  W  D  L  GF  GA   GD  Pts
-------------------------------------------------------
✓ Spain                    3  3  0  0   9   1    8    9
✓ Uruguay                  3  2  0  1   4   2    2    6
  Saudi Arabia             3  1  0  2   2   5   -3    3
  Cape Verde               3  0  0  3   1   8   -7    0

--- Group I ---
Team                       P  W  D  L  GF  GA   GD  Pts
-------------------------------------------------------
✓ France                   3  3  0  0   6   2    4    9
✓ Norway                   3  2  0  1   6   4    2    6
  Senegal                  3  1  0  2   4   5   -1    3
  Iraq                     3  0  0  3   2   7   -5    0

--- Group J ---
Team                       P  W  D  L  GF  GA   GD  Pts
-------------------------------------------------------
✓ Argentina                3  3  0  0   6   3    3    9
✓ Austria                  3  2  0  1   5   4    1    6
  Algeria                  3  1  0  2   4   5   -1    3
  Jordan                   3  0  0  3   3   6   -3    0

--- Group K ---
Team                       P  W  D  L  GF  GA   GD  Pts
-------------------------------------------------------
✓ Portugal                 3  3  0  0   6   3    3    9
✓ Colombia                 3  2  0  1   5   4    1    6
  DR Congo                 3  0  1  2   3   5   -2    1
  Uzbekistan               3  0  1  2   3   5   -2    1

--- Group L ---
Team                       P  W  D  L  GF  GA   GD  Pts
-------------------------------------------------------
✓ England                  3  3  0  0   7   1    6    9
✓ Croatia                  3  2  0  1   5   4    1    6
  Ghana                    3  1  0  2   3   5   -2    3
  Panama                   3  0  0  3   2   7   -5    0

=======================================================
   2026 WORLD CUP  (PREDICTED BRACKET)
=======================================================

-------------------------------------------------------
   Round of 32
-------------------------------------------------------
   South Korea 1 - 2 Canada  -> Canada
   Germany 2 - 1 Scotland  -> Germany
   Netherlands 2 - 1 Brazil  -> Netherlands
   Morocco 2 - 1 Japan  -> Morocco
   France 2 - 1 Paraguay  -> France
   Ecuador 1 - 2 Norway  -> Norway
   Mexico 2 - 1 Czech Republic  -> Mexico
   England 1 - 0 Ivory Coast  -> England
   Turkey 2 - 1 Bosnia and Herzegovina  -> Turkey
   Belgium 2 - 1 Tunisia  -> Belgium
   Colombia 2 - 1 Croatia  -> Colombia
   Spain 2 - 1 Austria  -> Spain
   Switzerland 2 - 1 Egypt  -> Switzerland
   Argentina 2 - 1 Uruguay  -> Argentina
   Portugal 2 - 1 Senegal  -> Portugal
   Australia 2 - 1 Iran  -> Australia

-------------------------------------------------------
   Round of 16
-------------------------------------------------------
   Canada 1 - 2 Germany  -> Germany
   Netherlands 1 - 2 Morocco  -> Morocco
   France 2 - 1 Norway  -> France
   Mexico 0 - 2 England  -> England
   Turkey 1 - 2 Belgium  -> Belgium
   Colombia 1 - 2 Spain  -> Spain
   Switzerland 1 - 2 Argentina  -> Argentina
   Portugal 2 - 1 Australia  -> Portugal

-------------------------------------------------------
   Quarter-Finals
-------------------------------------------------------
   Germany 1 - 2 Morocco  -> Morocco
   France 1 - 2 England  -> England
   Belgium 1 - 2 Spain  -> Spain
   Argentina 2 - 1 Portugal  -> Argentina

-------------------------------------------------------
   Semi-Finals
-------------------------------------------------------
   Morocco 1 - 2 England  -> England
   Spain 2 - 1 Argentina  -> Spain

-------------------------------------------------------
   Final
-------------------------------------------------------
   England 1 - 2 Spain  -> Spain

=======================================================

Winner: Spain
Runner-up: England
Third place teams: Morocco, Argentina
