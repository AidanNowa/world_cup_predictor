data_loader.py -- filter and weight recent matches

poisson_model.py -- fit the model and validate it's in the right ballpark

match_simulator.py -- simulate individual matches

tournament.py -- setup hardcoded matchups for the 2026 World Cup groups

monte_carlo.py -- run simulations, output a probability table

current predictions (5/31/26)

--------------------------------------------------------------------------------------

 2026 FIFA World Cup Monte Carlo Results  (10,000 simulations)
 

Rank  Team                      Grp    Win%   Final%   Semi%   Advance%  1st in Group%


1     Spain                       H   21.07    32.21   45.45      99.24          82.92

2     England                     L   12.92    21.23   33.05      96.83          72.77

3     Argentina                   J    9.84    17.03   30.90      93.78          67.67

4     Germany                     E    8.11    16.39   30.93      94.39          59.03

5     Morocco                     C    6.02    12.11   22.93      92.24          47.38

6     France                      I    5.76    11.58   21.25      84.62          40.19

7     Norway                      I    5.36    11.69   22.59      86.81          45.62

8     Ecuador                     E    4.46     9.51   18.71      82.11          28.36

9     Portugal                    K    3.74     7.97   17.11      84.03          49.86

10    Netherlands                 F    3.72     8.33   17.13      84.84          46.89

11    Switzerland                 B    3.66     7.81   17.39      96.24          72.06

12    Japan                       F    2.70     6.18   13.63      80.32          37.31

13    Brazil                      C    2.54     6.29   14.49      88.88          37.34

14    Colombia                    K    1.60     3.77    9.62      74.25          35.18

15    Croatia                     L    1.19     3.35    7.87      80.98          23.26

16    Uruguay                     H    1.17     3.14    7.87      78.88          14.99

17    Belgium                     G    0.86     2.43    6.80      85.17          53.40

18    Senegal                     I    0.74     2.18    6.01      51.63          12.71

19    Austria                     J    0.74     2.39    6.60      68.39          19.50

20    Turkey                      D    0.60     2.12    6.21      70.30          34.52

--------------------------------------------------------------------------------------


------------------------------------------------------------
Group Stage Probabilities (10,000 simulations)
------------------------------------------------------------

   -------Group A--------
   
   Team   1st%   Advance%

   Mexico  34.79      69.84
   
   South Korea  27.89      63.00
   
   Czech Republic  24.64      60.02
   
   South Africa  12.68      38.85

   -------Group B--------
   
   Team   1st%   Advance%
   
   Switzerland  72.06      96.24
   
   Canada  19.42      71.74
   
   Bosnia and Herzegovina   6.96      40.58
   
   Qatar   1.56      13.22

   -------Group C--------
   
   Team   1st%   Advance%
   
   Morocco  47.38      92.24
   
   Brazil  37.34      88.88
   
   Scotland  15.18      67.66
   
   Haiti   0.10       1.05

   -------Group D--------
   
   Team   1st%   Advance%
   
   Turkey  34.52      70.30
   
   Australia  27.19      62.81
   
   Paraguay  24.85      60.87
   
   United States  13.44      39.60

   -------Group E--------
   
   Team   1st%   Advance%

   Germany  59.03      94.39
   
   Ecuador  28.36      82.11
   
   Ivory Coast  12.19      57.02
   
   Curaçao   0.42       5.54

   -------Group F--------
   
   Team   1st%   Advance%

   Netherlands  46.89      84.84
   
   Japan  37.31      80.32
   
   Tunisia   8.59      37.17
   
   Sweden   7.21      29.64

   -------Group G--------
   
   Team   1st%   Advance%
   

   Belgium  53.40      85.17
   
   Iran  23.61      62.29
   
   Egypt  17.62      53.95
   
   New Zealand   5.37      22.93

   -------Group H--------
   
   Team   1st%   Advance%
 
   Spain  82.92      99.24
   
   Uruguay  14.99      78.88
   
   Saudi Arabia   1.45      23.07
   
   Cape Verde   0.64      14.32

   -------Group I--------
   
   Team   1st%   Advance%

   Norway  45.62      86.81
   
   France  40.19      84.62
   
   Senegal  12.71      51.63
   
   Iraq   1.48      10.15

   -------Group J--------
   
   Team   1st%   Advance%

   Argentina  67.67      93.78
   
   Austria  19.50      68.39
   
   Algeria   8.74      41.05
   
   Jordan   4.09      22.18

   -------Group K--------
   
   Team   1st%   Advance%

   Portugal  49.86      84.03
   
   Colombia  35.18      74.25
   
   Uzbekistan   7.96      31.88
   
   DR Congo   7.00      28.19

   -------Group L--------
   
   Team   1st%   Advance%

   England  72.77      96.83

   Croatia  23.26      80.98
   
   Ghana   2.98      23.35
   
   Panama   0.99      10.33




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

  Uzbekistan               3  0  1  2   3   5   -2    1

  DR Congo                 3  0  1  2   3   5   -2    1



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

   Germany 2 - 1 Paraguay  -> Germany

   Netherlands 2 - 1 Brazil  -> Netherlands

   Morocco 2 - 1 Japan  -> Morocco

   France 2 - 1 Paraguay  -> France

   Ecuador 1 - 2 Norway  -> Norway

   Mexico 2 - 1 Ivory Coast  -> Mexico

   England 1 - 0 Ivory Coast  -> England

   Turkey 2 - 1 Ivory Coast  -> Turkey

   Belgium 2 - 1 Ivory Coast  -> Belgium

   Colombia 2 - 1 Croatia  -> Colombia

   Spain 2 - 1 Austria  -> Spain

   Switzerland 2 - 1 Ivory Coast  -> Switzerland

   Argentina 2 - 1 Uruguay  -> Argentina

   Portugal 2 - 1 Paraguay  -> Portugal

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

=======================================================

