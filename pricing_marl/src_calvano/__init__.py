"""
src_calvano: engine for exp09, the Calvano-ladder mechanism decomposition.

Spans a 2x2 factorial (state in {full price vector, market minimum} x actions
in {all prices, the 3 repricer rules}) plus a fifth commitment rung at K = 30,
so the collusion gain can be attributed to state reduction, action reduction,
their interaction, and commitment.

See Extention_Plan.md section 4 for the full design and decision log.
"""
