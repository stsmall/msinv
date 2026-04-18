"""Shared constants for the hull test suite.

NEGLIGIBLE_GAMMA satisfies the gamma>0 enforcement (an InversionSpec
with gamma=0 is now rejected at construction time) while firing zero
flux events at typical Ne/L/t_inv combinations used in the test suite.
Use it whenever a test is structured around gamma=0 baseline behavior.
"""

NEGLIGIBLE_GAMMA = 1e-15
