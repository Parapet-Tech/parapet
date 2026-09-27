"""P3 benign reference construction: the frozen reference / held-out split and the
helpers every downstream step re-derives from it (side inventories, sample order,
carrier choice within a cell).

These rules are pre-registered. Changing any constant here changes which benign
carriers calibrate the reference, so a change is a re-freeze, not a refactor.
Tests pin the regression values.
"""
