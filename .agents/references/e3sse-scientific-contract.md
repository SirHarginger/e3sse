# E3-SSE Scientific Contract

The project tests whether systematic softening of foundation interatomic
potentials can be measured from DFT-labelled forces and converted into a
training-free correction of Li-ion transport predictions.

Core sequence:

1. inspect and validate reference datasets;
2. estimate softening without prohibited leakage;
3. test correction on migration barriers;
4. test transfer to finite-temperature transport;
5. quantify uncertainty and label requirements;
6. construct conformal/FDR-controlled selections;
7. demonstrate the frozen workflow prospectively.

Every analysis must keep reference data, predictors, fitted quantities,
calibration information, and held-out outcomes distinguishable.

The project must remain falsifiable: a method failing its predefined test is a
scientific result, not something to repair by looking at held-out outcomes.
