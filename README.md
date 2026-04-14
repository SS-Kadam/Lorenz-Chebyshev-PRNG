# Lorenz-Chebyshev Chaotic PRNG

> A hybrid chaotic pseudorandom number generator combining a continuous 3D Lorenz attractor with a discrete 1D Chebyshev map, seeded via Argon2id and validated against the full NIST SP 800-22 test suite.

---

## Overview

This generator fuses two complementary chaotic systems — the 3D **Lorenz attractor** and the 1D **Chebyshev map** — into a single high-quality PRNG. Argon2id-based seeding expands into a 1024-bit key space, and ~260-bit arbitrary-precision arithmetic (via `mpmath`) prevents the finite-precision degradation that commonly afflicts chaotic generators. The two chaotic streams are XOR-combined after transient discarding to produce the final output bitstream.

---

## Features

- **Hybrid chaotic core** — 3D Lorenz attractor and 1D Chebyshev map with bitwise XOR mixing
- **Cryptographically strong seeding** — Argon2id → HKDF-style expansion to a 1024-bit key space
- **High-precision arithmetic** — ~260-bit (78 decimal digit) precision with RK4 integration
- **Transient discard** — first 10,000 iterations removed to eliminate initial non-chaotic behaviour
- **Bit extraction** — fractional-part doubling method for high-quality randomness
- **Full statistical validation** — passed the complete NIST SP 800-22 test suite (15 tests)

---

## Results

| Test Suite | Result | Notes |
|---|---|---|
| NIST SP 800-22 (15 tests) | Passed | p-values well above threshold |
| Shannon Entropy | Excellent | Near-ideal uniformity |
| Linear Complexity / Autocorrelation | Strong | No detectable structure |
| ML / Neural-network distinguishers | Highly resistant | See Security Analysis |

---

## Updated 'nistrng' library

This repository includes a patched version of the [`nistrng`](https://github.com/InsaneMonster/NistRng) Python library. Several bugs identified in the original package and test modules have been resolved. The original author's copyright is retained in full; all modifications are clearly documented alongside it.

--

## Security Analysis

Post-publication red-teaming was conducted using modern machine-learning distinguishers to evaluate resistance against predictive attacks.

**Setup:** Over 3.5 million samples were evaluated with lagged prediction windows ranging from k = 50 to k = 200, using Linear Regression, MLP, ExtraTreesRegressor, and HistGradientBoostingRegressor.

**Outcome:** No model outperformed the trivial "predict the mean" baseline. MSE values remained at the theoretical minimum for uniform [0, 1] data (~0.0834) and correlation coefficients stayed at noise level — indicating that the multi-map chaos, high-precision arithmetic, Argon2id seeding, and XOR mixing collectively destroy short- to medium-term structure beyond the reach of contemporary AI-based cryptanalysis.

Full experimental setup, hyperparameters, and raw logs are available in `IMPLEMENTATION.md` and the `ml_attacks/` directory.

---

## Documentation

For complete implementation details — mathematical formulation, RK4 integration, parameter mapping, bit extraction, test-suite commands, and the full ML attack methodology — see:

→ [`IMPLEMENTATION.md`](./IMPLEMENTATION.md)

---

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to open an issue or submit a pull request.
