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
| Linear Complexity / Serial correlation / Poker-Test | Strong | No detectable structure |
| ML / Neural-network distinguishers | Highly resistant | See Security Analysis |

---

## Updated 'nistrng' library

This repository includes a patched version of the [`nistrng`](https://github.com/InsaneMonster/NistRng) Python library. Several bugs identified in the original package and test modules have been resolved. The original author's copyright is retained in full; all modifications are clearly documented alongside it.

---

## Security Analysis

## Machine Learning-Based Red-Teaming Evaluation

To evaluate resistance against predictive attacks, the generator was tested using a range of machine learning models with increasing expressive power on large-scale outputs.

### Experimental Setup

* Data: up to **10 million bits** (~10⁷ samples)
* Task: **next-bit prediction**
* Input: sliding windows with **k = 50–200**
* Evaluation: **80/20 train–test split**
* Baseline: random guessing (**50% accuracy**, MSE ≈ theoretical minimum)

---

## Models Evaluated

### 1. Linear Regression

* Result: No improvement over baseline
* Accuracy ≈ 50%, correlation ≈ 0

**Interpretation:**
No detectable linear relationship between past and future bits.

---

### 2. Logistic Regression

* Train Accuracy: ~0.5005
* Test Accuracy: ~0.5008

**Interpretation:**
No statistically meaningful predictive advantage. Consistency between train and test results indicates absence of generalizable structure.

---

### 3. Tree-Based Models (ExtraTrees, HistGradientBoosting)

* Result: No improvement over baseline
* MSE ≈ 0.0834 (theoretical value for uniform [0,1] data)

**Interpretation:**
No evidence of nonlinear partitioning or exploitable structure within the input space.

---

### 4. Feedforward Neural Network (MLP)

* Architecture: single hidden layer (ReLU activation, 64 units)
* Train Accuracy: ~0.5003–0.5004
* Test Accuracy: ~0.4995–0.5001

**Interpretation:**
No learnable nonlinear structure within the tested window sizes. Predictions remain indistinguishable from random guessing.

---

## Statistical Analysis

* Test set size: ~2 million samples
* Expected standard deviation: ≈ 0.035%
* Observed deviations: within ~0.05–0.08% (approximately 1–2 standard deviations)

These deviations fall within expected statistical noise.

---

## Conclusion

Across linear models, ensemble methods, and nonlinear neural networks, no model achieved predictive performance beyond random guessing on held-out data.

These findings indicate:

* No detectable short- to medium-range linear or nonlinear structure
* Output behavior consistent with random sequences under the tested conditions

---

## Limitations

This evaluation does not address:

* Long-range dependencies
* Sequence-based models (e.g., LSTM, Transformers)
* Formal cryptographic security proofs

Accordingly, these results provide empirical evidence of resistance to practical machine learning-based prediction, but do not constitute a proof of cryptographic security.


---

## Documentation

For complete implementation details — see:

→ [`r_Implementation.md`]

---

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to open an issue or submit a pull request.
