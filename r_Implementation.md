# Implementation Guide

This document provides precise instructions for using, testing, and reproducing results from the **Lorenz–Chebyshev PRNG** implementation.

---

## Pipeline Overview

Password → Argon2id → Seed → Lorenz System + Chebyshev Map → XOR Combination → Bitstream Output

---

## 1. Main_PRNG_source_code

* Contains the core implementation of the PRNG.
* Includes three primary files:

  * Lorenz module
  * Chebyshev module
  * `Main_PRNG 1.py` (main execution file)

### Core Functionality

* Lorenz and Chebyshev modules independently generate sequences.
* Outputs are combined using an XOR operation in `Main_PRNG 1.py`.
* Argon2id is used to derive a **1024-bit seed** from the user-provided password.

### Important Notes

* Argon2id is used **only for high-entropy seed derivation**.
* It does **not guarantee cryptographic security** of the PRNG.

### Reproducibility Conditions

Reproducibility requires identical:

* Password input
* Argon2id parameters
* RK4 timestep
* Floating-point precision/environment
* Warm-up length (10,000 iterations)

Any variation in the above may produce different outputs.

* For reproducibility demonstrations, salt/IV is not used.
* For real-world usage, salts **must be used**.

### Usage Instructions

1. Navigate to `Main_PRNG_source_code/`
2. Run `Main_PRNG 1.py`
3. Enter and confirm password
4. Modify `out_bits` in code to set output length

### Output Behavior

* Generator performs a warm-up phase (first 10,000 iterations discarded)
* Output is written to a `.BIN` file
* After execution, the following are printed:

  * Lorenz parameters
  * Chebyshev parameters
  * RK4 timestep
  * Total execution time

---

## 2. Statistical_Python_tests

* Contains statistical tests including:

  * Lempel-Ziv Complexity (LZC)
  * Poker Test
  * Shannon Entropy
  * Serial Correlation
  * Chi-Square Test

### Usage

1. Generate a `.BIN` file using the PRNG
2. Open `Python_stats.py`
3. Provide the file name/path
4. Run the script

### Example

* Password: `TestRUN@123`
* out_bits: `10,000,000`
* Output: e.g., Serial Correlation ≈ -0.000275

---

## 3. Throughput_Speed

* Measures performance of the PRNG
* Uses identical logic to the main generator
* Output bits are discarded (not written to file)

### Usage

1. Run `Throughput.py` (multiple cycles are executed)
2. Enter password and number of bits
3. Execution time is measured using `time.perf_counter()`

### Note

* Throughput varies with Argon2id parameters as well as with high bit precision

---

## 4. Updated_nistrng

* Python wrapper for **NIST SP 800-22** test suite (15 tests)
* Original BSD license preserved
* Modifications documented within the code

### Usage

1. Open `main_NIST_test.py`
2. Set:

   * `FILE_PATH` → path to `.BIN` file
   * `TEST_TO_RUN` → desired test (e.g., `Sums`)
3. Run the script

### Output

* Returns p-values for selected tests
* Example: Maurer's Universal Test → p-value = 0.584

---

## 5. Visualization_plotting

* Provides visualization of chaotic systems and LLE computation

### Lorenz System

* Input password
* Outputs:

  * Lorenz parameters
  * Largest Lyapunov Exponent (LLE)
  * 3D Lorenz attractor plot

### Chebyshev System

* Outputs:

  * Phase space plot
  * Return (lagged) map
  * LLE value

---

## 6. Requirements

* Python 3.12+

### Required Libraries

* argon2-cffi (Argon2id)
* mpmath (high precision arithmetic)
* scipy (RK4 integration, statistics)
* numpy
* futures

---

## 7. Reference Test Vector

This section includes fixed input/output pairs for independent verification.

**Example format:**

* Password: `TestRun1024Git`
* out_bits: `1,000,000`
* First N bits:

```
000110011000111001000111001100110011111010001001110100101001100111001010001010100110001...
```

* SHA-256 of full output:

```
48242e7f6da5b5da8aee38ce68df50e012fb599028935cf126fb6b197057ff37
```

> Bit-exact match is expected under identical environments. Minor variation may occur across different systems.

---

## 8. ML-Based Evaluation (Summary)

The PRNG has been evaluated against machine learning-based predictive models.

* Models include linear and non-linear regressors
* No implementation-specific defenses were added
* Observed resistance arises from generator structure

Refer to README for full experimental details and results.

---

## 9. Limitations

* This PRNG is **not proven to be cryptographically secure**
* Passing statistical tests does **not guarantee unpredictability**
* Output may vary across different hardware or floating-point implementations

Use in security-critical applications is **not recommended without further analysis**

---

## 10. Determinism

* Same password and parameters → same output (under identical environment conditions)
* Changes in precision or parameters may alter results

---

End of document.
