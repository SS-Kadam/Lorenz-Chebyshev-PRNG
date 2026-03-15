#
# Copyright (C) 2019 Luca Pasqualini
# University of Siena - Artificial Intelligence Laboratory - SAILab
#
# Inspired by the work of David Johnston (C) 2017: https://github.com/dj-on-github/sp800_22_tests
#
# NistRng is licensed under a BSD 3-Clause.
#
# You should have received a copy of the license along with this
# work. If not, see <https://opensource.org/licenses/BSD-3-Clause>.
#
# --- MODIFICATIONS ---
# Modified by [SS Kadam], [2026]
# Changes made for correctness and compatibility with modern NumPy (1.20+) and Python 3.10+:
#   - Fixed critical overflow: bits now cast to int32 before cumsum
#     int8 overflows after ~127 steps, silently corrupting the entire random walk
#   - Fixed cycle detection off-by-one: enumerate(sum_prime[1:]) was reading
#     one position behind; replaced with direct value iteration
#   - Fixed frequency counting chained comparison: '5 > k == occurrences'
#     replaced with explicit 'k < 5 and occurrences == k' to avoid
#     Python chained comparison edge cases
#   - Added NIST minimum cycle count guard: J >= 500 required for valid test
#   - Replaced triple nested loop with vectorized NumPy counting (major speedup)
#   - Added division by zero guard in chi-square computation
#   - Added input normalization (flatten + where) consistent with other fixed tests
# Original source: https://github.com/lucapasqualini/nistrng

# Import packages

import numpy
import scipy.special

# Import required src

from nistrng import Test, Result


class RandomExcursionTest(Test):
    """
    Random excursion test as described in NIST paper:
    https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-22r1a.pdf

    The focus of this test is the number of cycles having exactly K visits in a
    cumulative sum random walk. The cumulative sum random walk is derived from
    partial sums after the (0,1) sequence is transferred to the (-1, +1) sequence.

    A cycle of a random walk consists of a sequence of steps of unit length taken
    at random that begin at and return to the origin. The purpose of this test is
    to determine if the number of visits to a particular state within a cycle
    deviates from what one would expect for a random sequence.

    This test is a series of eight tests, one for each state: -4,-3,-2,-1,+1,+2,+3,+4.
    All eight p-values must pass for the test to succeed.

    Minimum requirement: J >= 500 cycles (NIST SP 800-22 Section 2.14).
    The significance value of the test is 0.01.
    """

    def __init__(self):
        # NIST SP 800-22 Table 2.14.4
        # Pre-calculated visit probabilities π_k(|x|) for states |x| = 1..7
        # Each row: [k=0, k=1, k=2, k=3, k=4, k>=5]
        self._probabilities_xk = [
            numpy.array([0.5,    0.25,   0.125,  0.0625, 0.0312, 0.0312]),  # |x|=1
            numpy.array([0.75,   0.0625, 0.0469, 0.0352, 0.0264, 0.0791]),  # |x|=2
            numpy.array([0.8333, 0.0278, 0.0231, 0.0193, 0.0161, 0.0804]),  # |x|=3
            numpy.array([0.875,  0.0156, 0.0137, 0.012,  0.0105, 0.0733]),  # |x|=4
            numpy.array([0.9,    0.01,   0.009,  0.0081, 0.0073, 0.0656]),  # |x|=5
            numpy.array([0.9167, 0.0069, 0.0064, 0.0058, 0.0053, 0.0588]),  # |x|=6
            numpy.array([0.9286, 0.0051, 0.0047, 0.0044, 0.0041, 0.0531]),  # |x|=7
        ]

        # States tested per NIST SP 800-22 Section 2.14
        self._states = [-4, -3, -2, -1, 1, 2, 3, 4]

        # Generate base Test class
        super(RandomExcursionTest, self).__init__("Random Excursion", 0.01)

    def _execute(self, bits: numpy.ndarray) -> Result:
        """
        Overridden method of Test class: check its docstring for further information.
        """
        # --- STEP 1: NORMALIZATION ---
        # Flatten to 1D, map 0→-1 and 1→+1, cast to int32
        # int32 is REQUIRED — cumsum of 10M values of ±1 far exceeds int8 range (max 127)
        bits = numpy.asarray(bits).flatten()
        bits = numpy.where(bits > 0, 1, -1).astype(numpy.int32)
        n = bits.size

        # --- STEP 2: BUILD PADDED CUMULATIVE SUM (S') ---
        # Pad with zeros at both ends as required by NIST
        sum_prime = numpy.concatenate((
            numpy.array([0], dtype=numpy.int32),
            numpy.cumsum(bits),
            numpy.array([0], dtype=numpy.int32)
        ))

        # --- STEP 3: CYCLE DETECTION ---
        # A cycle starts at 0 and ends the next time the walk returns to 0.
        # Iterate values directly (not indices) to avoid off-by-one errors.
        cycles = []
        cycle = [0]
        for val in sum_prime[1:]:
            val = int(val)
            cycle.append(val)
            if val == 0:
                cycles.append(cycle)
                cycle = [0]

        # Total number of cycles J
        j = len(cycles)

        # --- STEP 4: NIST MINIMUM CYCLE COUNT CHECK ---
        # NIST SP 800-22 Section 2.14: test is invalid if J < 500
        if j < 500:
            return Result(self.name, False, numpy.array([0.0] * 8))

        # --- STEP 5: BUILD FREQUENCY TABLE V_k(x) ---
        # For each state x and each visit count k (0..5),
        # count how many cycles visit state x exactly k times (k=5 means >=5)
        frequencies_table = {state: numpy.zeros(6, dtype=int)
                             for state in self._states}

        for state in self._states:
            # Vectorized: count visits to this state in each cycle
            visit_counts = numpy.array([
                int(numpy.count_nonzero(numpy.array(c) == state))
                for c in cycles
            ])

            # Bin into k=0,1,2,3,4 and k>=5
            for k in range(5):
                # Explicit comparison — avoids Python chained comparison bug
                frequencies_table[state][k] = int(numpy.sum(visit_counts == k))
            frequencies_table[state][5] = int(numpy.sum(visit_counts >= 5))

        # --- STEP 6: CHI-SQUARE AND P-VALUE PER STATE ---
        # χ²(x) = Σ(k=0..5) (V_k(x) - J·π_k)² / (J·π_k)
        # p-value = igamc(5/2, χ²/2)
        scores = []
        for state in self._states:
            probs     = self._probabilities_xk[abs(state) - 1]
            expected  = j * probs
            observed  = frequencies_table[state].astype(float)

            # Guard against zero denominator (should not occur with valid J and probs)
            if numpy.any(expected == 0):
                scores.append(0.0)
                continue

            chi_square = float(numpy.sum(((observed - expected) ** 2) / expected))
            score      = float(scipy.special.gammaincc(5.0 / 2.0, chi_square / 2.0))
            scores.append(score)

        # All 8 state p-values must pass
        success = all(s >= self.significance_value for s in scores)
        return Result(self.name, success, numpy.array(scores))

    def is_eligible(self, bits: numpy.ndarray) -> bool:
        """
        Overridden method of Test class: check its docstring for further information.

        NIST requires J >= 500 cycles. A rough lower bound on sequence length
        to achieve this is n >= 1,000,000 bits, though this is checked
        dynamically inside _execute via the cycle count.
        """
        return bits.size >= 1_000_000