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
# Changes made for compatibility with modern NumPy (1.20+) and Python 3.10+:
#   - Fixed Gaussian elimination pivot logic (was column-based, now correctly row-based)
#   - Added full row elimination (above AND below pivot) for correct reduced form
#   - Removed redundant and potentially incorrect final non-zero row recount
#   - Added flatten() + where() normalization for multidimensional array safety
#   - Added N < 38 guard per NIST minimum requirement
#   - Explicit int8 casting for memory efficiency on large bit arrays
# Original source: https://github.com/lucapasqualini/nistrng

# Import packages

import math
import numpy
import scipy.special
from nistrng import Test, Result


class BinaryMatrixRankTest(Test):
    """
    Binary Matrix Rank test as described in NIST paper:
    https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-22r1a.pdf

    The focus of this test is the rank of disjoint sub-matrices of the entire sequence.
    The purpose of this test is to check for linear dependence among fixed-length
    substrings of the original sequence.

    Uses 32x32 binary matrices with pre-calculated NIST probability constants.
    Minimum sequence length: 32 * 32 * 38 = 38912 bits (NIST requirement of N >= 38 matrices).
    """

    def __init__(self):
        self._rows = 32
        self._cols = 32
        # Pre-calculated NIST constants for 32x32 matrices
        # Source: NIST SP 800-22 Rev 1a, Section 2.5, Table 1
        self._p_full      = 0.2887880952   # P(rank = M)
        self._p_minus_1   = 0.5775761905   # P(rank = M-1)
        self._p_remainder = 0.1336357143   # P(rank <= M-2)
        super(BinaryMatrixRankTest, self).__init__("Binary Matrix Rank", 0.01)

    def _execute(self, bits: numpy.ndarray) -> Result:
        # --- STEP 1: NORMALIZATION ---
        # Flatten to 1D and ensure clean binary values (0 or 1)
        bits = numpy.asarray(bits).flatten()
        bits = numpy.where(bits > 0, 1, 0).astype(numpy.int8)
        n = bits.size

        # --- STEP 2: BLOCK PARAMETERS ---
        matrix_size = self._rows * self._cols  # 1024 bits per matrix
        N = n // matrix_size                   # Total number of matrices

        # NIST requires at least 38 matrices for the test to be valid
        if N < 38:
            return Result(self.name, False, numpy.array(0.0))

        # --- STEP 3: RANK CALCULATION ---
        full_rank  = 0   # Matrices with rank == 32
        minus_1    = 0   # Matrices with rank == 31
        remainder  = 0   # Matrices with rank <= 30

        for i in range(N):
            block = bits[i * matrix_size: (i + 1) * matrix_size].reshape(self._rows, self._cols)
            rank  = self._binary_matrix_rank(block)

            if rank == 32:
                full_rank += 1
            elif rank == 31:
                minus_1 += 1
            else:
                remainder += 1

        # --- STEP 4: CHI-SQUARE STATISTIC ---
        # Formula per NIST SP 800-22 Section 2.5:
        # chi_sq = sum((F_i - N*p_i)^2 / (N*p_i))
        e_full     = self._p_full      * N
        e_minus_1  = self._p_minus_1   * N
        e_remainder= self._p_remainder * N

        chi_sq = (
            ((full_rank - e_full)      ** 2 / e_full)      +
            ((minus_1   - e_minus_1)   ** 2 / e_minus_1)   +
            ((remainder - e_remainder) ** 2 / e_remainder)
        )

        # --- STEP 5: P-VALUE ---
        # NIST uses exponential approximation: p-value = exp(-chi_sq / 2)
        score = math.exp(-chi_sq / 2.0)

        success = score >= self.significance_value
        return Result(self.name, success, numpy.array(score))

    def _binary_matrix_rank(self, matrix: numpy.ndarray) -> int:
        """
        Computes the rank of a binary matrix using Gaussian elimination over GF(2).

        Algorithm:
        - Iterate over each column
        - Find a pivot row at or below the current rank position
        - Swap pivot row into position
        - Eliminate all other rows (above AND below) using XOR
        - Increment rank counter

        Returns the integer rank of the matrix.
        """
        m = matrix.copy().astype(numpy.int8)
        rank = 0

        for col in range(self._cols):
            # --- Find pivot row ---
            pivot_row = -1
            for row in range(rank, self._rows):
                if m[row][col] == 1:
                    pivot_row = row
                    break

            # No pivot in this column — skip
            if pivot_row == -1:
                continue

            # --- Swap pivot row into current rank position ---
            m[[rank, pivot_row]] = m[[pivot_row, rank]]

            # --- Eliminate all other rows above AND below ---
            for row in range(self._rows):
                if row != rank and m[row][col] == 1:
                    m[row] = (m[row] + m[rank]) % 2

            rank += 1

        return rank

    def is_eligible(self, bits: numpy.ndarray) -> bool:
        """
        Checks NIST minimum requirement:
        At least 38 matrices of size 32x32 = 38,912 bits minimum.
        """
        return bits.size >= (self._rows * self._cols * 38)