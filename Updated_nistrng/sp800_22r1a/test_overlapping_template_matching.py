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
#   - Fixed off-by-one in matching window: range end changed from (M-m) to (M-m+1)
#     to include the final valid window position
#   - Fixed chi_square == 0 case: now correctly returns True (perfect pass)
#     instead of False (silent failure)
#   - Removed misleading hard-coded probability array that was never actually used;
#     replaced with clean zero-initialized array computed entirely from _get_probabilities
#   - Removed duplicate imports (numpy, scipy.special, nistrng appeared twice)
#   - Replaced custom _log_gamma helper with scipy.special.gammaln (more stable, native)
#   - Added template length enforcement: NIST block parameters are calibrated for m=9 only
#   - Added input normalization (flatten + where) consistent with other fixed tests
#   - Added division by zero guard in chi-square computation
# Original source: https://github.com/lucapasqualini/nistrng

# Import packages

import numpy
import scipy.special

# Import required src

from nistrng import Test, Result


class OverlappingTemplateMatchingTest(Test):
    """
    Overlapping Template Matching test as described in NIST paper:
    https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-22r1a.pdf

    The focus of this test is the number of occurrences of pre-specified target strings
    in the sequence, where overlapping matches ARE counted (unlike the non-overlapping test).

    The default and NIST-standard template is nine consecutive ones: "111111111".
    The block parameters (N=968 blocks, M=1062 bits per block) are calibrated specifically
    for m=9 per NIST SP 800-22 Section 2.8. Passing a different template length will
    raise a ValueError to prevent silently incorrect results.

    Minimum sequence length: N * M = 968 * 1062 = 1,028,016 bits.
    The significance value of the test is 0.01.
    """

    def __init__(self, template: str = "111111111"):
        # --- TEMPLATE VALIDATION ---
        # NIST block parameters are calibrated for m=9 only.
        # Enforcing this prevents silently wrong chi-square results
        # when a different length template is passed.
        if len(template) != 9:
            raise ValueError(
                f"Template must be exactly 9 bits per NIST SP 800-22 Section 2.8. "
                f"Got length {len(template)}."
            )

        # Validate template contains only '0' and '1' characters
        if not all(c in "01" for c in template):
            raise ValueError(
                f"Template must contain only '0' and '1' characters. Got: '{template}'"
            )

        self.template = template
        self._template_bits_length: int = len(template)

        # NIST SP 800-22 Section 2.8 fixed parameters for m=9
        self._blocks_number: int = 968           # N — number of blocks
        self._freedom_degrees: int = 5           # degrees of freedom for chi-square
        self._substring_bits_length: int = 1062  # M — bits per block

        # Generate base Test class
        super().__init__("Overlapping Template Matching", 0.01)

    def _execute(self, bits: numpy.ndarray) -> Result:
        """
        Overridden method of Test class: check its docstring for further information.
        """
        # --- STEP 1: NORMALIZATION ---
        # Flatten to 1D and ensure clean binary values (0 or 1)
        bits = numpy.asarray(bits).flatten()
        bits = numpy.where(bits > 0, 1, 0).astype(numpy.int8)

        # Convert template string to numpy array once, outside the loop
        b_template = numpy.array([int(b) for b in self.template], dtype=numpy.int8)
        m = self._template_bits_length

        # --- STEP 2: COUNT OVERLAPPING MATCHES PER BLOCK ---
        # For overlapping matching, the window slides ONE position at a time
        # (unlike non-overlapping where it jumps past the found pattern).
        # Use range(M - m + 1) to include the final valid window position.
        matches_distributions = numpy.zeros(self._freedom_degrees + 1, dtype=int)

        for i in range(self._blocks_number):
            block = bits[
                i * self._substring_bits_length:
                (i + 1) * self._substring_bits_length
            ]
            count = 0

            # +1 ensures final window at position (M-m) is included (off-by-one fix)
            for position in range(self._substring_bits_length - m + 1):
                if (block[position:position + m] == b_template).all():
                    count += 1

            # Cap at freedom_degrees (bin 5 catches all counts >= 5)
            matches_distributions[min(count, self._freedom_degrees)] += 1

        # --- STEP 3: COMPUTE η ---
        # η = (M - m + 1) / 2^(m+1)
        # This is the expected number of overlapping matches per block / 2
        eta = (
            (self._substring_bits_length - m + 1.0)
            / (2.0 ** m)
            / 2.0
        )

        # --- STEP 4: COMPUTE PROBABILITIES ---
        # Probabilities for bins k=0..4 computed analytically via NIST formula.
        # Bin k=5 absorbs all remaining probability mass (sum to 1).
        # No hard-coded fallback values — computed entirely from _get_probabilities.
        probabilities = numpy.zeros(self._freedom_degrees + 1)
        probabilities[:self._freedom_degrees] = self._get_probabilities(
            numpy.arange(self._freedom_degrees), eta
        )
        # Final bin gets whatever probability mass remains
        probabilities[-1] = 1.0 - numpy.sum(probabilities[:self._freedom_degrees])

        # Clamp to avoid tiny negative values from floating point
        probabilities = numpy.clip(probabilities, 0.0, 1.0)

        # --- STEP 5: CHI-SQUARE ---
        # χ² = Σ (observed - N*p)² / (N*p)
        expected = self._blocks_number * probabilities

        # Guard against zero expected values
        if numpy.any(expected == 0.0):
            return Result(self.name, False, numpy.array(0.0))

        chi_square = float(
            numpy.sum(
                ((matches_distributions - expected) ** 2) / expected
            )
        )

        # --- STEP 6: P-VALUE ---
        # chi_square == 0 means observed perfectly matches expected — this is a pass
        if chi_square == 0.0:
            return Result(self.name, True, numpy.array(1.0))

        score = float(scipy.special.gammaincc(
            self._freedom_degrees / 2.0,
            chi_square / 2.0
        ))

        success = score >= self.significance_value
        return Result(self.name, success, numpy.array(score))

    def is_eligible(self, bits: numpy.ndarray) -> bool:
        """
        Overridden method of Test class: check its docstring for further information.

        Minimum: N * M = 968 * 1062 = 1,028,016 bits.
        """
        return bits.size >= (self._blocks_number * self._substring_bits_length)

    @staticmethod
    def _get_probabilities(freedom_degree_values: numpy.ndarray,
                           eta_value: float) -> list:
        """
        Compute bin probabilities P(V=u) for u = 0, 1, ..., freedom_degrees-1.

        Uses the log-space NIST formula to avoid underflow for large η:
            P(V=0) = exp(-η)
            P(V=u) = Σ_{i=1}^{u} exp(
                        -η - u·ln2 + i·ln(η)
                        - lnΓ(i+1) + lnΓ(u) - lnΓ(i) - lnΓ(u-i+1)
                     )

        Uses scipy.special.gammaln directly (replaces the redundant _log_gamma helper).

        :param freedom_degree_values: array of k values [0, 1, ..., freedom_degrees-1]
        :param eta_value: computed η for this block configuration
        :return: list of probabilities, one per freedom degree value
        """
        probabilities = []

        for u in freedom_degree_values:
            if u == 0:
                probability = float(numpy.exp(-eta_value))
            else:
                indexes = numpy.arange(1, u + 1)
                probability = float(
                    numpy.sum(
                        numpy.exp(
                            - eta_value
                            - u * numpy.log(2)
                            + indexes * numpy.log(eta_value)
                            - scipy.special.gammaln(indexes + 1)
                            + scipy.special.gammaln(u)
                            - scipy.special.gammaln(indexes)
                            - scipy.special.gammaln(u - indexes + 1)
                        )
                    )
                )

            probabilities.append(probability)

        return probabilities