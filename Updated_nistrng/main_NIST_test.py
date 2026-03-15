import numpy as np
import os

# --- IMPORT ALL TESTS ---
from nistrng.sp800_22r1a.test_monobit import MonobitTest
from nistrng.sp800_22r1a.test_frequency_within_block import FrequencyWithinBlockTest
from nistrng.sp800_22r1a.test_runs import RunsTest
from nistrng.sp800_22r1a.test_longest_run_ones_in_a_block import LongestRunOnesInABlockTest
from nistrng.sp800_22r1a.test_binary_matrix_rank import BinaryMatrixRankTest
from nistrng.sp800_22r1a.test_non_overlapping_template_matching import NonOverlappingTemplateMatchingTest
from nistrng.sp800_22r1a.test_overlapping_template_matching import OverlappingTemplateMatchingTest
from nistrng.sp800_22r1a.test_maurers_universal import MaurersUniversalTest
from nistrng.sp800_22r1a.test_linear_complexity import LinearComplexityTest
from nistrng.sp800_22r1a.test_serial import SerialTest
from nistrng.sp800_22r1a.test_approximate_entropy import ApproximateEntropyTest
from nistrng.sp800_22r1a.test_random_excursion import RandomExcursionTest
from nistrng.sp800_22r1a.test_random_excursion_variant import RandomExcursionVariantTest
from nistrng.sp800_22r1a.test_discrete_fourier_transform import DiscreteFourierTransformTest
from nistrng.sp800_22r1a.test_cumulative_sums import CumulativeSumsTest


# ----------------------------------------------------------------------
# CONFIG AREA
# ----------------------------------------------------------------------
FILE_PATH = r"C:\Users\Square Panda Admin\PycharmProjects\HelloWorld\app.py\file_name"
TEST_TO_RUN = "monobit"

BLOCK_SIZE = 128
SERIAL_BLOCK = 16
TEMPLATE = "111111111"


# ----------------------------------------------------------------------
# LOAD BITSTREAM AS NUMPY ARRAY
# ----------------------------------------------------------------------
def load_bits(path):
    with open(path, "rb") as f:
        data = f.read()

    bits = np.unpackbits(
        np.frombuffer(data, dtype=np.uint8),
        bitorder='big'      # MSB first — correct for NIST tests
    ).astype(np.int8)       # int8 — memory efficient, clean binary values

    print("Total bits :", bits.size)
    print("Ones ratio :", np.mean(bits))  # Sanity check — should be ~0.5

    return bits
# ----------------------------------------------------------------------
# TEST FACTORY (ALL USE execute())
# ----------------------------------------------------------------------
def get_test(name, bits):
    tests = {
        "monobit": lambda: MonobitTest()._execute(bits),

        "frequency": lambda: FrequencyWithinBlockTest()._execute(bits),

        "runs": lambda: RunsTest()._execute(bits),

        "longest_run": lambda: LongestRunOnesInABlockTest()._execute(bits),

        "matrix_rank": lambda: BinaryMatrixRankTest()._execute(bits),

        "non_overlapping": lambda: NonOverlappingTemplateMatchingTest()._execute(bits, ),

        "overlapping": lambda: OverlappingTemplateMatchingTest()._execute(bits),

        "maurer": lambda: MaurersUniversalTest()._execute(bits),

        "linear": lambda: LinearComplexityTest()._execute(bits),

        "serial": lambda: SerialTest()._execute(bits),

        "approx": lambda: ApproximateEntropyTest()._execute(bits),

        "excursion": lambda: RandomExcursionTest()._execute(bits),

        "excursion_variant": lambda: RandomExcursionVariantTest()._execute(bits),

        "dft": lambda: DiscreteFourierTransformTest()._execute(bits),

        "sums": lambda: CumulativeSumsTest()._execute(bits),
         }

    if name not in tests:
            raise ValueError(f"Test '{name}' not recognized.")

    return tests[name]()

# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(FILE_PATH)

    bits = load_bits(FILE_PATH)
    result = get_test(TEST_TO_RUN, bits)

    print("\n=== RESULT ===")
    print("Test:", result.name)
    print("Passed:", result.passed)
    print("p-value:", result.score)

