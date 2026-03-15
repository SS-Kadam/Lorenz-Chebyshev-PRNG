# =============================================================================
# Copyright (C) 2026 SS Kadam
# All rights reserved.
#
# Performance benchmarking suite for the chaos-based PRNG.
# Measures throughput, timing, and output volume across varying
# bit generation targets.
#
# Development environment : Local (private development machine)
# Development year        : 2026
#
# Licensed under the MIT License.
# See LICENSE file in the project root for full license text.
# =============================================================================

""" USE CASE: Run the Throughput program in the RUN terminal on the left side of the Pycharm IDE
    Use the following code: { python Throughput.py --benchmark --bits 1000000 --password test123 }
"""

from __future__ import annotations
import os
import time
from typing import Optional
from binascii import hexlify

# Import your modules
from lorenx import generate_lorenx_bits, derive_okm_argon2
from chebyxhev import generate_chebyxhev_bits

# ----------------------------------------------------------------------
# Helper: XOR two equal-length binary files
# ----------------------------------------------------------------------
def xor_two_files(file_a: str, file_b: str, out_file: str) -> None:
    with open(file_a, "rb") as fa, open(file_b, "rb") as fb, open(out_file, "wb") as fo:
        while True:
            a = fa.read(1 << 20)
            b = fb.read(len(a))
            if len(a) != len(b):
                raise ValueError("Files have different sizes")
            if not a:
                break
            fo.write(bytes(x ^ y for x, y in zip(a, b)))

# ----------------------------------------------------------------------
# Hybrid PRNG generator (normal mode - writes to disk)
# ----------------------------------------------------------------------
def generate_hybrid_bits(
    password: bytes,
    salt: bytes,
    out_bits: int = 10_000_000,
    bits_per_sample: int = 32,
    lorenz_gap: int = 5,
    dt: float = 1e-3,
    mp_binary_bits: int = 128,
    out_file: str = "hybrid_output.bin",
):
    okm_bytes = derive_okm_argon2(password, salt, outlen_bytes=128)
    lorenz_file = "temp_lorenz.bin"
    cheb_file = "temp_cheb.bin"

    print("Generating Lorenz bits...")
    generate_lorenx_bits(
        password=password,
        salt=salt,
        out_bits=out_bits,
        bits_per_sample=bits_per_sample,
        sample_gap=lorenz_gap,
        dt=dt,
        mp_binary_bits=mp_binary_bits,
        okm_bytes=okm_bytes,
        out_file=lorenz_file
    )
    print("Generating Chebyshev bits...")
    generate_chebyxhev_bits(
        password=None,
        salt=None,
        out_bits=out_bits,
        bits_per_sample=bits_per_sample,
        mp_binary_bits=mp_binary_bits,
        okm_bytes=okm_bytes,
        out_file=cheb_file
    )
    print("XORing outputs...")
    xor_two_files(lorenz_file, cheb_file, out_file)
    os.remove(lorenz_file)
    os.remove(cheb_file)
    print("Done:", out_file)

# ----------------------------------------------------------------------
# CLEAN IN-MEMORY BENCHMARK (seeding excluded from timing)
# ----------------------------------------------------------------------
def benchmark_prng(
    password: bytes,
    salt: bytes,
    out_bits: int,
    runs: int = 3,
    show_warmup: bool = False
):
    print("\n" + "=" * 50)
    print("IN-MEMORY BENCHMARK (Argon2id seeding excluded from timing)")
    print(f"Target: {out_bits:,} bits  |  Runs: {runs} (plus 1 warm-up)")
    print("=" * 50)

    # Derive key material ONLY ONCE — outside any timing
    print("Deriving OKM (Argon2id) ... ", end="", flush=True)
    start_seed = time.perf_counter()
    okm_bytes = derive_okm_argon2(password, salt, outlen_bytes=128)
    seed_time = time.perf_counter() - start_seed
    print(f"done in {seed_time:.3f} s")

    times = []

    # Warm-up run (helps stabilize CPU freq, mpmath caches, etc.)
    print("\nWarm-up run...", end="", flush=True)
    generate_lorenx_bits(
        password=None, salt=None, out_bits=out_bits,
        okm_bytes=okm_bytes, benchmark_mode=True
    )
    generate_chebyxhev_bits(
        password=None, salt=None, out_bits=out_bits,
        okm_bytes=okm_bytes, benchmark_mode=True
    )
    print(" done")

    for i in range(runs):
        print(f"Run {i+1}/{runs} ... ", end="", flush=True)
        start = time.perf_counter()

        generate_lorenx_bits(
            password=None, salt=None, out_bits=out_bits,
            okm_bytes=okm_bytes, benchmark_mode=True
        )

        generate_chebyxhev_bits(
            password=None, salt=None, out_bits=out_bits,
            okm_bytes=okm_bytes, benchmark_mode=True
        )

        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"{elapsed:.3f} s")

    if not times:
        print("No runs completed.")
        return

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    throughput_mbps = (out_bits / avg_time) / 1_000_000
    throughput_mibps = (out_bits / avg_time) / (1024 * 1024)
    throughput_MBps = throughput_mbps / 8

    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Bits generated per run:     {out_bits:,}")
    print(f"Average time per run:       {avg_time:.3f} s  (min: {min_time:.3f}, max: {max_time:.3f})")
    print(f"Throughput:                 {throughput_mbps:7.2f} Mb/s")
    print(f"Throughput:                 {throughput_mibps:7.2f} MiB/s")
    print(f"Throughput:                 {throughput_MBps:7.2f} MB/s")
    print(f"Seeding (Argon2id) one-time: {seed_time:.3f} s")
    print("=" * 50)

    return avg_time, throughput_mbps

# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Lorenz + Chebyshev Hybrid PRNG")
    parser.add_argument("--salt", type=str, default="hybrid-salt-v1")
    parser.add_argument("--out", type=str, default="hybrid_output.bin")
    parser.add_argument("--bits", type=int, default=1_000_000)
    parser.add_argument("--password", type=str, help="Password")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    if args.password is not None:
        password = args.password
    else:
        while True:
            p1 = input("Enter password: ").strip()
            p2 = input("Confirm password: ").strip()
            if p1 and p1 == p2:
                password = p1
                break
            print("Mismatch. Try again.\n")

    password_bytes = password.encode("utf-8")
    salt_bytes = args.salt.encode("utf-8")

    if args.benchmark:
        benchmark_prng(
            password=password_bytes,
            salt=salt_bytes,
            out_bits=args.bits,
            runs=3
        )
    else:
        generate_hybrid_bits(
            password=password_bytes,
            salt=salt_bytes,
            out_bits=args.bits,
            out_file=args.out
        )