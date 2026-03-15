# =============================================================================
# Copyright (C) 2026 SS Kadam
# All rights reserved.
#
# This software and its source code are the original work of SS Kadam,
# developed independently as part of original research into chaos-based
# pseudo-random number generation (PRNG).
#
# Development environment : Local (private development machine)
# Development year        : 2025/26
#
# Licensed under the MIT License.
# See LICENSE file in the project root for full license text.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#   1. Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#
#   2. Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#   3. Neither the name of SS Kadam nor the names of any contributors may
#      be used to endorse or promote products derived from this software
#      without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED. IN NO EVENT SHALL SS KADAM BE LIABLE FOR ANY CLAIM, DAMAGES,
# OR OTHER LIABILITY ARISING FROM THE USE OF THIS SOFTWARE.
# =============================================================================

from __future__ import annotations
import os
import math
from typing import Optional
from binascii import hexlify

# Import your two modules
from Lorenz import generate_lorenz_bits, derive_okm_argon2
from Chebyshev import generate_chebyshev_bits

# ----------------------------------------------------------------------
# Helper: XOR two equal-length binary files
# ----------------------------------------------------------------------
def xor_two_files(file_a: str, file_b: str, out_file: str) -> None:
    with open(file_a, "rb") as fa, open(file_b, "rb") as fb, open(out_file, "wb") as fo:
        while True:
            a = fa.read(1 << 20)
            b = fb.read(len(a))
            if len(a) != len(b):
                raise ValueError(f"Files have different sizes: {file_a} vs {file_b}")
            if not a:
                break
            fo.write(bytes(x ^ y for x, y in zip(a, b)))

# ----------------------------------------------------------------------
# Main hybrid generator
# ----------------------------------------------------------------------
def generate_hybrid_bits(
    password: bytes,
    salt: bytes,
    out_bits: int = 1_000_000,
    bits_per_sample: int = 32,
    lorenz_gap: int = 10,
    cheb_gap: int = 1,
    dt: float = 1e-3,
    burn_in_lorenz: int = 10_000,
    burn_in_cheb: int = 10_000,
    mp_binary_bits: int = 256,
    out_file: str = "/mnt/data/hybrid_lorenz_chebyshev.bin",
    cleanup: bool = True,
) -> tuple[str, dict]:
    okm_bytes = derive_okm_argon2(password, salt, outlen_bytes=128)

    temp_dir = os.path.dirname(out_file) or "."
    lorenz_file = os.path.join(temp_dir, "temp_lorenz.bin")
    cheb_file   = os.path.join(temp_dir, "temp_cheb.bin")

    print("Generating Lorenz bits...")
    generate_lorenz_bits(
        password=password, salt=salt,
        out_bits=out_bits,
        bits_per_sample=bits_per_sample,
        sample_gap=lorenz_gap,
        dt=dt,
        burn_in_steps=burn_in_lorenz,
        mp_binary_bits=mp_binary_bits,
        okm_bytes=okm_bytes,
        out_file=lorenz_file,
    )

    print("Generating Chebyshev bits...")
    generate_chebyshev_bits(
        password=None, salt=None,
        out_bits=out_bits,
        bits_per_sample=bits_per_sample,
        burn_in_steps=burn_in_cheb,
        mp_binary_bits=mp_binary_bits,
        okm_bytes=okm_bytes,
        out_file=cheb_file,
    )

    print(f"XORing → {out_file}")
    xor_two_files(lorenz_file, cheb_file, out_file)

    if cleanup:
        for f in (lorenz_file, cheb_file):
            try: os.remove(f)
            except: pass

    meta = {
        "hybrid_file": out_file,
        "okm_hex_prefix": hexlify(okm_bytes[:16]).decode(),
        "out_bits": out_bits,
    }
    print(f"Done: {out_file}")
    return out_file, meta


# ----------------------------------------------------------------------
# CLI: Interactive Password (NO getpass)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Lorenz XOR Chebyshev Hybrid PRNG")
    p.add_argument("--salt", type=str, default="hybrid-salt-v1")
    p.add_argument("--out", type=str, default="hybrid_output.bin")
    p.add_argument("--bits", type=int, default=1_000_000)
    p.add_argument("--bps", type=int, default=32)
    p.add_argument("--lgap", type=int, default=10)
    p.add_argument("--cgap", type=int, default=1)
    p.add_argument("--dt", type=float, default=1e-3)
    p.add_argument("--mpbits", type=int, default=256)
    p.add_argument("--password", type=str, help="Password (if omitted → prompt)")

    args = p.parse_args()

    if args.password is not None:
        password = args.password
    else:
        while True:
            p1 = input("Enter password: ").strip()
            p2 = input("Confirm password: ").strip()
            if p1 and p1 == p2:
                password = p1
                print()
                break
            print("Mismatch or empty. Try again.\n")

    generate_hybrid_bits(
        password=password.encode("utf-8"),
        salt=args.salt.encode("utf-8"),
        out_bits=args.bits,
        bits_per_sample=args.bps,
        lorenz_gap=args.lgap,
        cheb_gap=args.cgap,
        dt=args.dt,
        mp_binary_bits=args.mpbits,
        out_file=args.out,
    )