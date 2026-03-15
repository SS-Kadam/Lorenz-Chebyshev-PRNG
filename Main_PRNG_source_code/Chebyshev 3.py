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

"""
chebyshev.py

Chebyshev entropy-source module.

- Derives a 1024-bit OKM from password+salt using Argon2id (requires argon2-cffi).
- Uses the last 256 bits of the OKM for Chebyshev seeding:
    seed_int -> k = 16 + (seed_int % 3)  (k in {16,17,18})
    x0 = 0.1 + u * 0.8  where u = seed_int / 2^256
- Uses mpmath with ~256-bit binary precision.
- Burns in 10_000 iterations then samples, extracting fractional bits from (x+1)/2.
- Produces N raw bits (default 10_000_000) and writes them to a binary file (path returned).

Usage (import):
    from chebyshev import generate_chebyshev_bits
    out_file, meta = generate_chebyshev_bits(password=b"p", salt=b"s", out_bits=10_000_000)

Usage (CLI):
    python chebyshev.py --password mypass --out /mnt/data/cheb_bits_10M.bin
"""

from __future__ import annotations
import os, math, time, sys
from typing import Optional
from binascii import hexlify

# argon2
try:
    from argon2.low_level import hash_secret_raw, Type
    HAVE_ARGON2 = True
except Exception:
    HAVE_ARGON2 = False

# mpmath for high precision
from mpmath import mp, mpf, cos, acos, floor

# ------------------------- Utility functions -------------------------
def derive_okm_argon2(password: bytes, salt: bytes, outlen_bytes: int = 128,
                      time_cost: int = 3, memory_cost_kb: int = 64*1024, parallelism: int = 1) -> bytes:
    """Derive OKM using Argon2id raw output. Requires argon2-cffi."""
    if not HAVE_ARGON2:
        raise RuntimeError("argon2-cffi not installed. Install with: pip install argon2-cffi")
    okm = hash_secret_raw(secret=password,
                          salt=salt,
                          time_cost=time_cost,
                          memory_cost=memory_cost_kb,
                          parallelism=parallelism,
                          hash_len=outlen_bytes,
                          type=Type.ID)
    return okm

def bytes_to_bitstr(b: bytes) -> str:
    return ''.join(f"{byte:08b}" for byte in b)

def bits_from_int_bigendian(val: int, m: int):
    """Return list of bits (0/1) from MSB to LSB for m-bit value."""
    return [(val >> i) & 1 for i in range(m-1, -1, -1)]

# ------------------------- Chebyshev map step -------------------------
def cheb_step(x: mpf, k: int) -> mpf:
    # x in [-1,1]; iterate x_{n+1} = cos(k * arccos(x))
    return cos(mpf(k) * acos(x))

# ------------------------- Main generator -------------------------
def generate_chebyshev_bits(password: Optional[bytes],
                            salt: Optional[bytes],
                            out_bits: int = 10_000_000,
                            bits_per_sample: int = 32,
                            burn_in_steps: int = 10_000,
                            mp_binary_bits: int = 256,
                            okm_bytes: Optional[bytes] = None,
                            okm_time_cost: int = 3,
                            okm_memory_kb: int = 64*1024,
                            okm_parallelism: int = 1,
                            out_file: str = "/mnt/data/cheb_bits_10M.bin"):
    """
    Generate out_bits raw bits from Chebyshev iteration.
    - If okm_bytes (128 bytes) is provided, uses its last 256 bits as seed.
    - Otherwise derives OKM from Argon2id(password, salt).
    Returns (out_file_path, metadata_dict).
    """

    # --- set precision from mp_binary_bits (binary bits -> decimal digits) ---
    mp.dps = int(math.ceil(mp_binary_bits / math.log2(10)))
    # slight safety margin
    # mp.dps = mp.dps + 2

    # --- get OKM (128 bytes) ---
    if okm_bytes is None:
        if password is None or salt is None:
            raise ValueError("Either okm_bytes must be provided, or password and salt must be provided.")
        okm_bytes = derive_okm_argon2(password, salt, outlen_bytes=128,
                                      time_cost=okm_time_cost,
                                      memory_cost_kb=okm_memory_kb,
                                      parallelism=okm_parallelism)
    if len(okm_bytes) != 128:
        raise ValueError("okm_bytes must be exactly 128 bytes (1024 bits).")

    okm_bits = bytes_to_bitstr(okm_bytes)
    # take last 256 bits for Chebyshev seed
    seed_bits = okm_bits[-256:]
    remaining_bits = okm_bits[:-256]  # the rest, if needed

    # interpret seed_int
    seed_int = int(seed_bits, 2)
    two256 = 2**256
    u = mp.mpf(seed_int) / mp.mpf(two256)   # in [0,1)
    # mapping option B: x0 = 0.1 + u * 0.8
    x0 = mp.mpf('0.1') + u * mp.mpf('0.8')

    # determine k in {16,17,18}
    k = 35 + (seed_int % 3)

    # print parameters
    print("Chebyshev parameters mapped from OKM (last 256 bits):")
    print(" seed_int (hex prefix) =", hexlify(okm_bytes[-32:])[:32] if isinstance(okm_bytes, (bytes, bytearray)) else "(okm not bytes)")
    print(" k =", k)
    print(" x0 =", x0)
    print(" mapping: x0 = 0.1 + u * 0.8 (u in [0,1))")
    print(" mp.dps (decimal digits) =", mp.dps, " (approx binary bits ~{:.0f})".format(mp.dps * math.log2(10)))
    print(" burn_in_steps =", burn_in_steps)
    print(" requested out_bits =", out_bits)
    print(" bits_per_sample =", bits_per_sample)
    print(" output file =", out_file)

    # compute how many samples needed
    samples_needed = math.ceil(out_bits / bits_per_sample)

    # prepare output file
    out_dir = os.path.dirname(out_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    f = open(out_file, "wb")

    byte_acc = 0
    nbits_in_acc = 0
    def push_bits_list_to_file(bits_list):
        nonlocal byte_acc, nbits_in_acc
        for b in bits_list:
            byte_acc = (byte_acc << 1) | (1 if b else 0)
            nbits_in_acc += 1
            if nbits_in_acc == 8:
                f.write(bytes((byte_acc,)))
                byte_acc = 0
                nbits_in_acc = 0

    def flush_remainder():
        nonlocal byte_acc, nbits_in_acc
        if nbits_in_acc > 0:
            pad = 8 - nbits_in_acc
            byte_acc = (byte_acc << pad)
            f.write(bytes((byte_acc,)))
            byte_acc = 0
            nbits_in_acc = 0

    # initialize state
    x = mp.mpf(x0)

    # burn-in
    print("Starting burn-in...")
    t0 = time.time()
    for _ in range(burn_in_steps):
        x = cheb_step(x, k)
    t_burn = time.time() - t0
    print(f" Burn-in done in {t_burn:.2f}s")

    # sampling loop
    print(f"Sampling: need {samples_needed} samples (each {bits_per_sample} bits)...")
    samples_collected = 0
    steps_done = 0
    report_every = max(1, samples_needed // 20)
    start_time = time.time()

    for sample_idx in range(samples_needed):
        # iterate one step (we could iterate multiple if desired)
        x = cheb_step(x, k)
        steps_done += 1
        # map x in [-1,1] -> u in [0,1): u = (x + 1) / 2
        u_cur = (x + mp.mpf(1)) / mp.mpf(2)
        # quantize m bits
        val_mpf = floor(u_cur * mp.power(2, bits_per_sample))
        val = int(val_mpf)
        bits = bits_from_int_bigendian(val, bits_per_sample)
        push_bits_list_to_file(bits)
        samples_collected += 1

        if (sample_idx + 1) % report_every == 0:
            elapsed = time.time() - start_time
            rate = samples_collected / elapsed if elapsed > 0 else 0.0
            rem_samples = samples_needed - samples_collected
            est_rem = rem_samples / rate if rate > 0 else float('inf')
            print(f"  samples {samples_collected}/{samples_needed}, elapsed {elapsed:.1f}s, est rem {est_rem:.1f}s")

    flush_remainder()
    f.close()
    total_time = time.time() - start_time
    print(f"Finished generation: wrote {out_bits} bits to {out_file} in {total_time:.2f}s (steps done {steps_done})")

    # return metadata
    meta = {
        "out_file": out_file,
        "okm_last32_hex": hexlify(okm_bytes[-32:]).decode(),
        "k": int(k),
        "x0": x0,
        "mp_dps": mp.dps,
        "burn_in_steps": burn_in_steps,
        "bits_per_sample": bits_per_sample,
        "out_bits": out_bits,
    }
    return out_file, meta

# ------------------------- If run as script -------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Generate Chebyshev chaotic raw bits (high precision).")
    p.add_argument("--password", type=str, help="Password (will be used by Argon2id)")
    p.add_argument("--salt", type=str, default="cheb-salt-v1", help="Salt (string)")
    p.add_argument("--out", type=str, default="/mnt/data/cheb_bits_10M.bin", help="Output binary file")
    p.add_argument("--bits", type=int, default=50_000_000, help="Number of output bits")
    p.add_argument("--bps", type=int, default=32, help="Bits per sample (from fractional part)")
    p.add_argument("--burn", type=int, default=10_000, help="Burn-in iterations")
    p.add_argument("--mpbits", type=int, default=256, help="Binary precision bits for mpmath")
    args = p.parse_args()

    if args.password is None:
        print("\nNo password provided. Print help and exit.\n")
        p.print_help()
        sys.exit(0)

    pw = args.password.encode("utf-8")
    salt_bytes = args.salt.encode("utf-8")

    out_file, meta = generate_chebyshev_bits(password=pw,
                                             salt=salt_bytes,
                                             out_bits=args.bits,
                                             bits_per_sample=args.bps,
                                             burn_in_steps=args.burn,
                                             mp_binary_bits=args.mpbits,
                                             okm_bytes=None,
                                             out_file=args.out)
    print("Meta:", meta)
