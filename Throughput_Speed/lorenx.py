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
from mpmath import mp, mpf

# Fast local aliases (speed optimization)
mpf_local = mpf
mp_floor = mp.floor
mp_power = mp.power

# ------------------------- Utility functions -------------------------
def derive_okm_argon2(password: bytes, salt: bytes, outlen_bytes: int = 128,
                      time_cost: int = 3, memory_cost_kb: int = 64 * 1024, parallelism: int = 1) -> bytes:
    """
    Derive OKM using Argon2id raw output. Requires argon2-cffi.
    Returns exactly outlen_bytes bytes.
    """
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

def bitsegment_to_mpf(bitstr: str) -> mpf:
    """Interpret n-bit segment as integer I and return mp.mpf(I)/2^n."""
    n = len(bitstr)
    I = int(bitstr, 2)
    return mp.mpf(I) / mp.power(2, n)

def map_u_to_range(u: mpf, low: mpf, high: mpf) -> mpf:
    return low + u * (high - low)

# ------------------------- Lorenz integrator -------------------------
def lorenz_deriv(x: mpf, y: mpf, z: mpf, sigma: mpf, rho: mpf, beta: mpf):
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return dx, dy, dz

def rk4_step(x: mpf, y: mpf, z: mpf, sigma: mpf, rho: mpf, beta: mpf, dt: mpf):
    k1x, k1y, k1z = lorenz_deriv(x, y, z, sigma, rho, beta)
    k2x, k2y, k2z = lorenz_deriv(x + (dt/2)*k1x, y + (dt/2)*k1y, z + (dt/2)*k1z, sigma, rho, beta)
    k3x, k3y, k3z = lorenz_deriv(x + (dt/2)*k2x, y + (dt/2)*k2y, z + (dt/2)*k2z, sigma, rho, beta)
    k4x, k4y, k4z = lorenz_deriv(x + dt*k3x, y + dt*k3y, z + dt*k3z, sigma, rho, beta)
    xn = x + (dt/6) * (k1x + 2*k2x + 2*k3x + k4x)
    yn = y + (dt/6) * (k1y + 2*k2y + 2*k3y + k4y)
    zn = z + (dt/6) * (k1z + 2*k2z + 2*k3z + k4z)
    return xn, yn, zn

# ------------------------- Extraction primitives -------------------------

def extract_m_bits_from_mpf_fraction(x: mpf, low: mpf, high: mpf, m: int):
    """
    Map x to fractional u in [0,1), then return integer floor(u * 2^m)
    """
    u = (x - low) / (high - low)
    u = u - mp_floor(u)
    val = mp_floor(u * mp_power(2, m))
    return int(val)


def bits_from_int_bigendian(val: int, m: int):
    """Return list of bits (0/1) from MSB to LSB for m-bit value."""
    bits = [(val >> i) & 1 for i in range(m-1, -1, -1)]
    return bits

# ------------------------- Main generator -------------------------
def generate_lorenx_bits(password: bytes,
                         salt: bytes,
                         out_bits: int = 10_000_000,
                         bits_per_sample: int = 32,
                         sample_gap: int = 5,
                         dt: float = 1e-3,
                         burn_in_steps: int = 10_000,
                         mp_binary_bits: int = 128,
                         okm_bytes: Optional[bytes] = None,
                         okm_time_cost: int = 3,
                         okm_memory_kb: int = 64*1024,
                         okm_parallelism: int = 1,
                         out_file: str = "/mnt/data/lorenz_bits_1M.bin",
                         benchmark_mode: bool = False):
    """
    Generate out_bits raw bits from high-precision Lorenz RK4. Writes binary output file and returns path.
    - password, salt: bytes for Argon2id
    - bits_per_sample: how many fractional bits to extract from x per sample
    - sample_gap: number of RK4 steps between recorded samples
    - mp_binary_bits: desired binary precision (e.g., 256)
    - okm_bytes: optional precomputed 128-byte OKM (if provided, Argon2id is not called)
    """

    # === set mp.dps from mp_binary_bits ===
    mp.dps = int(math.ceil(mp_binary_bits / math.log2(10)))  # mp.dps is decimal digits; approx bits/log2(10)



    # small correction to ensure we are not under-provisioned
    # (mp.dps will be at least ceil(bits/log2(10)))
    # print precision info
    # print(f"mp.dps set to {mp.dps} (approx binary bits {int(mp.dps * math.log2(10))})")

    # === derive OKM ===
    if okm_bytes is None:
        okm_bytes = derive_okm_argon2(password, salt, outlen_bytes=128,
                                      time_cost=okm_time_cost,
                                      memory_cost_kb=okm_memory_kb,
                                      parallelism=okm_parallelism)
    if len(okm_bytes) != 128:
        raise ValueError("OKM must be exactly 128 bytes (1024 bits)")

    okm_bits = bytes_to_bitstr(okm_bytes)
    # partition 6 segments × 128 bits = 768 bits for Lorenz
    if len(okm_bits) < 6 * 128:
        raise RuntimeError("OKM too short")
    segs = [okm_bits[i*128:(i+1)*128] for i in range(6)]
    remaining_bits = okm_bits[6*128:]

    # mapping ranges (adjustable if you want)
    sigma_range = (mp.mpf('9.0'), mp.mpf('11.0'))
    rho_range   = (mp.mpf('26.0'), mp.mpf('30.0'))
    beta_range  = (mp.mpf('2.0'), mp.mpf('3.0'))
    x0_range    = (mp.mpf('-10.0'), mp.mpf('10.0'))
    y0_range    = (mp.mpf('-10.0'), mp.mpf('10.0'))
    z0_range    = (mp.mpf('0.0'), mp.mpf('50.0'))

    # interpret segments
    u_sigma = bitsegment_to_mpf(segs[0]); sigma = map_u_to_range(u_sigma, sigma_range[0], sigma_range[1])
    u_rho   = bitsegment_to_mpf(segs[1]); rho   = map_u_to_range(u_rho,   rho_range[0],   rho_range[1])
    u_beta  = bitsegment_to_mpf(segs[2]); beta  = map_u_to_range(u_beta,  beta_range[0],  beta_range[1])
    u_x0    = bitsegment_to_mpf(segs[3]); x0    = map_u_to_range(u_x0,    x0_range[0],    x0_range[1])
    u_y0    = bitsegment_to_mpf(segs[4]); y0    = map_u_to_range(u_y0,    y0_range[0],    y0_range[1])
    u_z0    = bitsegment_to_mpf(segs[5]); z0    = map_u_to_range(u_z0,    z0_range[0],    z0_range[1])

    # optional tiny dither from remaining bits (if present)
    eps = mp.mpf('0')
    if len(remaining_bits) >= 32:
        extra = remaining_bits[:32]
        eps_val = int(extra, 2) / (2**32)
        # scale epsilon relative to precision scale (very small)
        eps = mp.mpf(str(eps_val)) * mp.power(10, -20)
        x0 += eps; y0 += eps; z0 += eps

    # print mapped parameters to stdout for reproducibility
    print("Lorenz parameters mapped from OKM:")
    print(" sigma =", sigma)
    print(" rho   =", rho)
    print(" beta  =", beta)
    print(" x0    =", x0)
    print(" y0    =", y0)
    print(" z0    =", z0)
    if eps != 0:
        print(" tiny dither eps =", eps)

    # compute how many samples needed
    samples_needed = math.ceil(out_bits / bits_per_sample)

    # open output file for binary write
    out_dir = os.path.dirname(out_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    if not benchmark_mode:
        f = open(out_file, "wb")
    else:
        f = None

    # function to flush accumulated bits to file as bytes
    byte_acc = 0
    nbits_in_acc = 0
    def push_bits_list_to_file(bits_list):
        nonlocal byte_acc, nbits_in_acc
        for b in bits_list:
            byte_acc = (byte_acc << 1) | (1 if b else 0)
            nbits_in_acc += 1
            if nbits_in_acc == 8:
                if f:
                    f.write(bytes((byte_acc,)))
                byte_acc = 0
                nbits_in_acc = 0

    def flush_remainder():
        nonlocal byte_acc, nbits_in_acc
        if nbits_in_acc > 0:
            # pad the last byte with zeros in LSB
            pad = 8 - nbits_in_acc
            byte_acc = (byte_acc << pad)
            if f:
                f.write(bytes((byte_acc,)))
            byte_acc = 0
            nbits_in_acc = 0

    # initialize integrator state
    x = mp.mpf(x0); y = mp.mpf(y0); z = mp.mpf(z0)
    dt_mpf = mp.mpf(str(dt))

    # burn-in
    print(f"Burn-in: {burn_in_steps} RK4 steps (dt={dt}) ...", flush=True)
    t0 = time.time()
    for _ in range(burn_in_steps):
        x, y, z = rk4_step(x, y, z, sigma, rho, beta, dt_mpf)
    t_burn = time.time() - t0
    print(f" Burn-in done in {t_burn:.2f}s")

    # sampling loop
    print(f"Sampling: need {samples_needed} samples (bits_per_sample={bits_per_sample}, sample_gap={sample_gap})")
    samples_collected = 0
    steps_done = 0
    report_every = max(1, samples_needed // 10)
    start_time = time.time()
    while samples_collected < samples_needed:
        # advance sample_gap steps
        for _ in range(sample_gap):
            x, y, z = rk4_step(x, y, z, sigma, rho, beta, dt_mpf)
            steps_done += 1
        # extract bits_per_sample bits from fractional part of x (MSB->LSB)
        val = extract_m_bits_from_mpf_fraction(x, x0_range[0], x0_range[1], bits_per_sample)
        bits = bits_from_int_bigendian(val, bits_per_sample)
        push_bits_list_to_file(bits)
        samples_collected += 1
        # progress
        if samples_collected % report_every == 0:
            elapsed = time.time() - start_time
            rate = samples_collected / elapsed
            rem = samples_needed - samples_collected
            est = rem / rate if rate > 0 else float('inf')
            print(f"  samples {samples_collected}/{samples_needed}, elapsed {elapsed:.1f}s, est remaining {est:.1f}s", flush=True)

    flush_remainder()
    if f:
        f.close()
    total_time = time.time() - start_time
    print(f"Finished generation: wrote {out_bits} bits to {out_file} in {total_time:.2f}s (steps done {steps_done})")

    # also return hex prefix and some metadata
    hk_hex_prefix = hexlify(okm_bytes[:16]).decode()
    meta = {
        "out_file": out_file,
        "okm_hex_prefix": hk_hex_prefix,
        "sigma": sigma,
        "rho": rho,
        "beta": beta,
        "x0": x0,
        "y0": y0,
        "z0": z0,
        "mp_dps": mp.dps,
        "dt": dt_mpf,
        "burn_in_steps": burn_in_steps,
        "sample_gap": sample_gap,
        "bits_per_sample": bits_per_sample,
        "out_bits": out_bits,
    }
    return out_file, meta

# ------------------------- If run as script -------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Generate Lorenz chaotic raw bits (high precision RK4). "
                    "If no arguments are supplied, script only prints help."
    )

    # make password optional
    p.add_argument("--password", type=str, help="Password (will be used by Argon2id)")
    p.add_argument("--salt", type=str, default="lorenz-salt-v1", help="Salt (string)")
    p.add_argument("--out", type=str, default="/mnt/data/lorenz_bits_1M.bin", help="Output binary file")
    p.add_argument("--bits", type=int, default=10_000_000, help="Number of output bits")
    p.add_argument("--bps", type=int, default=32, help="Bits per sample (from fractional part of x)")
    p.add_argument("--gap", type=int, default=5, help="RK4 steps between recorded samples")
    p.add_argument("--dt", type=float, default=1e-3, help="RK4 timestep")
    p.add_argument("--mpbits", type=int, default=128, help="Binary precision bits for mpmath")

    args = p.parse_args()

    # if run without password → show help and exit safely
    if args.password is None:
        print("\nNo password provided. Running in 'help-only' mode.\n")
        p.print_help()
        sys.exit(0)

    pw = args.password.encode("utf-8")
    salt_bytes = args.salt.encode("utf-8")

    out_file, meta = generate_lorenx_bits(
        password=pw, salt=salt_bytes,
        out_bits=args.bits,
        bits_per_sample=args.bps,
        sample_gap=args.gap,
        dt=args.dt,
        burn_in_steps=10000,
        mp_binary_bits=args.mpbits,
        out_file=args.out
    )

    print("Meta:", meta)