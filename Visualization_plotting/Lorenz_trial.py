# =============================================================================
# Copyright (C) 2026 SS Kadam
# All rights reserved.
#
# Visualization code for the chaotic map systems used as the mathematical
# foundation of the PRNG developed by SS Kadam (2026).
#
# Included for transparency and academic reproducibility purposes.
# Not part of the core PRNG implementation.
#
# Development environment : Local (private development machine)
# Development year        : 2026
#
# Licensed under the MIT License.
# See LICENSE file in the project root for full license text.
# =============================================================================



# Python code to perform the requested trial run:
# - Argon2id (if available) -> HKDF expand to 1024 bits
# - Use 6 segments of 128 bits each for sigma,rho,beta,x0,y0,z0 (remaining bits kept)
# - Map segments to ranges using u=I/2^n
# - Use mpmath at ~256-bit binary precision (mp.dps ~ 80)
# - Integrate Lorenz with fixed-step RK4 (dt=1e-3), burn-in 10000 steps
# - Extract fractional bits from x (mantissa-like) to produce a raw bitstream (default 100k bits)
# - Compute approximate largest Lyapunov exponent with two nearby trajectories
# - Plot attractor and print parameters. Output raw bits in hex.
#
# Notes:
# - If argon2 is not installed, falls back to PBKDF2-HMAC-SHA256 for demo. For real experiments use Argon2id.
# - This code is intended to be reproducible. It may be slow depending on mp.dps and number of bits requested.
# - You can change NUM_BITS, BITS_PER_SAMPLE, SAMPLE_GAP, mp.dps etc. at the top.
#
# Run-time: with mp.dps=80 this will be slower than double precision; expect tens of seconds to minutes depending
# on NUM_BITS. Adjust NUM_BITS and BITS_PER_SAMPLE downward for quick tests.


import math, time, hmac, hashlib, sys
from math import floor, ceil
from binascii import hexlify

# Try to import argon2 (argon2-cffi) for Argon2id; fallback to PBKDF2 if not available
try:
    from argon2.low_level import hash_secret_raw, Type
    HAVE_ARGON2 = True
except Exception as e:
    HAVE_ARGON2 = False

# mpmath for arbitrary precision arithmetic
from mpmath import mp, mpf

# plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---------------------- User-tunable parameters ----------------------
PASSWORD = b"example-password"   # replace with your secret password (bytes)
SALT = b"example-salt-0001"      # replace with a secure salt (bytes)
ARGON2_MEM = 64 * 1024           # KB (~64 MB) - adjust for your machine if using Argon2
ARGON2_TIME = 2
ARGON2_PAR = 1

OKM_BYTES = 128  # 1024 bits

# split into 6 segments of 128 bits each for parameters
SEG_BITS = 128
NUM_PARAM_SEGMENTS = 6

# mpmath precision: target ~256 bits binary -> mp.dps ~ 80 decimal digits
TARGET_BINARY_BITS = 256
mp.dps = int(ceil(TARGET_BINARY_BITS / 3.3219280948873626))  # ~bits / log2(10)
# confirm
print(f"mpmath mp.dps={mp.dps} (approx binary bits ~{int(mp.dps*3.3219280948873626)})")

# Lorenz integration settings
DT = mp.mpf('0.001')  # timestep for RK4
BURN_IN_STEPS = 10000
SAMPLE_GAP = 10       # record every SAMPLE_GAP steps
BITS_PER_SAMPLE = 32  # bits extracted per sample from fractional part of x (safe start ~16-32)
NUM_BITS = 100000     # total raw bits target (NIST baseline)
# derived:
SAMPLES_NEEDED = ceil(NUM_BITS / BITS_PER_SAMPLE)

# mapping ranges for Lorenz (as discussed earlier)
SIGMA_RANGE = (mp.mpf('9.0'), mp.mpf('11.0'))
RHO_RANGE   = (mp.mpf('26.0'), mp.mpf('30.0'))
BETA_RANGE  = (mp.mpf('2.0'), mp.mpf('3.0'))
X0_RANGE    = (mp.mpf('-10.0'), mp.mpf('10.0'))
Y0_RANGE    = (mp.mpf('-10.0'), mp.mpf('10.0'))
Z0_RANGE    = (mp.mpf('0.0'), mp.mpf('50.0'))

# Lyapunov exponent estimation settings
LYAP_DELTA = mp.mpf('1e-12')  # initial separation for lyapunov estimate (small)
LYAP_RENORM_EVERY = 10       # renormalize separation every this many recorded samples
# --------------------------------------------------------------------

def derive_okm_argon2(password, salt, outlen_bytes):
    """Derive OKM using Argon2id if available, otherwise PBKDF2-HMAC-SHA256 fallback.
       Returns bytes of length outlen_bytes.
    """
    if HAVE_ARGON2:
        # Argon2id produces raw bytes
        okm = hash_secret_raw(secret=password,
                              salt=salt,
                              time_cost=ARGON2_TIME,
                              memory_cost=ARGON2_MEM,
                              parallelism=ARGON2_PAR,
                              hash_len=outlen_bytes,
                              type=Type.ID)
        return okm
    else:
        # fallback: PBKDF2-HMAC-SHA256 (NOT Argon2 but usable for experiments)
        print("Warning: argon2 not available. Falling back to PBKDF2-HMAC-SHA256 (not Argon2id).")
        # iterations choose reasonably high for demo (but DO NOT rely on this for security)
        iterations = 200000
        okm = hashlib.pbkdf2_hmac('sha256', password, salt, iterations, dklen=outlen_bytes)
        return okm

def hkdf_expand(prk, info=b"", length=OKM_BYTES, hashmod=hashlib.sha256):
    """HKDF-Expand (simple implementation); prk = pseudo-random key from extract stage (we'll treat Argon2 output as prk).
       Returns length bytes.
    """
    hash_len = hashmod().digest_size
    n = ceil(length / hash_len)
    okm = b""
    t = b""
    for i in range(1, n+1):
        t = hmac.new(prk, t + info + bytes([i]), hashmod).digest()
        okm += t
    return okm[:length]

def bytes_to_bitstring(b):
    return ''.join(f"{byte:08b}" for byte in b)

def bits_segment_to_mpf(bitstr):
    """Interpret bitstr as integer I and return mp.mpf(I) / 2^len(bitstr)"""
    n = len(bitstr)
    I = int(bitstr, 2)
    return mp.mpf(I) / mp.power(2, n)

def map_u_to_range(u, low, high):
    return low + u * (high - low)

# RK4 step using mpmath mpf
def lorenz_deriv(x, y, z, sigma, rho, beta):
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return dx, dy, dz

def rk4_step(x, y, z, sigma, rho, beta, dt):
    k1x, k1y, k1z = lorenz_deriv(x, y, z, sigma, rho, beta)
    k2x, k2y, k2z = lorenz_deriv(x + (dt/2)*k1x, y + (dt/2)*k1y, z + (dt/2)*k1z, sigma, rho, beta)
    k3x, k3y, k3z = lorenz_deriv(x + (dt/2)*k2x, y + (dt/2)*k2y, z + (dt/2)*k2z, sigma, rho, beta)
    k4x, k4y, k4z = lorenz_deriv(x + dt*k3x, y + dt*k3y, z + dt*k3z, sigma, rho, beta)
    xn = x + (dt/6) * (k1x + 2*k2x + 2*k3x + k4x)
    yn = y + (dt/6) * (k1y + 2*k2y + 2*k3y + k4y)
    zn = z + (dt/6) * (k1z + 2*k2z + 2*k3z + k4z)
    return xn, yn, zn

def extract_bits_from_mpf_frac(x, low, high, m):
    """Map x to fractional u in [0,1), then extract m bits by repeated doubling.
       Returns list of bits (0/1).
    """
    # map to u = (x - low) / (high - low), then fractional part
    u = (x - low) / (high - low)
    # fractional part
    u = u - mp.floor(u)
    bits = []
    for _ in range(m):
        u = u * 2
        if u >= 1:
            bits.append(1)
            u -= 1
        else:
            bits.append(0)
    return bits

def lyapunov_exponent_estimate(x0, y0, z0, sigma, rho, beta, dt, burn_in, samples, sample_gap, delta0):
    """Rudimentary largest Lyapunov exponent estimate:
       - initialize two trajectories separated by delta0 along x direction
       - after burn-in, for each sample block of sample_gap steps, compute separation, renormalize to delta0,
         accumulate ln(growth) and compute exponent = sum ln(growth) / total_time
    """
    x1 = mp.mpf(x0); y1 = mp.mpf(y0); z1 = mp.mpf(z0)
    x2 = mp.mpf(x0 + delta0); y2 = mp.mpf(y0); z2 = mp.mpf(z0)
    # burn-in both
    for _ in range(burn_in):
        x1, y1, z1 = rk4_step(x1, y1, z1, sigma, rho, beta, dt)
        x2, y2, z2 = rk4_step(x2, y2, z2, sigma, rho, beta, dt)
    sum_log = mp.mpf('0')
    count = 0
    total_time = mp.mpf('0')
    for sample_index in range(samples):
        # evolve sample_gap steps and measure separation
        for _ in range(sample_gap):
            x1, y1, z1 = rk4_step(x1, y1, z1, sigma, rho, beta, dt)
            x2, y2, z2 = rk4_step(x2, y2, z2, sigma, rho, beta, dt)
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1
        dist = mp.sqrt(dx*dx + dy*dy + dz*dz)
        if dist == 0:
            # numerical underflow; reset tiny separation
            dist = mp.mpf('1e-30')
        growth = dist / delta0
        sum_log += mp.log(growth)
        total_time += dt * sample_gap
        # renormalize second trajectory to be delta0 away along direction
        # compute unit vector of separation
        ux = dx / dist; uy = dy / dist; uz = dz / dist
        x2 = x1 + ux * delta0
        y2 = y1 + uy * delta0
        z2 = z1 + uz * delta0
        count += 1
    # average exponent per unit time
    lyap = sum_log / total_time
    return lyap

# -------------------- Main procedure --------------------
start_time = time.time()

# 1) Argon2id -> OKM (1024 bits)
okm = derive_okm_argon2(PASSWORD, SALT, OKM_BYTES)
print(f"Derived OKM bytes length = {len(okm)}")

# 2) HKDF-expand the OKM just to be extra safe (we treat okm as PRK and expand again to ensure uniformity)
#    but here okm already length OKM_BYTES, so just use it (or we can re-expand if desired).
# For simplicity we'll treat okm as source of bits.
bitstr = bytes_to_bitstring(okm)
print(f"Bitstring length = {len(bitstr)} bits")

# 3) partition into six 128-bit segments for sigma,rho,beta,x0,y0,z0
if len(bitstr) < NUM_PARAM_SEGMENTS * SEG_BITS:
    raise ValueError("OKM too short for requested segmentation")

segments = []
offset = 0
for i in range(NUM_PARAM_SEGMENTS):
    seg = bitstr[offset: offset + SEG_BITS]
    segments.append(seg)
    offset += SEG_BITS

# keep remaining bits for future (we'll keep in 'remaining_bits')
remaining_bits = bitstr[offset:]
print(f"Segments extracted: {len(segments)} (each {SEG_BITS} bits). Remaining bits = {len(remaining_bits)} bits")

# 4) map each segment to parameter ranges
u_sigma = bits_segment_to_mpf(segments[0]); sigma = map_u_to_range(u_sigma, SIGMA_RANGE[0], SIGMA_RANGE[1])
u_rho   = bits_segment_to_mpf(segments[1]); rho   = map_u_to_range(u_rho,   RHO_RANGE[0],   RHO_RANGE[1])
u_beta  = bits_segment_to_mpf(segments[2]); beta  = map_u_to_range(u_beta,  BETA_RANGE[0],  BETA_RANGE[1])
u_x0    = bits_segment_to_mpf(segments[3]); x0    = map_u_to_range(u_x0,    X0_RANGE[0],    X0_RANGE[1])
u_y0    = bits_segment_to_mpf(segments[4]); y0    = map_u_to_range(u_y0,    Y0_RANGE[0],    Y0_RANGE[1])
u_z0    = bits_segment_to_mpf(segments[5]); z0    = map_u_to_range(u_z0,    Z0_RANGE[0],    Z0_RANGE[1])

print("Mapped parameter values:")
print(f"  sigma = {sigma}")
print(f"  rho   = {rho}")
print(f"  beta  = {beta}")
print(f"  x0    = {x0}")
print(f"  y0    = {y0}")
print(f"  z0    = {z0}")

# Dither: use first 32 bits of remaining bits as extra tiny perturbation for x0,y0,z0
if len(remaining_bits) >= 32:
    dseg = remaining_bits[:32]
    dval = int(dseg, 2) / (2**32)
    eps = mp.mpf(dval) * mp.mpf('1e-20')  # very small epsilon relative to precision
    x0 += eps; y0 += eps; z0 += eps
    print(f"Applied tiny dither eps={eps} to initial conditions.")

# 5) Integrate Lorenz with RK4: burn-in then sample
print("Starting integration (burn-in + sampling)...")
x = mp.mpf(x0); y = mp.mpf(y0); z = mp.mpf(z0)
# burn-in
t0 = time.time()
for i in range(BURN_IN_STEPS):
    x, y, z = rk4_step(x, y, z, sigma, rho, beta, DT)
t_burn_time = time.time() - t0
print(f"Burn-in of {BURN_IN_STEPS} steps complete (wall time {t_burn_time:.2f}s).")

# sampling loop: extract bits from fractional part of x (and optionally y,z)
raw_bits = []
sampled_points = []  # store for plotting (you may limit to some number to save memory)
samples_collected = 0
steps_done = 0
# we'll also collect a smaller number of points for plotting, e.g., first 20000 samples
PLOT_POINTS_LIMIT = 20000

while samples_collected < SAMPLES_NEEDED:
    # advance SAMPLE_GAP steps
    for _ in range(SAMPLE_GAP):
        x, y, z = rk4_step(x, y, z, sigma, rho, beta, DT)
        steps_done += 1
    # extract bits from x fractional part
    bits = extract_bits_from_mpf_frac(x, X0_RANGE[0], X0_RANGE[1], BITS_PER_SAMPLE)
    raw_bits.extend(bits)
    samples_collected += 1
    if len(sampled_points) < PLOT_POINTS_LIMIT:
        sampled_points.append((float(x), float(y), float(z)))
# truncate to exact NUM_BITS
raw_bits = raw_bits[:NUM_BITS]
print(f"Collected {len(raw_bits)} raw bits in total. Sampling steps done: {steps_done}.")

# convert raw bits list to bytes and hex
def bits_to_bytes_hex(bits):
    # pad to full bytes
    extra = (8 - (len(bits) % 8)) % 8
    bits_padded = bits + [0]*extra
    b = bytearray()
    for i in range(0, len(bits_padded), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits_padded[i+j]
        b.append(byte)
    return bytes(b), hexlify(bytes(b)).decode()

raw_bytes, raw_hex = bits_to_bytes_hex(raw_bits)
print(f"Raw output (hex, first 200 chars): {raw_hex[:200]} ...")

# 6) Estimate largest Lyapunov exponent (approx)
print("Estimating largest Lyapunov exponent (this may take some time)...")
lyap_start = time.time()
# for the Lyapunov estimate we will use the same sample count but a smaller sample gap to get smoother average
lyap_samples = max(100, SAMPLES_NEEDED // 4)
lyap = lyapunov_exponent_estimate(x0, y0, z0, sigma, rho, beta, DT, BURN_IN_STEPS, lyap_samples, SAMPLE_GAP, LYAP_DELTA)
lyap_time = time.time() - lyap_start
print(f"Estimated largest Lyapunov exponent ≈ {lyap} (computed in {lyap_time:.2f}s)")

# 7) Plot attractor (use sampled_points)
print("Plotting attractor (a subset of sampled points)...")
xs = [p[0] for p in sampled_points]
ys = [p[1] for p in sampled_points]
zs = [p[2] for p in sampled_points]

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(xs, ys, zs, lw=0.3)
ax.set_title("Lorenz attractor (sampled subset)")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
plt.tight_layout()
plt.show()

end_time = time.time()
print(f"Total script wall time: {end_time - start_time:.2f} seconds")

# Summarize outputs for user
result = {
    "sigma": sigma,
    "rho": rho,
    "beta": beta,
    "x0": x0,
    "y0": y0,
    "z0": z0,
    "raw_bits_len": len(raw_bits),
    "raw_hex_prefix": raw_hex[:256],
    "lyapunov": lyap,
    "mp_dps": mp.dps,
    "dt": DT,
    "burn_in_steps": BURN_IN_STEPS,
    "sample_gap": SAMPLE_GAP,
    "bits_per_sample": BITS_PER_SAMPLE,
}
print(result)