"""FIREFLY numeric block -- mathematics stored *as* structure, not learned.

The same trick as the encoder, applied to the integers instead of to bytes:
identity carried in phase, composition performed by phase addition. A firefly
signals by the rhythm of its flashes and synchronises with its neighbours by
adding phase; encode an integer as its phase against several independent
rhythms and composing two integers becomes multiplying two phasors.

The rhythms are chosen COPRIME so the representation is unambiguous over the
product of the moduli -- that part is the Chinese Remainder Theorem, not
biology, and it is what makes the decode exact rather than approximate.

Two blocks, both frozen, both tiny:

  ADD block   z_m(n) = exp(2*pi*i * n / P_m)
              z(a) * z(b) = exp(2*pi*i*(a+b)/P_m) = z(a+b)

  MUL block   u_m(n) = 0                              if n = 0 (mod P_m)
                     = exp(2*pi*i * ind_g(n) / (P_m-1))  otherwise
              u(a) * u(b) = u(a*b)
              because Z_P* is cyclic of order P-1, so the discrete logarithm
              turns multiplication into addition of indices.

In both cases ELEMENTWISE COMPLEX MULTIPLICATION OF THE EMBEDDINGS IS THE
ARITHMETIC. Nothing is trained and nothing is approximated: 9 + 9 lands exactly
on 18 and 9 * 9 lands exactly on 81, recovered by the Chinese Remainder Theorem.

Crucially there are no carries. Residues against coprime moduli are mutually
independent, so the hard part of multi-digit addition -- carry propagation, the
thing that breaks length generalisation in digit-tokenised models -- simply does
not exist in this representation.

Cost: 10 primes x 2 blocks x 2 (re, im) = 40 real dimensions, appended to the
8192-d spectral code. Exact for results below prod(PRIMES) = 6,469,693,230.
"""
from __future__ import annotations

import numpy as np

# Twelve primes rather than ten: ten stops at 6.47e9, which is short of the
# largest product this project tests (99999 * 99999 = 1.0e10). Extending to 37
# costs 8 extra dimensions and removes an awkward "except the top-right cell"
# caveat from the multiplication table.
PRIMES: tuple[int, ...] = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
MODULUS = int(np.prod(np.array(PRIMES, dtype=object)))     # 7_420_738_134_810
N_NUMERIC_DIMS = len(PRIMES) * 2 * 2                       # add + mul, re + im


# --------------------------------------------------------------------------
# Chinese Remainder Theorem
# --------------------------------------------------------------------------
def crt(residues, mods=PRIMES) -> int:
    """Reconstruct the unique integer in [0, prod(mods)) from its residues."""
    total, m_all = 0, 1
    for m in mods:
        m_all *= int(m)
    for r, m in zip(residues, mods):
        m, r = int(m), int(r)
        mi = m_all // m
        total += r * mi * pow(mi, -1, m)
    return total % m_all


# --------------------------------------------------------------------------
# additive block: phase = residue
# --------------------------------------------------------------------------
def add_encode(n: int, primes=PRIMES) -> np.ndarray:
    p = np.array(primes, dtype=np.float64)
    return np.exp(2j * np.pi * (int(n) % np.array(primes)) / p)


def add_decode(z: np.ndarray, primes=PRIMES) -> int:
    p = np.array(primes)
    res = np.round(np.angle(z) / (2 * np.pi) * p).astype(np.int64) % p
    return crt(res, primes)


# --------------------------------------------------------------------------
# multiplicative block: phase = discrete logarithm
# --------------------------------------------------------------------------
def _primitive_root(p: int) -> int:
    """Smallest generator of Z_p*. p = 2 degenerates to the trivial group {1}."""
    if p == 2:
        return 1
    factors = set()
    phi, n = p - 1, p - 1
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    for g in range(2, p):
        if all(pow(g, phi // f, p) != 1 for f in factors):
            return g
    raise ValueError(f"no primitive root for {p}")


def _tables(primes=PRIMES):
    """(index table, power table) per prime. ind[p][a] = discrete log of a."""
    ind, pw = {}, {}
    for p in primes:
        g = _primitive_root(p)
        ind_p = {}
        pw_p = {}
        for e in range(p - 1):
            a = pow(g, e, p)
            pw_p[e] = a
            ind_p.setdefault(a, e)
        ind[p], pw[p] = ind_p, pw_p
    return ind, pw


_IND, _POW = _tables()


def mul_encode(n: int, primes=PRIMES) -> np.ndarray:
    """Zero magnitude encodes residue 0 -- which is exactly right, because
    0 * anything = 0 propagates correctly under elementwise multiplication."""
    out = np.zeros(len(primes), dtype=np.complex128)
    for i, p in enumerate(primes):
        r = int(n) % p
        if r == 0:
            continue
        out[i] = np.exp(2j * np.pi * _IND[p][r] / (p - 1))
    return out


def mul_decode(u: np.ndarray, primes=PRIMES, tol: float = 1e-6) -> int:
    res = []
    for i, p in enumerate(primes):
        if abs(u[i]) < tol:
            res.append(0)
            continue
        e = int(np.round(np.angle(u[i]) / (2 * np.pi) * (p - 1))) % (p - 1)
        res.append(_POW[p][e])
    return crt(res, primes)


# --------------------------------------------------------------------------
# the operator: elementwise complex multiply, for both blocks
# --------------------------------------------------------------------------
def compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The single fixed bilinear op. On the ADD block it performs integer
    addition; on the MUL block, integer multiplication. No parameters."""
    return a * b


def conj_inverse(a: np.ndarray) -> np.ndarray:
    """Conjugation inverts the group operation: subtraction / division."""
    return np.conj(a)


# --------------------------------------------------------------------------
# packed real feature vector for feeding a model
# --------------------------------------------------------------------------
def numeric_block(n: int, primes=PRIMES) -> np.ndarray:
    """40 real dims: [Re add | Im add | Re mul | Im mul]."""
    a, m = add_encode(n, primes), mul_encode(n, primes)
    return np.concatenate([a.real, a.imag, m.real, m.imag])


def numeric_block_batch(ns, primes=PRIMES) -> np.ndarray:
    return np.stack([numeric_block(int(n), primes) for n in np.ravel(ns)])


def residues(n: int, primes=PRIMES) -> np.ndarray:
    """The integer as its residue vector -- the natural OUTPUT space.

    E2 found that putting exact structure in the embedding buys nothing while
    the readout is still digit tokens: the bottleneck simply moves to the output
    side, where carries and digit counts come back. Predicting residues instead
    keeps the representation consistent at both ends, and a residue does not
    depend on how many digits the number has.
    """
    return np.array([int(n) % p for p in primes], dtype=np.int64)


# --------------------------------------------------------------------------
# convenience: full round trip used by the demo and the tests
# --------------------------------------------------------------------------
def exact_add(a: int, b: int) -> int:
    return add_decode(compose(add_encode(a), add_encode(b)))


def exact_sub(a: int, b: int) -> int:
    return add_decode(compose(add_encode(a), conj_inverse(add_encode(b))))


def exact_mul(a: int, b: int) -> int:
    return mul_decode(compose(mul_encode(a), mul_encode(b)))
