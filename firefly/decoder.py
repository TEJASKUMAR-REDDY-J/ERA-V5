"""Getting the bytes back out -- the answer to Problem 5, with zero parameters.

If a spectral code can be turned back into bytes, the |V| x d_model softmax head
disappears and vocabulary size stops costing anything: a 1M-token vocabulary is
free on the output side, and byte strings outside any vocabulary decode too.

    Phi = X . F / sqrt(L)      =>      X = sqrt(L) . Phi . pinv(F)

F[p,k] = z_k^p with distinct nodes z_k = exp(-i w_k) is Vandermonde, so it has
full rank whenever K >= L and the inversion is EXACT.

The one hole in that argument is that the decoder must know L, which a
vocabulary-free head does not. The code already carries it: a uniform grid
includes w = 0, where Phi[c,0] = count(c)/sqrt(L), so summing over the byte axis
gives sqrt(L). No side channel is needed.

-- On the learned decoder that used to live here --

An earlier version of this file carried a Neural Cellular Automaton decoder: a
local, shared, iterated update rule seeded with the spectral code, motivated by
FourierDiff-NCA (arXiv:2401.06291), where Fourier injection gives an NCA global
reach in a single step. The idea was that a learned decoder would beat the
closed form on noisy codes, since a real vocabulary-free head predicts the code
rather than computing it.

It was measured and it lost, in every regime tested, including the
underdetermined one built specifically for it:

    well posed  K=24    matched filter 1.000 exact   NCA 0.742 exact
    compressed  K=10    matched filter 0.991 byte    NCA 0.977 byte

The reasoning behind the prediction was simply wrong. With a uniform grid the
columns of F are orthogonal, so pinv IS a matched filter, and a matched filter
is the optimal estimator under additive white noise. There was never any noise
amplification for a cellular automaton to repair, and a learned decoder cannot
beat the optimal linear estimator at the thing it is optimal at.

So the NCA is gone -- 630k parameters deleted, along with a torch dependency in
this module -- and Problem 5 is answered by six lines of linear algebra. The
full measurement is preserved in results/e5_decoding.json.
"""
from __future__ import annotations

import numpy as np

from .encoder import K_DEFAULT, freqs


def freqs_uniform(k: int) -> np.ndarray:
    """Uniform grid. With K >= L this makes the code a literal DFT of the byte
    indicator, so inversion is an exact IDFT."""
    return 2 * np.pi * np.arange(k) / k


def invertible_grid_for(length: int) -> np.ndarray:
    """Smallest uniform grid that guarantees exact inversion of `length` bytes."""
    return freqs_uniform(max(K_DEFAULT, int(length)))


def infer_length(code: np.ndarray) -> int:
    """Recover L from the code itself. Requires a grid containing w = 0."""
    return int(round(float(np.real(code[:, 0].sum())) ** 2))


def matched_filter_decode(code: np.ndarray, length: int,
                          omega: np.ndarray | None = None) -> bytes:
    """Recover bytes from a spectral code. Exact when K >= length."""
    omega = freqs() if omega is None else omega
    f = np.exp(-1j * np.outer(np.arange(length), omega))     # (L, K)
    x = np.sqrt(length) * (code @ np.linalg.pinv(f))         # (256, L)
    return bytes(np.argmax(x.real, axis=0).astype(np.uint8))


def decode(code: np.ndarray, omega: np.ndarray | None = None) -> bytes:
    """Full zero-parameter round trip: code -> length -> bytes."""
    omega = freqs() if omega is None else omega
    return matched_filter_decode(code, infer_length(code), omega)


def decode_batch(codes: np.ndarray, omega: np.ndarray) -> list[bytes]:
    """codes: real (N, 2*256*K) as produced by encoder.to_real."""
    half = codes.shape[1] // 2
    out = []
    for row in codes:
        cx = row[:half].reshape(256, -1) + 1j * row[half:].reshape(256, -1)
        out.append(decode(cx, omega))
    return out
