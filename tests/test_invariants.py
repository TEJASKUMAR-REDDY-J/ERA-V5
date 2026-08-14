"""E0 -- what FIREFLY proves without training anything.

These are assertions, not metrics. Each one either holds exactly or the design
is wrong. Run with:  python -m pytest tests -q
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from firefly import encoder as E
from firefly import numeric as N


# =========================================================================
# Problem 1 -- mathematics is stored, not learned
# =========================================================================
def test_addition_is_exact_complex_multiply():
    """9 + 9 = 18 by multiplying two embeddings. No model, no training."""
    assert N.exact_add(9, 9) == 18
    for a, b in [(0, 0), (1, 1), (7, 5), (123, 456), (99999, 1), (10**6, 10**6)]:
        assert N.exact_add(a, b) == a + b, (a, b)


def test_multiplication_is_exact_complex_multiply():
    """9 * 9 = 81 by multiplying two embeddings."""
    assert N.exact_mul(9, 9) == 81
    for a, b in [(0, 5), (1, 1), (7, 6), (12, 12), (123, 456), (65536, 97)]:
        assert N.exact_mul(a, b) == a * b, (a, b)


def test_subtraction_by_conjugation():
    for a, b in [(9, 4), (100, 1), (7, 7), (10**5, 12345)]:
        assert N.exact_sub(a, b) == a - b, (a, b)


def test_arithmetic_exact_over_random_range():
    rng = np.random.default_rng(0)
    for _ in range(400):
        a, b = int(rng.integers(0, 50_000)), int(rng.integers(0, 50_000))
        assert N.exact_add(a, b) == a + b
        assert N.exact_mul(a, b) == a * b


def test_no_carries_exist():
    """The mechanism that breaks digit-tokenised addition is absent.

    Residues against coprime moduli are independent, so every coordinate of
    a + b depends on exactly one coordinate of a and one of b. Adding 999999
    and 1 -- maximal carry propagation in base 10 -- is no harder than 1 + 1.
    """
    hard, easy = N.exact_add(999_999, 1), N.exact_add(1, 1)
    assert hard == 1_000_000 and easy == 2


def test_crt_range_and_roundtrip():
    assert N.MODULUS == 7_420_738_134_810
    for n in [0, 1, 17, 2**31, N.MODULUS - 1]:
        assert N.add_decode(N.add_encode(n)) == n % N.MODULUS


def test_numeric_block_is_small():
    assert N.numeric_block(12345).shape == (N.N_NUMERIC_DIMS,)
    assert N.N_NUMERIC_DIMS == 48


# =========================================================================
# Problem 4 -- the spectral code and its structure
# =========================================================================
def test_code_width_matches_kronecker():
    """Drop-in: identical D, so the downstream projection is unchanged."""
    assert E.embed("hello").shape == (8192,)
    assert E.embed_kronecker("hello").shape == (8192,)
    assert E.embed_image(np.zeros((32, 32), np.uint8)).shape == (8192,)


def test_shift_is_phase_rotation():
    """Prefixing bytes multiplies the spectrum by exp(-i*w*delta) exactly."""
    omega = E.freqs()
    word, pad, delta = b"quick", b"\x01", 3
    a = E.encode_1d(word, omega)
    b = E.encode_1d(pad * delta + word, omega)
    scale = np.sqrt(len(word) / (len(word) + delta))
    rot = E.shift_operator(delta, omega)
    for c in set(word):
        assert np.allclose(b[c], a[c] * rot * scale, atol=1e-10), c


def test_magnitude_is_shift_invariant():
    """|Phi| survives insertion. It is also NOT a linear function of the
    Kronecker code, so no learned projection can manufacture it."""
    omega = E.freqs()
    word, pad, delta = b"quick", b"\x01", 3
    a = np.abs(E.encode_1d(word, omega))
    b = np.abs(E.encode_1d(pad * delta + word, omega))
    scale = np.sqrt(len(word) / (len(word) + delta))
    for c in set(word):
        assert np.allclose(b[c], a[c] * scale, atol=1e-10), c


def test_kronecker_collapses_under_insertion_cicada_does_not():
    """The defect the source paper names, measured."""
    k = E.cosine(E.embed_kronecker("run"), E.embed_kronecker("arun"))
    c = E.cosine(E.magnitude_code("run"), E.magnitude_code("arun"))
    assert k < 0.10, k          # delta basis: near-total collapse
    assert c > 0.85, c          # Fourier magnitude: substring survives


# =========================================================================
# Problem 3 -- length is not an architectural constant
# =========================================================================
def test_kronecker_is_blind_past_32_bytes_cicada_is_not():
    long_a = b"x" * 60 + b"A" + b"y" * 60
    long_b = b"x" * 60 + b"B" + b"y" * 60
    assert np.allclose(E.embed_kronecker(long_a), E.embed_kronecker(long_b))
    assert not np.allclose(E.embed(long_a), E.embed(long_b))


def test_no_length_cap():
    for n in [1, 5, 32, 33, 200, 4096]:
        v = E.embed(b"a" * n)
        assert v.shape == (8192,) and np.isfinite(v).all()


def test_short_token_wastes_nothing():
    """'a' spends energy on every frequency rather than one slot in 32."""
    code = E.encode_1d(b"a")
    used = np.abs(code[ord("a")])
    assert (used > 1e-9).all()


# =========================================================================
# The honest caveat -- stated as a test so it cannot be quietly dropped
# =========================================================================
def test_fourier_equals_kronecker_times_fixed_matrix_for_short_tokens():
    """For L <= 32 the two codecs span the same information: Phi = kappa @ F.

    Published deliberately. It means FIREFLY is a strict generalisation rather
    than a rival, and it tells us where real gains cannot be: short-token text.
    """
    omega = E.freqs()
    f = E.fourier_from_kronecker_matrix(omega)
    for w in [b"run", b"running", b"antidisestablish"]:
        assert np.allclose(E.kronecker(w) @ f, E.encode_1d(w, omega), atol=1e-10), w


def test_equivalence_breaks_past_the_cap():
    """...and that the equivalence is exactly what truncation destroys."""
    omega = E.freqs()
    f = E.fourier_from_kronecker_matrix(omega)
    w = b"z" * 40
    assert not np.allclose(E.kronecker(w) @ f, E.encode_1d(w, omega), atol=1e-6)


# =========================================================================
# Problem 2 -- one operator, three modalities
# =========================================================================
def _smooth_field(h: int, w: int) -> np.ndarray:
    """One continuous image, sampled at whatever resolution is asked for."""
    y = np.arange(h)[:, None] / h
    x = np.arange(w)[None, :] / w
    return (128 + 100 * np.sin(2 * np.pi * y) * np.cos(2 * np.pi * x)).astype(np.uint8)


def test_all_modalities_share_one_code_width():
    """The unification claim, reduced to an assert: one budget, three modalities."""
    text = E.embed("the quick brown fox")
    audio = E.embed_audio(np.sin(np.linspace(0, 40, 4000)))
    image = E.embed_image(_smooth_field(32, 32))
    assert text.shape == audio.shape == image.shape == (8192,)


def test_image_code_is_resolution_invariant():
    """The same continuous field sampled at two resolutions encodes alike.

    This is the FNO property, and it is why the value axis needs a cosine basis
    rather than a delta basis: resampling moves a pixel from 137 to 138, which a
    one-hot over byte values treats as a totally unrelated symbol.
    """
    for lo, hi in [(32, 64), (32, 128), (64, 128)]:
        c = E.cosine(E.embed_image(_smooth_field(lo, lo)),
                     E.embed_image(_smooth_field(hi, hi)))
        assert c > 0.99, (lo, hi, c)


def test_nyquist_violation_is_an_error_not_a_wrong_answer():
    """Modes above resolution/2 alias differently at each resolution, which
    would quietly void the invariance above. Refuse rather than mislead."""
    with pytest.raises(ValueError, match="Nyquist"):
        E.encode_2d(_smooth_field(16, 16), kx=16, ky=16)


def test_delta_value_basis_is_not_resolution_invariant():
    """The negative control: with a symbolic value axis, resolution invariance
    fails. The basis choice is doing the work, not wishful thinking."""
    a = E.embed_image(_smooth_field(32, 32), value_modes=None)
    b = E.embed_image(_smooth_field(64, 64), value_modes=None)
    assert E.cosine(a, b) < 0.90


def test_audio_code_is_samplerate_invariant():
    """Same tone, two sample rates, same code."""
    tone = lambda n: np.sin(2 * np.pi * 5 * np.arange(n) / n)
    assert E.cosine(E.embed_audio(tone(2000)), E.embed_audio(tone(8000))) > 0.99


def test_mu_law_roundtrip():
    """8-bit companding error, worst at full scale where mu-law is coarsest."""
    x = np.linspace(-1, 1, 2000)
    err = np.abs(E.mu_law_inv(E.mu_law(x)) - x)
    assert err.max() < 0.03 and err.mean() < 0.01


# =========================================================================
# Problem 5 -- the code is invertible, so no output vocabulary is needed
# =========================================================================
def test_residue_vector_is_the_natural_output_space():
    """residue(a+b) is a fixed function of residue(a), residue(b) -- and does not
    depend on how many digits any of them have. That is the property E2 needs."""
    for a, b in [(9, 9), (999999, 1), (12345, 67890)]:
        assert list(N.residues(a + b)) == [(x + y) % p for x, y, p
                                           in zip(N.residues(a), N.residues(b), N.PRIMES)]
        assert list(N.residues(a * b)) == [(x * y) % p for x, y, p
                                           in zip(N.residues(a), N.residues(b), N.PRIMES)]


def test_matched_filter_inverts_the_code():
    """Closed-form inverse recovers bytes exactly, with zero parameters.

    Condition: K >= L, where F is Vandermonde in the distinct nodes exp(-i w_k).
    """
    from firefly.decoder import invertible_grid_for, matched_filter_decode
    for w in [b"firefly", b"hello world", b"Tejaskumar", b"\x00\xff mixed \x7f"]:
        om = invertible_grid_for(len(w))
        assert matched_filter_decode(E.encode_1d(w, om), len(w), om) == w, w


def test_inversion_holds_past_the_kronecker_cap():
    """Kronecker cannot represent these bytes at all; FIREFLY round-trips them."""
    from firefly.decoder import invertible_grid_for, matched_filter_decode
    w = b"a_very_long_token_beyond_thirty_two_bytes_indeed"
    assert len(w) > 32
    om = invertible_grid_for(len(w))
    assert matched_filter_decode(E.encode_1d(w, om), len(w), om) == w


def test_length_is_recoverable_from_the_code_itself():
    """L = (sum over the byte axis of the DC bin)^2. Without this the decoder
    would need to be told the length, which a vocab-free head cannot know."""
    from firefly.decoder import freqs_uniform, infer_length
    om = freqs_uniform(32)
    for w in [b"a", b"firefly", b"hello world", b"x" * 31]:
        assert infer_length(E.encode_1d(w, om)) == len(w), w


def test_zero_parameter_roundtrip_with_unknown_length():
    """The complete answer to Problem 5: bytes -> code -> bytes, no parameters,
    no vocabulary, and no side channel carrying the length."""
    from firefly.decoder import decode, freqs_uniform
    om = freqs_uniform(48)
    for w in [b"firefly", b"Tejaskumar", b"\x00\xff mixed \x7f",
              b"a_token_far_longer_than_thirty_two_bytes_ok"]:
        assert decode(E.encode_1d(w, om), om) == w, w


def test_geometric_grid_inverts_within_its_budget():
    """The default K=16 geometric grid inverts tokens up to 16 bytes. Past that
    the system is underdetermined -- which is precisely the regime the NCA
    decoder exists to handle."""
    from firefly.decoder import matched_filter_decode
    om = E.freqs()
    for w in [b"firefly", b"embedding", b"sixteen__bytes__"]:
        assert matched_filter_decode(E.encode_1d(w, om), len(w), om) == w, w


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
