"""FIREFLY spectral encoder.

One operator, three modalities.

A token, an audio patch and an image patch are all the same object: a *byte
field* over some index domain. Text and audio are fields over a 1-D domain,
images over a 2-D domain. The embedding is that field's Fourier transform
evaluated on a FIXED frequency grid:

    Phi[c, k] = (1/sqrt(L)) * sum_{p : b_p = c} exp(-i * w_k * p)

Because the grid is fixed and chosen in advance, the code's size depends only
on the grid -- never on the field's length, resolution or dimensionality. That
single property is what buys unbounded token length, resolution invariance and
a projection matrix shared across modalities.

Contrast with the Kronecker codec this replaces:

    kappa[c, p] = (1/sqrt(L)) * 1[b_p = c]          256 x 32 = 8192

Kronecker uses a *delta* basis on the position axis. FIREFLY uses a *Fourier*
basis on the same axis. With K = 16 frequencies the output dimension is
256 * 16 * 2 = 8192 -- identical, so this is a drop-in swap that needs no
change to the downstream projection.
"""
from __future__ import annotations

import numpy as np

N_CHANNELS = 256          # byte alphabet, lossless
K_DEFAULT = 16            # frequencies -> 256 * 16 * 2 = 8192
D_MODEL_CODE = N_CHANNELS * K_DEFAULT * 2

# Kronecker paper's settings, kept for the baseline arm.
KRON_POSITIONS = 32


# --------------------------------------------------------------------------
# frequency grids
# --------------------------------------------------------------------------
def freqs(k: int = K_DEFAULT, p_min: float = 2.0, p_max: float = 64.0) -> np.ndarray:
    """Geometric ladder of angular frequencies, periods p_max -> p_min bytes.

    Geometric rather than linear because morphology is multi-scale: a
    one-byte inflection and an eight-byte stem need different resolutions.
    p_min = 2 puts the top of the ladder exactly at Nyquist (w = pi).
    """
    if k == 1:
        return np.array([2 * np.pi / p_max])
    periods = p_max * (p_min / p_max) ** (np.arange(k) / (k - 1))
    return 2 * np.pi / periods


def image_freqs(kx: int = 4, ky: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Integer cycles-per-patch on normalised [0,1) coordinates.

    Normalised (not absolute) coordinates are what make the code resolution
    invariant: a 16x16 and a 32x32 crop of the same picture land on the same
    frequencies, so they produce comparable codes of identical size.
    """
    return 2 * np.pi * np.arange(kx), 2 * np.pi * np.arange(ky)


def freqs_linear(k: int) -> np.ndarray:
    """Integer cycles-per-window on normalised [0,1) time. Used for audio, where
    the meaningful frequencies are harmonics of the window, not byte-scale."""
    return 2 * np.pi * np.arange(k)


# --------------------------------------------------------------------------
# the VALUE axis
# --------------------------------------------------------------------------
# The index axis is always Fourier. The basis on the *value* axis is a property
# of the data, not of the method:
#
#   symbolic values (text bytes)  -> delta basis. 'a' is not a near-miss of 'b';
#                                    identity is all that matters, metric is noise.
#   magnitude values (pixels,     -> truncated cosine basis. 137 really is next to
#   mu-law audio samples)            138, and a delta basis throws that away.
#
# These are the same construction. A delta basis is a complete (untruncated)
# transform of the value alphabet; truncating it to J modes is what *creates* a
# metric on values. Exactly FNO's mode truncation, applied to the value axis
# instead of the spatial one.
def value_basis(j_modes: int, n_vals: int = N_CHANNELS) -> np.ndarray:
    """(n_vals, j_modes) real DCT-style basis, low modes first.

    Half-period (pi, not 2*pi) so the basis is NOT circular: byte 0 and byte 255
    are the extremes of a scale, not neighbours on a ring.
    """
    c = np.arange(n_vals)[:, None]
    j = np.arange(j_modes)[None, :]
    b = np.cos(np.pi * j * (c + 0.5) / n_vals)
    return b / np.sqrt((b ** 2).sum(axis=0, keepdims=True))


# --------------------------------------------------------------------------
# 1-D: text and audio
# --------------------------------------------------------------------------
def as_bytes(x) -> np.ndarray:
    if isinstance(x, str):
        return np.frombuffer(x.encode("utf-8"), dtype=np.uint8)
    if isinstance(x, (bytes, bytearray, memoryview)):
        return np.frombuffer(bytes(x), dtype=np.uint8)
    return np.asarray(x, dtype=np.uint8)


def encode_1d(x, omega: np.ndarray | None = None, normalized: bool = False,
              n_ch: int = N_CHANNELS, value_modes: int | None = None) -> np.ndarray:
    """Spectral code of a 1-D byte field. Returns complex (n_ch or J, K).

    normalized=False  absolute byte offsets. Correct for TEXT: morphology
                      cares where a byte sits, and 'un-' at offset 0 must not
                      look like '-un' at offset 5.
    normalized=True   positions rescaled to [0,1). Correct for AUDIO patches,
                      where the sample rate is an accident of recording and
                      two clips of the same sound at different rates should
                      encode alike.

    value_modes=None  delta basis on values -- symbolic. The text setting.
    value_modes=J     J cosine modes on values -- magnitudes. The audio setting.

    No length cap. Nothing is truncated on the index axis. That is the point.
    """
    omega = freqs() if omega is None else omega
    b = as_bytes(x)
    lo = b.size
    j = n_ch if value_modes is None else value_modes
    if lo == 0:
        return np.zeros((j, omega.size), dtype=np.complex128)
    pos = (np.arange(lo) / lo) if normalized else np.arange(lo, dtype=np.float64)
    phase = np.exp(-1j * np.outer(pos, omega))            # (L, K)
    if value_modes is None:
        out = np.zeros((n_ch, omega.size), dtype=np.complex128)
        np.add.at(out, b, phase)                          # scatter into byte channel
    else:
        out = value_basis(value_modes, n_ch)[b].T @ phase  # (J, L) @ (L, K)
    return out / np.sqrt(lo)


def encode_2d(patch: np.ndarray, kx: int = 16, ky: int = 16,
              n_ch: int = N_CHANNELS, value_modes: int | None = 16) -> np.ndarray:
    """Spectral code of a 2-D byte field (image patch). Returns complex (J, ky, kx).

    Default budget J=16 value modes x 16 x 16 spatial modes -> 16*16*16*2 = 8192,
    the same width as the 1-D text code. The budget is constant across
    modalities; only how it is *split* between the value axis and the index axes
    changes, according to where each modality carries its information.
    """
    patch = np.asarray(patch, dtype=np.uint8)
    h, w = patch.shape
    # Nyquist. Modes above resolution/2 alias, and alias DIFFERENTLY at each
    # resolution -- which silently destroys the resolution invariance that is
    # the entire reason for working in mode space. Fail loudly instead.
    if kx > w // 2 or ky > h // 2:
        raise ValueError(
            f"{ky}x{kx} modes exceed Nyquist for a {h}x{w} patch; "
            f"need patch >= {2 * ky}x{2 * kx}")
    wx, wy = image_freqs(kx, ky)
    py = np.exp(-1j * np.outer(np.arange(h) / h, wy))      # (H, ky)
    px = np.exp(-1j * np.outer(np.arange(w) / w, wx))      # (W, kx)
    basis = (py[:, None, :, None] * px[None, :, None, :]).reshape(h * w, ky, kx)
    flat = patch.ravel()
    if value_modes is None:
        out = np.zeros((n_ch, ky, kx), dtype=np.complex128)
        np.add.at(out, flat, basis)
    else:
        v = value_basis(value_modes, n_ch)[flat]           # (H*W, J)
        out = np.einsum("nj,nyx->jyx", v, basis)
    return out / np.sqrt(h * w)


# --------------------------------------------------------------------------
# audio -> bytes
# --------------------------------------------------------------------------
def mu_law(x: np.ndarray, mu: int = 255) -> np.ndarray:
    """Companding to 8 bits. Audio then *is* a byte string, so the identical
    1-D text encoder applies with no new code at all."""
    x = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    return np.clip(np.round((y + 1) / 2 * 255), 0, 255).astype(np.uint8)


def mu_law_inv(b: np.ndarray, mu: int = 255) -> np.ndarray:
    y = np.asarray(b, dtype=np.float64) / 255 * 2 - 1
    return np.sign(y) * ((1 + mu) ** np.abs(y) - 1) / mu


# --------------------------------------------------------------------------
# baseline: the Kronecker codec being replaced
# --------------------------------------------------------------------------
def kronecker(x, d_p: int = KRON_POSITIONS, n_ch: int = N_CHANNELS) -> np.ndarray:
    """Reference implementation of Eq.1 from the Kronecker Embeddings paper.

    Note the truncation at d_p: bytes past position 32 are DISCARDED. That is
    information destroyed, not re-parameterised, and it is the one gap a
    learned projection cannot close.
    """
    b = as_bytes(x)
    lo = b.size
    out = np.zeros((n_ch, d_p), dtype=np.float64)
    if lo == 0:
        return out
    t = b[:d_p]
    out[t, np.arange(t.size)] = 1.0
    return out / np.sqrt(lo)


# --------------------------------------------------------------------------
# flattening / normalisation
# --------------------------------------------------------------------------
def to_real(code: np.ndarray) -> np.ndarray:
    """Complex code -> flat real vector [Re | Im]."""
    if np.iscomplexobj(code):
        return np.concatenate([code.real.ravel(), code.imag.ravel()])
    return code.ravel()


def znorm(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-token z-normalisation, as the Kronecker paper applies to its codec.
    Used on both arms so the comparison is like-for-like."""
    return (v - v.mean()) / (v.std() + eps)


# Per-modality mode budgets. All three multiply out to 8192 real dimensions;
# they differ only in how the budget is split between value and index axes.
AUDIO_VALUE_MODES, AUDIO_TIME_MODES = 16, 256      # 16 * 256 * 2 = 8192
IMAGE_VALUE_MODES, IMAGE_SPATIAL = 16, 16          # 16 *  16 * 16 * 2 = 8192
IMAGE_MIN_SIDE = 2 * IMAGE_SPATIAL                 # Nyquist: patches >= 32x32
# Both budgets are 8192; only the split differs. Measured, not guessed --
# experiments/e4_multimodal.py sweeps the split and reports resolution
# invariance for each. Sixteen value modes is the knee: fewer loses intensity
# detail, more starts re-creating the delta basis and invariance collapses
# (J=256 scores 0.70 where J=16 scores 0.995).


def embed(x, omega: np.ndarray | None = None, normalized: bool = False,
          normalize_out: bool = True) -> np.ndarray:
    """Text bytes -> flat real 8192-d code. Delta basis on values: symbolic."""
    v = to_real(encode_1d(x, omega=omega, normalized=normalized))
    return znorm(v) if normalize_out else v


def embed_audio(samples: np.ndarray, normalize_out: bool = True) -> np.ndarray:
    """Float audio in [-1,1] -> flat real 8192-d code.

    mu-law companding turns audio into a byte string, so the identical 1-D
    encoder applies. Values use a cosine basis (amplitudes have a metric) and
    time is normalised (sample rate is an accident of recording).
    """
    b = mu_law(samples)
    code = encode_1d(b, omega=freqs_linear(AUDIO_TIME_MODES), normalized=True,
                     value_modes=AUDIO_VALUE_MODES)
    v = to_real(code)
    return znorm(v) if normalize_out else v


def embed_image(patch: np.ndarray, kx: int = IMAGE_SPATIAL, ky: int = IMAGE_SPATIAL,
                value_modes: int = IMAGE_VALUE_MODES,
                normalize_out: bool = True) -> np.ndarray:
    """Image patch -> flat real 8192-d code, same width as embed()."""
    v = to_real(encode_2d(patch, kx=kx, ky=ky, value_modes=value_modes))
    return znorm(v) if normalize_out else v


def embed_kronecker(x, normalize_out: bool = True) -> np.ndarray:
    v = to_real(kronecker(x))
    return znorm(v) if normalize_out else v


# --------------------------------------------------------------------------
# phase-covariant and phase-invariant halves
# --------------------------------------------------------------------------
# The complex code splits cleanly into two parts with opposite behaviour under a
# shift of the index origin:
#
#   [Re, Im]   shift-COVARIANT   -- rotates by exp(-i.w.delta). Carries WHERE.
#   |Phi|      shift-INVARIANT   -- does not move at all. Carries WHAT.
#
# Which one a task wants is a property of the task, and measured in E4:
#   MNIST digit identity depends on where the strokes are      -> phase wins
#   language ID depends on which bytes occur, not where        -> magnitude wins
#   a spoken digit's offset in the recording is an artefact    -> magnitude wins
#   decoding bytes back out needs the positions                -> phase required
#
# A delta basis offers no such decomposition: it has no shift-invariant part to
# extract. Note also that |Phi| is nonlinear, so a linear projection cannot
# manufacture it from [Re, Im] -- it has to be supplied.
def with_magnitude(code: np.ndarray, normalize_out: bool = True) -> np.ndarray:
    """[Re | Im | |Phi|] -- both halves, so the model can use either."""
    ri = to_real(code)
    mag = np.abs(code).ravel()
    if normalize_out:
        ri, mag = znorm(ri), znorm(mag)
    return np.concatenate([ri, mag])


def embed_full(x, omega: np.ndarray | None = None, normalized: bool = False):
    """Text/audio bytes -> 12288-d code carrying phase and magnitude."""
    return with_magnitude(encode_1d(x, omega=omega, normalized=normalized))


def embed_audio_full(samples: np.ndarray) -> np.ndarray:
    return with_magnitude(encode_1d(mu_law(samples),
                                    omega=freqs_linear(AUDIO_TIME_MODES),
                                    normalized=True, value_modes=AUDIO_VALUE_MODES))


def embed_image_full(patch: np.ndarray) -> np.ndarray:
    return with_magnitude(encode_2d(patch, kx=IMAGE_SPATIAL, ky=IMAGE_SPATIAL,
                                    value_modes=IMAGE_VALUE_MODES))


# --------------------------------------------------------------------------
# structural properties (used by the test suite and the probes)
# --------------------------------------------------------------------------
def shift_operator(delta: int, omega: np.ndarray | None = None) -> np.ndarray:
    """Prefixing a field by `delta` bytes multiplies its spectrum by this.

    Shifting the index domain is a PHASE ROTATION, not a re-indexing. That is
    the property the delta basis does not have: under Kronecker, inserting one
    byte moves every subsequent spike to a different coordinate and the cosine
    collapses toward zero.
    """
    omega = freqs() if omega is None else omega
    return np.exp(-1j * omega * delta)


def fourier_from_kronecker_matrix(omega: np.ndarray | None = None,
                                  d_p: int = KRON_POSITIONS) -> np.ndarray:
    """The (d_p, K) matrix F with Phi = kappa @ F for fields of length <= d_p.

    We publish this deliberately. For short tokens the two codecs span the same
    information and a learned projection can absorb F, so on short-token text
    they are equivalent up to optimisation geometry. Every genuine FIREFLY gain
    lives where this matrix does NOT apply: length > d_p, non-integer or 2-D
    index domains, magnitude features, and the appended numeric block.
    """
    omega = freqs() if omega is None else omega
    return np.exp(-1j * np.outer(np.arange(d_p), omega))


def magnitude_code(x, omega: np.ndarray | None = None) -> np.ndarray:
    """|Phi| -- shift invariant, and NOT computable from kappa by any linear map."""
    return np.abs(encode_1d(x, omega=omega)).ravel()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(a @ b / (na * nb))
