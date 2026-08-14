"""FIREFLY -- Fourier Interference Representations for Embedding
Fields of any Length or dimensionalitY.

A frozen, parameter-free embedding operator. Text, audio and images are all byte
fields over an index domain; the embedding is the field's spectrum on a fixed
frequency grid, so the code width depends on the grid alone -- never on length,
resolution or dimensionality.

Named for the firefly for two reasons that are actually the method:

  * a firefly's SPECIES IS ITS FLASH FREQUENCY. Entomologists identify them by
    flash rate and inter-pulse interval, not by appearance. Identity encoded as
    a temporal frequency signature is exactly what this operator does to a token.

  * fireflies SYNCHRONISE BY PHASE COUPLING. Thousands of independent local
    oscillators sum into global coherence. That is the encoder (each character
    contributes a wave; the word is their interference pattern) and it is the
    numeric block (phases add elementwise, and the sum is the arithmetic).

The coprime moduli in the numeric block are a Chinese-Remainder-Theorem choice,
not an entomological one -- the firefly analogy is to phase, not to primes.
"""
__version__ = "2.0.0"
