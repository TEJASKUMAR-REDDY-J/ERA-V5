"""Shared helpers: language config, NFC normalization, consistent word split."""
import re
import unicodedata

# One place defining the 4 languages, their wiki URLs, and file keys.
LANGUAGES = [
    {"key": "en", "name": "English", "wiki_url": "https://en.wikipedia.org/wiki/India"},
    {"key": "hi", "name": "Hindi",   "wiki_url": "https://hi.wikipedia.org/wiki/भारत"},
    {"key": "te", "name": "Telugu",  "wiki_url": "https://te.wikipedia.org/wiki/భారతదేశం"},
    {"key": "mr", "name": "Marathi", "wiki_url": "https://mr.wikipedia.org/wiki/भारत"},
]

VOCAB_PER_LANG = 2500          # target per monolingual sub-tokenizer
TOTAL_VOCAB = 10000            # final merged vocab size


def normalize(text: str) -> str:
    """NFC normalize; collapse whitespace. Identical across all languages.
    No case-folding — would corrupt non-cased Indic scripts and isn't wanted for English either."""
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip()


# Word = run of non-whitespace after stripping edge punctuation. Same rule for every
# language so counts are comparable. Indic scripts have no spaces inside words, so
# whitespace splitting is a fair, consistent proxy across all four.
_PUNCT_EDGE = re.compile(r"^[^\wऀ-ॿఀ-౿]+|[^\wऀ-ॿఀ-౿]+$")


# Equal-sized corpora per language: a smaller, comparable word budget lets every
# language (English especially) reach low fertility inside the shared 10k vocab, and
# is the fairer basis for cross-lingual comparison. Matches the assignment's "say 5000
# words" framing. Train and evaluate BOTH read through capped_text so counts stay consistent.
CORPUS_WORD_CAP = 3000


def capped_text(text: str) -> str:
    return " ".join(normalize(text).split()[:CORPUS_WORD_CAP])


def word_count(text: str) -> int:
    return len(words(text))


def words(text: str) -> list[str]:
    out = []
    for tok in normalize(text).split():
        tok = _PUNCT_EDGE.sub("", tok)
        if tok:
            out.append(tok)
    return out
