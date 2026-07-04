"""STEP 1 — Fetch the India Wikipedia article in 4 languages, clean to plain text.

Raw HTML -> data/raw/{key}.html
Clean text -> data/clean/{key}.txt
Prints word count per language (consistent whitespace+punctuation split).
"""
import pathlib
import requests
from bs4 import BeautifulSoup

from utils import LANGUAGES, word_count, normalize

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
HEADERS = {"User-Agent": "tokenizer-assignment/1.0 (educational; contact via course)"}


def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    # Some articles (English) render several mw-parser-output divs; pick the one
    # holding the actual prose (most <p> tags), else fall back to body.
    outputs = soup.select("div.mw-parser-output")
    body = max(outputs, key=lambda d: len(d.find_all("p")), default=None) or soup.body or soup

    # Drop non-article furniture: infoboxes, tables, nav, references, edit links, etc.
    kill = (
        "table, .infobox, .navbox, .vertical-navbox, .reflist, .reference, "
        "sup.reference, .mw-editsection, .thumb, figure, style, script, "
        ".hatnote, .metadata, .mw-empty-elt, .noprint, #toc, .toc, .gallery, "
        ".sistersitebox, .side-box, .ambox, .portal, .mw-references-wrap"
    )
    for el in body.select(kill):
        el.decompose()

    # Keep only real prose blocks.
    parts = []
    for el in body.find_all(["p", "h2", "h3", "li"]):
        txt = el.get_text(" ", strip=True)
        if txt:
            parts.append(txt)
    text = "\n".join(parts)

    # Strip leftover bracket citation markers like [1], [a], [note 2].
    import re
    text = re.sub(r"\[[^\]\n]{1,15}\]", "", text)
    return normalize(text)


def main():
    for lang in LANGUAGES:
        key = lang["key"]
        r = requests.get(lang["wiki_url"], headers=HEADERS, timeout=30)
        r.raise_for_status()
        (RAW / f"{key}.html").write_text(r.text, encoding="utf-8")

        text = clean_html(r.text)
        (CLEAN / f"{key}.txt").write_text(text, encoding="utf-8")
        print(f"{lang['name']:8s} ({key}): {word_count(text):>7,} words, {len(text):>8,} chars")


if __name__ == "__main__":
    main()
