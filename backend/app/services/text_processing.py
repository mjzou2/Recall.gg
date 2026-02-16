import json
import re
from pathlib import Path
from typing import Dict, List

from rapidfuzz import fuzz, process

from app.config import (
    FUZZY_CATEGORIES,
    FUZZY_STOPWORDS,
    MIN_FUZZY_WORD_LEN,
    FUZZY_SCORE_CUTOFF,
)


def normalize_lol_text(text: str) -> str:
    replacements = [
        # Common Whisper mishears
        (r"\bharold\b", "herald"),
        (r"\bword(s|ing|ed)?\b", r"ward\1"),
        # Severe mishears fuzzy matching can't catch (score < 82)
        (r"\bcass?ante\b", "K'Sante"),
        (r"\baura\b", "Aurora"),
        (r"\brookong\b", "Wukong"),
        (r"\bni[kc]os?\b", "Neeko"),  # matches: niko, nico, nikos, nicos
        (r"\bflush\b", "flash"),
        (r"\bani\b", "Annie"),
        # Space-separated apostrophe champions (parts too short for fuzzy)
        (r"\bk sante\b", "K'Sante"),
        (r"\bksante\b", "K'Sante"),
        (r"\bkha zix\b", "Kha'Zix"),
        (r"\bcho gath\b", "Cho'Gath"),
        (r"\bbel veth\b", "Bel'Veth"),
        (r"\bvel koz\b", "Vel'Koz"),
        (r"\bkog maw\b", "Kog'Maw"),
        (r"\brek sai\b", "Rek'Sai"),
        (r"\bkai sa\b", "Kai'Sa"),
        # Short word mishears (< 4 chars, fuzzy skips)
        (r"\bchen\b", "Shen"),
        (r"\bjacks?\b", "Jax"),  # matches: jax, jack, jacks, jack's (apostrophe stripped by re.IGNORECASE)
        # Short abbreviations (< 4 chars, fuzzy skips)
        (r"\b[dt]p\b", "TP"),  # matches: tp, dp
        (r"\bcc\b", "CC"),
        (r"\bcs\b", "CS"),
        (r"\badc\b", "ADC"),
    ]
    normalized = text
    for pattern, replacement in replacements:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    return normalized

def load_term_bank() -> Dict[str, List[str]]:
    """Load the League of Legends term bank from JSON."""
    term_bank_path = Path(__file__).resolve().parent.parent.parent / "term_bank.json"
    try:
        with open(term_bank_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Term bank not found at {term_bank_path}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse term bank: {e}")
        return {}


def _build_fuzzy_lookup(term_bank: Dict[str, List[str]]) -> Dict[int, Dict[str, str]]:
    """Build lookup tables for fuzzy matching, keyed by word count.

    Only includes champions, items, and objectives — common_comms, locations,
    and mechanics are skipped because they contain common English words that
    cause false positives (e.g. "taking" → "tracking").

    Returns dict mapping word_count -> {normalized_form: original_term}.
    e.g. {1: {"ksante": "K'Sante"}, 2: {"lee sin": "Lee Sin"}}
    """
    lookup: Dict[int, Dict[str, str]] = {}
    seen_normalized: set = set()

    for category, terms in term_bank.items():
        if category not in FUZZY_CATEGORIES:
            continue
        for term in terms:
            normalized = term.lower().replace("'", "")
            if normalized in seen_normalized:
                continue
            seen_normalized.add(normalized)
            word_count = len(normalized.split())
            if word_count not in lookup:
                lookup[word_count] = {}
            lookup[word_count][normalized] = term

    return lookup


def fuzzy_correct_text(text: str, fuzzy_lookup: Dict[int, Dict[str, str]]) -> str:
    """Fuzzy-match words against the term bank and replace confident matches.

    Two-pass approach:
      Pass 1: Multi-word n-grams (longest first)
      Pass 2: Single words (length >= MIN_FUZZY_WORD_LEN)
    """
    if not fuzzy_lookup:
        return text

    words = text.split()
    if not words:
        return text

    matched = [False] * len(words)
    output = list(words)

    # --- Pass 1: Multi-word matching (longest n-grams first) ---
    multi_sizes = sorted([n for n in fuzzy_lookup if n > 1], reverse=True)

    for n in multi_sizes:
        terms_for_n = fuzzy_lookup[n]
        if not terms_for_n:
            continue
        choices = list(terms_for_n.keys())

        i = 0
        while i <= len(words) - n:
            if any(matched[i:i + n]):
                i += 1
                continue

            raw = " ".join(words[i:i + n])
            window = raw.strip(".,!?;:\"'()[]").lower().replace("'", "")
            result = process.extractOne(
                window,
                choices,
                scorer=fuzz.ratio,
                score_cutoff=FUZZY_SCORE_CUTOFF,
            )

            if result is not None:
                best_match, score, _index = result
                original_term = terms_for_n[best_match]
                replacement_words = original_term.split()
                # Preserve trailing punctuation from the last word in the window
                last_word = words[i + n - 1]
                trailing_punct = last_word[len(last_word.rstrip(".,!?;:\"'()[]")):]
                for j in range(n):
                    if j < len(replacement_words):
                        output[i + j] = replacement_words[j]
                    else:
                        output[i + j] = ""
                    matched[i + j] = True
                if trailing_punct and replacement_words:
                    last_idx = i + min(n, len(replacement_words)) - 1
                    output[last_idx] += trailing_punct
                i += n
            else:
                i += 1

    # --- Pass 2: Single-word matching ---
    single_terms = fuzzy_lookup.get(1, {})
    if single_terms:
        choices_single = list(single_terms.keys())

        for i, word in enumerate(words):
            if matched[i]:
                continue

            stripped = word.strip(".,!?;:\"'()[]")
            if len(stripped) < MIN_FUZZY_WORD_LEN:
                continue

            normalized = stripped.lower().replace("'", "")
            if normalized in FUZZY_STOPWORDS:
                continue
            result = process.extractOne(
                normalized,
                choices_single,
                scorer=fuzz.ratio,
                score_cutoff=FUZZY_SCORE_CUTOFF,
            )

            if result is not None:
                best_match, score, _index = result
                original_term = single_terms[best_match]
                prefix = word[:len(word) - len(word.lstrip(".,!?;:\"'()[]"))]
                suffix = word[len(word.rstrip(".,!?;:\"'()[]")):]
                output[i] = prefix + original_term + suffix
                matched[i] = True

    return " ".join(w for w in output if w)
