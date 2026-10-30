"""
Step 2: Preprocessing = (1) clean_text  +  (2) split_clauses.

Design decisions (good viva talking points):
  * We do NOT lowercase, strip stop-words or remove punctuation by default.
    Sentiment tools rely on exactly those signals: "NOT good", "good!!!", and
    especially negators ("not", "never") which are stop-words in most lists.
  * We split into CLAUSES, not just sentences, because one sentence often
    contains several aspects with opposite sentiment:
       "original but delivery took 12 days and support never replied"
         -> ["original", "delivery took 12 days", "support never replied"]
"""
import html
import re
from typing import List

# ---------------------------------------------------------------- cleaning --
_URL = re.compile(r"(https?://\S+|www\.\S+)", re.I)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_EMOJI = re.compile("[\U0001F000-\U0001FFFF\u2600-\u27BF\uFE0F\u200d]")
_REPEAT_LETTERS = re.compile(r"([A-Za-z])\1{2,}")      # soooo -> soo
_WS = re.compile(r"\s+")

# A few emojis carry real sentiment, so convert them to words before removal.
EMOJI_TO_WORD = {
    "😡": " angry ", "😠": " angry ", "🤬": " angry ", "👎": " bad ",
    "👍": " good ", "😍": " love ", "❤": " love ", "😊": " happy ",
    "😞": " sad ", "😢": " sad ", "😭": " sad ", "👌": " good ",
    "🙄": " eyeroll ", "😒": " eyeroll ", "🙃": " eyeroll ",   # sarcasm markers (Step 5)
}

# Contractions -> expanded form, so negation is always an explicit "not".
CONTRACTIONS = [
    (r"\bwon'?t\b", "will not"), (r"\bcan'?t\b", "cannot"),
    (r"\bshan'?t\b", "shall not"),
    (r"\b(do|does|did|is|are|was|were|has|have|had|would|could|should)n'?t\b",
     r"\1 not"),
    (r"n't\b", " not"), (r"'re\b", " are"), (r"'m\b", " am"),
    (r"'ve\b", " have"), (r"'ll\b", " will"),
]

# Tiny slang dictionary typical of review text (extend as you see more data).
SLANG = {"pls": "please", "plz": "please", "u": "you", "gud": "good",
         "bcoz": "because", "bcz": "because", "ur": "your", "thx": "thanks"}


def clean_text(text: str, lowercase: bool = False) -> str:
    """Normalise a raw comment. Safe to call on any string."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)                          # &amp; -> &
    text = text.replace("\u2019", "'").replace("\u2018", "'")  # curly quotes
    text = _URL.sub(" ", text)
    text = _EMAIL.sub(" ", text)
    for emoji, word in EMOJI_TO_WORD.items():
        text = text.replace(emoji, word)
    text = _EMOJI.sub(" ", text)                        # drop remaining emoji
    text = _REPEAT_LETTERS.sub(r"\1\1", text)           # "soooo" -> "soo"
    text = re.sub(r"\.{2,}", ".", text)                 # "..." -> "."
    text = re.sub(r"!{2,}", "!!", text)
    text = re.sub(r"\?{2,}", "??", text)
    for pattern, repl in CONTRACTIONS:
        text = re.sub(pattern, repl, text, flags=re.I)
    text = re.sub(r"\b\w+\b",
                  lambda m: SLANG.get(m.group(0).lower(), m.group(0)), text)
    text = _WS.sub(" ", text).strip()
    return text.lower() if lowercase else text


# --------------------------------------------------------- clause splitting --
# Level 1: sentence boundaries.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+")

# Level 2: contrast words. "X but Y" almost always means X and Y have
# DIFFERENT sentiment, so always split here (even 1-word clauses are fine).
_CONTRAST = re.compile(
    r"(\s*,?\s*\b(?:but|however|although|though|yet|whereas|while|except)\b\s*)",
    re.I)

# Level 3: commas and "and". Riskier ("fast and cheap" must stay together),
# so we only split when BOTH sides have at least MIN_WORDS words.
_ADDITIVE = re.compile(r"(\s*,\s*(?:and\s+)?|\s+and\s+)", re.I)
MIN_WORDS_ADDITIVE = 2


def _word_count(s: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", s))


def _split_keep_seps(pattern: re.Pattern, text: str):
    """re.split with a capture group -> [piece, sep, piece, sep, piece ...]."""
    return pattern.split(text)


# An interjection on its own ("Oh great, another late delivery") must stay glued
# to what follows, otherwise the sarcasm is cut in half (Step 5).
_INTERJECTION = re.compile(
    r"^(?:oh\s+|well\s+)?(?:great|wow|nice|lovely|fantastic|wonderful|amazing|perfect|"
    r"brilliant|awesome|excellent|bravo|thanks|thank you|good job|well done|yeah right|sure)$",
    re.I)


def _split_additive(sentence: str) -> List[str]:
    """Split on ',' / 'and' but re-join pieces that are too short to stand alone."""
    parts = _split_keep_seps(_ADDITIVE, sentence)
    pieces, seps = parts[0::2], parts[1::2]
    merged = [pieces[0]]
    for sep, piece in zip(seps, pieces[1:]):
        # Too short on either side? Glue back together with the original separator.
        if (_word_count(merged[-1]) < MIN_WORDS_ADDITIVE or _word_count(piece) < MIN_WORDS_ADDITIVE
                or _INTERJECTION.match(merged[-1].strip())):
            merged[-1] = merged[-1] + sep + piece
        else:
            merged.append(piece)
    return merged


def split_clauses(text: str) -> List[str]:
    """Split a (cleaned) comment into clauses. Returns non-empty strings."""
    clauses: List[str] = []
    for sentence in _SENTENCE_SPLIT.split(text):
        if not sentence or not sentence.strip():
            continue
        # Split on contrast words (drop the conjunction itself).
        chunks = _CONTRAST.split(sentence)[0::2]      # [0::2] = pieces, no seps
        for chunk in chunks:
            for clause in _split_additive(chunk):
                clause = clause.strip(" ,.;!?-")
                if clause:
                    clauses.append(clause)
    return clauses


def preprocess(raw_comment: str) -> List[str]:
    """Full pipeline: raw comment -> list of clean clauses."""
    return split_clauses(clean_text(raw_comment))


if __name__ == "__main__":
    samples = [
        "Product was original but delivery took 12 days and support never replied.",
        "Sooooo slow!!! Didn't even get a refund... check www.spam.com 😡",
        "Fast and cheap delivery, great price",
        "Great product and fast delivery",
        "It's not bad, but the price is too high.",
    ]
    for s in samples:
        print("RAW    :", s)
        print("CLEAN  :", clean_text(s))
        print("CLAUSES:", preprocess(s), "\n")
