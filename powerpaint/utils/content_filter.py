"""NSFW/18+ content filtering for prompt text and images.

Layered approach:
1. Keyword blacklist (Vietnamese + English) on the *resolved* (post-translation)
   prompt, so both the original Vietnamese text and its English translation
   are checked.
2. A binary image classifier (Falconsai/nsfw_image_detection) on the input
   image and the generated output image.  Anything the model labels "nsfw"
   (including suggestive swimwear) is blocked.

The classifier loads eagerly when the app starts, so the first image check is
instant.  It is reused across input/output checks within one process.
"""

import logging
import re

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

logger = logging.getLogger(__name__)

# Words matched as whole tokens (bounded by non-alphanumeric characters) to
# avoid false positives such as "ass" inside "assignment" or "cụ" inside "cụm".
# English is checked separately from Vietnamese because Vietnamese does not
# use spaces to delimit words.
_ENGLISH_BLOCKED_WORDS = {
    "porn",
    "porno",
    "pornography",
    "pornographic",
    "nude",
    "nudes",
    "nudity",
    "naked",
    "undressed",
    "topless",
    "bare chest",
    "nsfw",
    "hentai",
    "sex",
    "erotic",
    "explicit",
    "fetish",
    "bondage",
    "bdsm",
    "masturbat",
    "penis",
    "vagina",
    "boobs",
    "tits",
    "nipple",
    "nipples",
    "anal",
    "oral sex",
    "blowjob",
    "incest",
    "rape",
    "molest",
    "child porn",
    "underage sex",
    "lolita",
    "teen porn",
    "barely legal",
    "snuff",
    "gore",
    "bestiality",
    "zoophilia",
}

# Vietnamese explicit words.  These commonly appear both with and without
# diacritics, so match a normalised form (accents stripped, lower-cased).
# Only clearly adult terms are listed; ambiguous everyday words ("to", "nàng",
# "chim", "các", "lớn", "buổi", "đầm", "hiệp", "thiếu nữ", "hang động" ...)
# are deliberately omitted so normal prompts are not blocked.
_VIETNAMESE_BLOCKED_WORDS = {
    "dit",
    "djt",
    "duma",
    "dume",
    "dm",
    "dmm",
    "dcm",
    "cmm",
    "cmn",
    "cl",
    "clme",
    "loz",
    "vkl",
    "vai lon",
    "vai loz",
    "khoa than",
    "thoa than",
    "dam duc",
    "hiep dam",
    "an dam",
    "se x",
    "sex",
    "hentai",
    "porn",
    "duong vat",
    "am dao",
    "nguc",
}

# The classifier's labels, in the order returned by the model
# (Falconsai/nsfw_image_detection is a binary normal/nsfw classifier).
_NSFW_LABELS = ("normal", "nsfw")

# The label that means "adult content, block it".  The binary classifier has
# no "sexy" class, so any non-normal image (including suggestive swimwear) is
# treated as adult.  This matches the product decision to block 18+ content.
_ADULT_LABELS = {"nsfw"}

_MODEL_ID = "Falconsai/nsfw_image_detection"

# ---------------------------------------------------------------------------
# Prompt (keyword) filter
# ---------------------------------------------------------------------------


def _normalise_text(text: str) -> str:
    """Lower-case and strip Vietnamese diacritics so accents do not bypass the
    filter ("địt" and "dit" both match)."""
    text = text.lower()
    # Map accented Vietnamese characters to their base ASCII letters.
    for accented, base in {
        "à": "a", "á": "a", "ả": "a", "ã": "a", "ạ": "a", "ă": "a", "ắ": "a",
        "ằ": "a", "ẳ": "a", "ẵ": "a", "ặ": "a", "â": "a", "ấ": "a", "ầ": "a",
        "ẩ": "a", "ẫ": "a", "ậ": "a",
        "è": "e", "é": "e", "ẻ": "e", "ẽ": "e", "ẹ": "e", "ê": "e", "ế": "e",
        "ề": "e", "ể": "e", "ễ": "e", "ệ": "e",
        "ì": "i", "í": "i", "ỉ": "i", "ĩ": "i", "ị": "i",
        "ò": "o", "ó": "o", "ỏ": "o", "õ": "o", "ọ": "o", "ô": "o", "ố": "o",
        "ồ": "o", "ổ": "o", "ỗ": "o", "ộ": "o", "ơ": "o", "ớ": "o", "ờ": "o",
        "ở": "o", "ỡ": "o", "ợ": "o",
        "ù": "u", "ú": "u", "ủ": "u", "ũ": "u", "ụ": "u", "ư": "u", "ứ": "u",
        "ừ": "u", "ử": "u", "ữ": "u", "ự": "u",
        "ỳ": "y", "ý": "y", "ỷ": "y", "ỹ": "y", "ỵ": "y",
        "đ": "d",
    }.items():
        text = text.replace(accented, base)
    return text


_ENGLISH_WORD_RE = re.compile(r"[a-z0-9]+")


def _contains_blocked_english(text: str) -> bool:
    """Match blocked multi-word phrases first, then whole-word tokens."""
    # Multi-word phrases ("oral sex", "child porn", ...) match as full phrases.
    for word in _ENGLISH_BLOCKED_WORDS:
        if " " in word and re.search(rf"(?<![a-z0-9]){re.escape(word)}(?![a-z0-9])", text):
            return True

    # Single-word tokens.  Match the exact word, or a small set of inflections.
    for token in _ENGLISH_WORD_RE.findall(text):
        for blocked in _ENGLISH_BLOCKED_WORDS:
            if " " in blocked:
                continue
            if token == blocked:
                return True
            # "sexy" is allowed, so never treat "sex" as a stem of "sexy".
            if token == "sexy":
                continue
            if token.startswith(blocked) and token[len(blocked) :] in {
                "s",
                "ing",
                "ed",
                "es",
                "ion",
                "ions",
            }:
                return True
    return False


def _contains_blocked_vietnamese(text: str) -> bool:
    """Vietnamese is not space-delimited per word, so match the normalised
    text directly.  Short words like "dm" and "cl" are only matched when
    surrounded by non-letters to avoid common false positives."""
    for word in _VIETNAMESE_BLOCKED_WORDS:
        if len(word) <= 3:
            # Whole-token match only, so "sex" does not match "sexy" and
            # "dm" does not match "dmm" by accident.
            if re.search(rf"(?<![a-z]){re.escape(word)}(?![a-z])", text):
                return True
        elif word in text:
            return True
    return False


def prompt_is_blocked(prompt: str) -> bool:
    """Return True when the (already resolved/translated) prompt contains
    adult keywords in either English or Vietnamese."""
    if not prompt or not prompt.strip():
        return False
    normalised = _normalise_text(prompt)
    return _contains_blocked_english(normalised) or _contains_blocked_vietnamese(normalised)


# ---------------------------------------------------------------------------
# Image (classifier) filter
# ---------------------------------------------------------------------------

class NSFWImageFilter:
    """NSFW image classifier, loaded eagerly at construction.

    Loading happens up front (during app startup) so the first image check is
    instant — the user never waits for the model on first upload.
    """

    def __init__(self, device=None, weight_dtype=torch.float16):
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._weight_dtype = weight_dtype
        self._processor = None
        self._model = None
        self._labels = _NSFW_LABELS
        self._ensure_loaded()

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            self._processor = AutoImageProcessor.from_pretrained(_MODEL_ID)
            self._model = AutoModelForImageClassification.from_pretrained(_MODEL_ID)
            if self._device.type == "cuda":
                self._model = self._model.to(self._device, dtype=self._weight_dtype)
            self._model.eval()
        except Exception:
            # Fail open: an unavailable classifier must not break generation.
            logger.warning("NSFW image classifier unavailable; image filtering is disabled.", exc_info=True)

    def check_image(self, image: Image.Image) -> bool:
        """Return True when the image contains adult (nsfw) content.

        The binary classifier labels everything non-normal as "nsfw", so
        suggestive swimwear is also blocked.  On any classifier error the
        image is allowed through rather than silently dropped.
        """
        try:
            self._ensure_loaded()
            if self._model is None:
                return False
            inputs = self._processor(images=image.convert("RGB"), return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self._model(**inputs).logits
            label_id = logits.argmax(dim=-1).item()
            return self._labels[label_id] in _ADULT_LABELS
        except Exception:
            # Fail open: an unavailable classifier must not break generation.
            return False
