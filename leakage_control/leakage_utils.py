"""Helpers for the leakage-control step.

    load_split()                  rationale file -> entries of one split
    log_odds_z_scores()           informative-Dirichlet log-odds z-scores (Monroe et al., 2008)
    extract_protected_words()     target words / phrase tokens that must never be masked
    rank_biased_words()           words ranked by |z|, protected words removed
    build_mask_regex(), mask_text_with_regex()
                                  delete a set of words and tidy the punctuation left behind
    leakage_probe_accuracy()      TF-IDF + logistic regression on the rationale text alone

Word selection only ever looks at the train split. The chosen word list is then
applied unchanged to every split.
"""

import json
import re
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold


TOKEN_RE = re.compile(r"[a-z]+")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def load_split(path: str, split_name: str, id_prefix: str) -> Dict[str, dict]:
    """
    Load a rationale file and keep the entries of one split, keyed by id.
    id_prefix is 'wic' or 'ps', matching the "<prefix>-<split>-<row>" ids.
    """
    with open(path) as f:
        data = json.load(f)
    pattern = re.compile(rf"^{re.escape(id_prefix)}-{re.escape(split_name)}-\d+$")
    out = {k: v for k, v in data.items() if pattern.match(k) and v.get("rationale")}
    return out


# --- word-bias detection ---

def log_odds_z_scores(rationales_label0: List[str], rationales_label1: List[str]
                       ) -> Dict[str, dict]:
    """
    Monroe, Colaresi & Quinn (2008) informative-Dirichlet-prior log-odds-ratio.
    Returns {word: {"z": z_score, "count0": n, "count1": n, "log_odds": delta}}
    sorted is NOT done here -- caller sorts by abs(z).

    A large positive z means the word is disproportionately common in
    label-1 rationales (i.e. it "gives away" label 1); large negative z
    means it gives away label 0. |z| near 0 means the word is roughly
    equally common in both classes (not label-informative).
    """
    tok0 = [tokenize(r) for r in rationales_label0]
    tok1 = [tokenize(r) for r in rationales_label1]

    counts0, counts1 = Counter(), Counter()
    for t in tok0:
        counts0.update(t)
    for t in tok1:
        counts1.update(t)

    background = Counter()
    background.update(counts0)
    background.update(counts1)
    n_bg = sum(background.values())
    n0 = sum(counts0.values())
    n1 = sum(counts1.values())

    vocab = set(background)
    alpha0 = n_bg  # standard choice: prior strength ~ size of background corpus
    results = {}
    for w in vocab:
        bg_freq = background[w] / n_bg
        a0 = alpha0 * bg_freq  # pseudo-count contribution to class 0
        a1 = alpha0 * bg_freq  # same prior for class 1 (symmetric)

        y0 = counts0[w] + a0
        y1 = counts1[w] + a1

        # log-odds within each class, with the (n - y) denominator using the
        # prior-adjusted total count, per Monroe et al. eq. 16-18
        denom0 = (n0 + alpha0) - y0
        denom1 = (n1 + alpha0) - y1
        if denom0 <= 0 or denom1 <= 0 or y0 <= 0 or y1 <= 0:
            continue

        delta = np.log(y1 / denom1) - np.log(y0 / denom0)
        variance = 1.0 / y0 + 1.0 / y1
        z = delta / np.sqrt(variance)

        results[w] = {
            "z": float(z),
            "count0": int(counts0[w]),
            "count1": int(counts1[w]),
            "log_odds": float(delta),
            "freq_ratio": float(max(counts0[w], counts1[w]) / max(min(counts0[w], counts1[w]), 1)),
        }
    return results


QUOTED_WORD_RE = re.compile(r'"([a-zA-Z]+)"')
TGT_SPAN_RE = re.compile(r"<TGT>(.*?)</TGT>")


def extract_protected_words(rationale_dict: Dict[str, dict], task: str) -> set:
    """
    Extract words that are TARGET CONTENT, not candidate leakage words, so
    they can never be masked regardless of log-odds score.

    WiC: the target word is quoted in the rationale text itself (e.g.
    'In Sentence 1, "draw" signifies...'). We pull every quoted single word
    out of every rationale in the corpus and protect the whole set.

    PS: the target phrase is wrapped in <TGT>...</TGT> inside
    sentence1_marked / sentence2_marked. We extract every word from every
    marked span and protect those.

    Without this, single-word log-odds scoring will flag frequently
    recurring target words as "leaky" purely from sampling imbalance in
    which label a given target word fell on in this split -- that is
    dataset skew, not rationale-generation leakage -- and masking it
    destroys the rationale's actual content.
    """
    protected = set()
    if task == "wic":
        for v in rationale_dict.values():
            rat = v.get("rationale", "")
            for m in QUOTED_WORD_RE.findall(rat):
                protected.add(m.lower())
    elif task == "ps":
        for v in rationale_dict.values():
            for field in ("sentence1_marked", "sentence2_marked"):
                span_text = v.get(field, "")
                for span in TGT_SPAN_RE.findall(span_text):
                    for tok in tokenize(span):
                        protected.add(tok)
    return protected


def rank_biased_words(rationale_dict: Dict[str, dict],
                       protected_words: set = None,
                       min_freq_ratio: float = 1.0) -> List[Tuple[str, dict]]:
    """Convenience wrapper: split a {id: {rationale, label, ...}} dict by
    label and return words ranked by |z| descending, EXCLUDING any word in
    protected_words (target-word/phrase content) from candidacy entirely,
    and also excluding any word whose freq_ratio (max(count0,count1) /
    min(count0,count1)) is below min_freq_ratio.

    Why min_freq_ratio matters: with large sample sizes, a word used
    almost equally often in both classes (e.g. 1489 vs 1168 occurrences,
    ratio 1.27x) can still cross a z-score significance threshold purely
    because of the large N -- that is a statistically reliable but
    practically meaningless imbalance (generic scaffolding language used
    in nearly every rationale), not a word that functions as a label
    shortcut. Requiring a minimum freq_ratio (e.g. 2.0-3.0x) separates
    "statistically detectable" from "practically label-revealing."
    """
    rats0 = [v["rationale"] for v in rationale_dict.values() if v.get("label") == 0]
    rats1 = [v["rationale"] for v in rationale_dict.values() if v.get("label") == 1]
    scores = log_odds_z_scores(rats0, rats1)
    if protected_words:
        scores = {w: s for w, s in scores.items() if w not in protected_words}
    if min_freq_ratio > 1.0:
        scores = {w: s for w, s in scores.items() if s["freq_ratio"] >= min_freq_ratio}
    ranked = sorted(scores.items(), key=lambda kv: abs(kv[1]["z"]), reverse=True)
    return ranked


# --- masking ---

def build_mask_regex(words: List[str]) -> re.Pattern:
    words_sorted = sorted(set(w.lower() for w in words), key=len, reverse=True)
    if not words_sorted:
        # matches nothing
        return re.compile(r"(?!)")
    pattern = r"(?<!\w)(" + "|".join(re.escape(w) for w in words_sorted) + r")(?!\w)"
    return re.compile(pattern, flags=re.IGNORECASE)


def mask_text_with_regex(text: str, regex: re.Pattern) -> str:
    masked = regex.sub("", text)
    masked = re.sub(r"\s{2,}", " ", masked)
    masked = re.sub(r"\s+([,.;:])", r"\1", masked)
    masked = re.sub(r"^\s*,\s*", "", masked)
    masked = re.sub(r",\s*,", ",", masked)
    masked = re.sub(r"\band\s*([,.;:])", r"\1", masked, flags=re.IGNORECASE)
    masked = re.sub(r"\s{2,}", " ", masked)
    return masked.strip()


# --- leakage probe ---

def leakage_probe_accuracy(texts: List[str], labels: List[int], cv: int = 5,
                            seed: int = 42) -> dict:
    """
    Trains a TF-IDF + Logistic Regression classifier on rationale text ALONE
    (no sentence pair, no target word/phrase) to predict the label, via
    stratified k-fold CV. High accuracy here (well above majority baseline)
    means the rationale text leaks label information a downstream encoder
    could shortcut on.
    """
    labels = np.array(labels)
    majority_baseline = max(np.mean(labels == 0), np.mean(labels == 1))

    non_empty_idx = [i for i, t in enumerate(texts) if t and t.strip()]
    if len(non_empty_idx) < cv * 2:
        return {"probe_accuracy": majority_baseline, "majority_baseline": majority_baseline,
                "note": "too few non-empty texts to run CV; reporting majority baseline"}

    texts_f = [texts[i] for i in non_empty_idx]
    labels_f = labels[non_empty_idx]

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 1))
    X = vectorizer.fit_transform(texts_f)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    clf = LogisticRegression(max_iter=1000)
    scores = cross_val_score(clf, X, labels_f, cv=skf)

    return {
        "probe_accuracy": float(scores.mean()),
        "probe_accuracy_std": float(scores.std()),
        "majority_baseline": float(majority_baseline),
        "n_examples_used": len(texts_f),
        "n_emptied_out": len(texts) - len(texts_f),
    }
