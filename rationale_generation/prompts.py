"""Prompts used to get label-agnostic rationales from Gemini."""

SYSTEM_PROMPTS = {
    "wic": "You are an expert in lexical semantics and word sense disambiguation.",
    "ps": "You are an expert in lexical semantics and contextual phrase similarity.",
}

# Words that would give the gold relation away. The prompt asks the model to
# avoid them, and filtering/mask_answer_words.py removes any that slip through.
AVOIDED_WORDS = [
    "same", "different", "distinct", "identical", "similar", "contrast",
    "however", "whereas", "both", "alike", "synonymous", "unrelated",
    "diverge", "overlap", "match", "consistent", "inconsistent", "separate",
    "shared", "common",
]
# Phrase similarity is also about paraphrasing, so these give the label away there.
PS_EXTRA_AVOIDED = ["paraphrase", "interchangeable", "substitut(e/ion)"]


def build_wic_prompt(s1_marked: str, s2_marked: str, word: str) -> str:
    # The gold label is deliberately not part of the prompt.
    return f"""You will be given two sentences, each containing the same target word marked with <TGT> and </TGT>. Your task is to write an expert analysis (2-3 sentences) describing how the target word is used in each sentence.

Your reasoning MUST:
(1) Identify the specific meaning of the target word in Sentence 1 (domain, semantic role, conceptual category)
(2) Identify the specific meaning of the target word in Sentence 2
(3) Reference specific contextual clues (surrounding words, syntactic role, collocations) from BOTH sentences

Rules:
- Describe only HOW the word is used in each sentence
- Do NOT state your conclusion about whether the senses match or differ, neither implicitly nor explicitly
- Avoid these words entirely: {", ".join(AVOIDED_WORDS)}
- End your reasoning with [END OF REASONING]
- On a new line output exactly one of:
    Answer: Yes    (if the target word carries the same sense in both sentences)
    Answer: No     (if the target word carries a different sense in each sentence)

Sentence 1: {s1_marked}
Sentence 2: {s2_marked}
Target word: {word}

Reasoning:"""


def build_ps_prompt(s1_marked: str, s2_marked: str, phrase1: str, phrase2: str) -> str:
    avoided = ", ".join(AVOIDED_WORDS + PS_EXTRA_AVOIDED)
    return f"""You will be given two sentences. In Sentence 1, a target phrase is marked with <TGT> and </TGT>. In Sentence 2, a different candidate phrase occupies the same slot, also marked with <TGT> and </TGT>. Your task is to write an expert analysis (2-3 sentences) describing how each marked phrase functions in its sentence.

Your reasoning MUST:
(1) Identify the specific meaning of Phrase 1 in Sentence 1 (domain, semantic role, conceptual category, what it denotes)
(2) Identify the specific meaning of Phrase 2 in Sentence 2 (domain, semantic role, conceptual category, what it denotes)
(3) Reference specific contextual clues (surrounding words, syntactic role, collocations) from BOTH sentences

Rules:
- Describe only HOW each phrase is used in its sentence
- Do NOT state your conclusion about whether the two phrases are semantically similar in this context, neither implicitly nor explicitly
- Avoid these words entirely: {avoided}
- End your reasoning with [END OF REASONING]
- On a new line output exactly one of:
    Answer: Yes    (if the two phrases carry the same meaning in this context)
    Answer: No     (if the two phrases carry a different meaning in this context)

Sentence 1: {s1_marked}
Sentence 2: {s2_marked}
Phrase 1: {phrase1}
Phrase 2: {phrase2}

Reasoning:"""
