import random
import re

KEYWORDS = ["important", "simple", "example", "because", "together", "future", "river", "light", "balance",
            "practice", "history", "nature", "energy", "community", "journey", "quality", "window", "garden"]
START_WORDS = ["Yes", "Well", "Today", "Sure", "First", "Imagine", "Honestly", "Absolutely"]
END_PHRASE = "Is there anything else I can help with?"


def words(s):
    return re.findall(r"\b\w+\b", s)


def make(kind, rng):
    if kind == "lowercase":
        return ("Your entire response should be in English, and in all lowercase letters. No capital letters are allowed.",
                lambda s: any(c.isalpha() for c in s) and s == s.lower())
    if kind == "uppercase":
        return ("Your entire response should be in English, and in all capital letters.",
                lambda s: any(c.isalpha() for c in s) and s == s.upper())
    if kind == "no_comma":
        return ("Do not use any commas in your response.", lambda s: "," not in s)
    if kind == "max_words":
        n = rng.choice([30, 50, 80])
        return (f"Answer with less than {n} words.", lambda s: 0 < len(words(s)) < n)
    if kind == "min_words":
        n = rng.choice([80, 150])
        return (f"Answer with at least {n} words.", lambda s: len(words(s)) >= n)
    if kind == "keyword":
        w = rng.choice(KEYWORDS)
        return (f'Include the keyword "{w}" in your response.', lambda s: re.search(rf"\b{w}\b", s, re.I) is not None)
    if kind == "end_phrase":
        return (f'Finish your response with this exact phrase "{END_PHRASE}". No other words should follow this phrase.',
                lambda s: s.strip().endswith(END_PHRASE))
    if kind == "quotation":
        return ("Wrap your entire response with double quotation marks.",
                lambda s: len(s.strip()) > 1 and s.strip()[0] == '"' and s.strip()[-1] == '"')
    if kind == "bullets":
        n = rng.choice([2, 3, 4, 5])
        return (f"Your answer must contain exactly {n} bullet points. Use the markdown bullet points such as:\n* This is point 1.",
                lambda s: len(re.findall(r"^\s*[*-]\s", s, re.M)) == n)
    if kind == "title":
        return ("Your answer must contain a title, wrapped in double angular brackets, such as <<poem of joy>>.",
                lambda s: re.search(r"<<[^\n<>]+>>", s) is not None)
    if kind == "start_word":
        w = rng.choice(START_WORDS)
        return (f'Start your response with the word "{w}".',
                lambda s: (words(s)[:1] or [""])[0].lower() == w.lower())
    if kind == "paragraphs":
        n = rng.choice([2, 3])
        return (f"There should be exactly {n} paragraphs. Paragraphs are separated with the markdown divider: ***",
                lambda s: len([p for p in re.split(r"\s?\*\*\*\s?", s) if p.strip()]) == n and s.count("***") == n - 1)
    if kind == "highlight":
        n = rng.choice([1, 2, 3])
        return (f"Highlight at least {n} sections in your answer with markdown, i.e. *highlighted section*.",
                lambda s: len(re.findall(r"\*[^\n\*]+\*", s)) >= n)
    if kind == "postscript":
        return ("At the end of your response, please explicitly add a postscript starting with P.S.",
                lambda s: re.search(r"\s*p\.\s?s\..*$", s, re.I | re.M) is not None)
    if kind == "placeholders":
        n = rng.choice([1, 2, 3])
        return (f"The response must contain at least {n} placeholders represented by square brackets, such as [address].",
                lambda s: len(re.findall(r"\[[^\[\]\n]+\]", s)) >= n)
    raise ValueError(kind)


KINDS = ["lowercase", "uppercase", "no_comma", "max_words", "min_words", "keyword", "end_phrase", "quotation",
         "bullets", "title", "start_word", "paragraphs", "highlight", "postscript", "placeholders"]
CONFLICT = {frozenset(p) for p in [
    ("lowercase", "uppercase"), ("lowercase", "end_phrase"), ("lowercase", "postscript"), ("lowercase", "start_word"),
    ("uppercase", "end_phrase"), ("uppercase", "start_word"), ("uppercase", "keyword"), ("max_words", "min_words"),
    ("quotation", "end_phrase"), ("quotation", "start_word"), ("quotation", "title"), ("quotation", "postscript"),
    ("end_phrase", "postscript"), ("bullets", "paragraphs"), ("max_words", "paragraphs"), ("max_words", "bullets"),
    ("title", "start_word"), ("lowercase", "title")]}


def sample(rng, max_n=2):
    n = 1 if rng.random() < 0.5 else max_n
    kinds = []
    while len(kinds) < n:
        k = rng.choice(KINDS)
        if k not in kinds and all(frozenset((k, o)) not in CONFLICT for o in kinds):
            kinds.append(k)
    made = [make(k, rng) for k in kinds]
    return kinds, " ".join(m[0] for m in made), [m[1] for m in made]


def build_pool(prompts, seed):
    out = []
    for i, p in enumerate(prompts):
        rng = random.Random(seed * 1_000_003 + i)
        kinds, text, checks = sample(rng)
        msgs = ([{"role": "system", "content": p["system"]}] if p.get("system") else [])
        msgs.append({"role": "user", "content": p["user"].rstrip() + "\n\n" + text})
        out.append({"messages": msgs, "kinds": kinds, "checks": checks})
    return out
