import re
from typing import Any, Dict, List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import spacy  # type: ignore

    _NLP = spacy.load("en_core_web_sm")
except Exception:  # spaCy is optional; degrade gracefully
    _NLP = None

_SKILLS: List[str] = [
    "python",
    "javascript",
    "typescript",
    "java",
    "sql",
    "machine learning",
    "react",
    "node",
    "flask",
    "django",
    "git",
    "docker",
    "aws",
]


def _skill_pattern(skill: str) -> re.Pattern:
    tokens = [t for t in re.split(r"\s+", skill.strip()) if t]
    if len(tokens) > 1:
        return re.compile(r"\b" + r"\s+".join(map(re.escape, tokens)) + r"\b", re.IGNORECASE)
    token = tokens[0] if tokens else skill
    return re.compile(r"\b" + re.escape(token) + r"\b", re.IGNORECASE)


def _extract_skill_matches(text: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    counts: Dict[str, int] = {}
    for skill in _SKILLS:
        pat = _skill_pattern(skill)
        counts[skill] = len(pat.findall(text))

    detected = [s for s, c in counts.items() if c > 0]
    if not detected:
        return [], []

    max_count = max(counts[s] for s in detected) or 1

    matches: List[Dict[str, Any]] = []
    for skill in detected:
        c = counts[skill]
        # Fake-but-plausible: relative frequency -> 70..100%
        pct = int(round(70 + 30 * (c / max_count)))
        pct = max(0, min(100, pct))
        matches.append({"skill": skill, "percent": pct, "count": c})

    matches.sort(key=lambda x: (-x["percent"], x["skill"]))
    return detected, matches


def _extract_email(text: str) -> Optional[str]:
    match = re.search(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", text, re.IGNORECASE)
    return match.group(0) if match else None


def _extract_phone(text: str) -> Optional[str]:
    # Tries to catch common international + local formats without being overly strict.
    match = re.search(
        r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3}[\s.-]?\d{3,4}(?!\d)",
        text,
    )
    if not match:
        return None
    phone = match.group(0).strip()
    if len(re.sub(r"\D", "", phone)) < 10:
        return None
    return phone


def _extract_links(text: str) -> List[str]:
    urls = re.findall(r"\bhttps?://[^\s)>\]]+\b", text, flags=re.IGNORECASE)
    wwws = re.findall(r"\bwww\.[^\s)>\]]+\b", text, flags=re.IGNORECASE)
    links: List[str] = []
    for u in urls + wwws:
        u = u.strip().rstrip(".,;")
        if u.lower().startswith("www."):
            u = "https://" + u
        links.append(u)
    # de-dupe but keep order
    seen = set()
    out: List[str] = []
    for u in links:
        key = u.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(u)
    return out[:4]


def _extract_name(text: str) -> Optional[str]:
    # Heuristic: first non-empty line that looks like a name (letters/spaces only, short).
    lines = [ln.strip() for ln in text.splitlines() if ln and ln.strip()]
    for ln in lines[:12]:
        ln_clean = re.sub(r"\s+", " ", ln)
        if len(ln_clean) < 3 or len(ln_clean) > 40:
            continue
        if "@" in ln_clean or "http" in ln_clean.lower() or "www." in ln_clean.lower():
            continue
        if re.search(r"\b(resume|curriculum vitae|cv)\b", ln_clean, re.IGNORECASE):
            continue
        if re.fullmatch(r"[A-Za-z][A-Za-z .'-]*[A-Za-z]", ln_clean) and len(ln_clean.split()) <= 4:
            return ln_clean
    return None


def _compute_job_match(resume_text: str, job_text: Optional[str]) -> Optional[Dict[str, Any]]:
    if not job_text:
        return None
    job_text = job_text.strip()
    if not job_text:
        return None

    docs = [resume_text, job_text]
    try:
        vec = TfidfVectorizer(stop_words="english")
        tfidf = vec.fit_transform(docs)
    except Exception:
        return None

    sim = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
    pct = int(round(max(0.0, min(1.0, sim)) * 100))
    return {"score": pct}


def _extract_keywords_spacy(text: str) -> List[str]:
    if not _NLP:
        return []
    doc = _NLP(text)
    phrases = {
        chunk.text.strip()
        for chunk in doc.noun_chunks
        if 3 <= len(chunk.text.strip()) <= 40
    }
    return list(phrases)[:10]


def analyze_resume(text: str, job_description: Optional[str] = None):
    original_text = text or ""
    found_skills, skill_matches = _extract_skill_matches(original_text)
    score = min(len(found_skills) * 10, 100)

    details = {
        "name": _extract_name(original_text),
        "email": _extract_email(original_text),
        "phone": _extract_phone(original_text),
        "links": _extract_links(original_text),
        "keywords": _extract_keywords_spacy(original_text),
    }

    job_match = _compute_job_match(original_text, job_description)

    return score, found_skills, details, skill_matches, job_match