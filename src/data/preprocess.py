"""Text cleaning, deduplication, near-duplicate audit, job sampling and text templates.

Every function returns new frames and never mutates its input. Rows that are removed are returned
in a separate ``dropped`` / ``aliases`` frame with a reason, so the data report can account for
every source row.
"""

from __future__ import annotations

import html
import re
import string
import unicodedata
from html.parser import HTMLParser
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.config import ConfigError
from src.data.label_rules import LabelRules, normalize_title
from src.utils.logging_utils import get_logger
from src.utils.reproducibility import stable_seed, text_sha1

log = get_logger(__name__)

_TAG_RE = re.compile(r"<\s*/?\s*[a-zA-Z][^>]*>")
_ENTITY_RE = re.compile(r"&(?:[a-zA-Z]+|#\d+|#x[0-9a-fA-F]+);")
_INVISIBLE_RE = re.compile("[​‌‍⁠﻿�]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS_RE = re.compile(r"\s+")


class _TextExtractor(HTMLParser):
    """Collect text content, putting line breaks at block elements and skipping script/style."""

    BLOCK = {
        "p", "div", "br", "li", "ul", "ol", "tr", "td", "th", "table", "section", "article",
        "header", "footer", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "hr",
    }
    SKIP = {"script", "style", "head", "noscript"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.SKIP:
            self._skip_depth += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data)


def looks_like_html(text: str) -> bool:
    return bool(_TAG_RE.search(text) or _ENTITY_RE.search(text))


def strip_html(text: str) -> str:
    """Remove tags (keeping their text) and decode entities. Plain text is returned unchanged."""
    if not text:
        return ""
    if _TAG_RE.search(text):
        parser = _TextExtractor()
        parser.feed(text)
        parser.close()
        return "".join(parser.parts)
    if _ENTITY_RE.search(text):
        return html.unescape(text)
    return text


def clean_text(text: Any, cfg: Mapping[str, Any] | None = None) -> str:
    """Normalise one text field according to ``data.yaml: preprocessing``.

    Steps: HTML/entity removal (if ``strip_html``), Unicode normalisation (``unicode_form``),
    removal of invisible/replacement/control characters, and whitespace collapsing.
    Missing values become the empty string.
    """
    cfg = cfg or {}
    if text is None or (not isinstance(text, str) and pd.isna(text)):
        return ""
    value = str(text)
    if cfg.get("strip_html", True):
        value = strip_html(value)
    form = cfg.get("unicode_form", "NFKC")
    if form:
        value = unicodedata.normalize(form, value)
    value = _INVISIBLE_RE.sub(" ", value)
    value = _CONTROL_RE.sub(" ", value)
    if cfg.get("collapse_whitespace", True):
        value = _WS_RE.sub(" ", value)
    value = value.strip()
    if not cfg.get("keep_case", True):
        value = value.lower()
    return value


def word_count(text: str) -> int:
    return len(text.split())


def _source_order(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by numeric source ID (falling back to the canonical ID) so 'keep the lowest ID' is defined."""
    numeric = pd.to_numeric(df["source_id"], errors="coerce") if "source_id" in df else None
    keyed = df.assign(_num=numeric if numeric is not None else np.nan)
    id_col = "resume_id" if "resume_id" in df else "job_id"
    return keyed.sort_values(["_num", id_col], na_position="last", kind="mergesort").drop(columns="_num")


# --------------------------------------------------------------------------------------------
# Resumes
# --------------------------------------------------------------------------------------------


def clean_resumes(
    raw: pd.DataFrame, cfg: Mapping[str, Any], text_column: str = "resume_text"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean resume text, apply the HTML fallback, and drop empty/too-short/uncategorised resumes.

    Returns ``(kept, dropped)``; ``dropped`` has ``resume_id, source_id, reason, n_words``.
    ``text_column`` is the source-column name recorded in ``text_source`` for the primary text.
    """
    pre = cfg["preprocessing"]
    fallback_name = cfg["resumes"].get("fallback_text_column") or "fallback"
    primary_label = cfg["resumes"]["columns"].get("resume_text", text_column)
    text = raw["resume_text"].map(lambda t: clean_text(t, pre))
    source = pd.Series(primary_label, index=raw.index, dtype=object)
    if "resume_text_fallback" in raw.columns:
        use_fallback = text.eq("") & raw["resume_text_fallback"].notna()
        fallback_text = raw.loc[use_fallback, "resume_text_fallback"].map(
            lambda t: clean_text(t, {**pre, "strip_html": True})
        )
        text = text.mask(use_fallback, fallback_text)
        source = source.mask(use_fallback, fallback_name)
    category = raw["resume_category"].map(lambda c: clean_text(c, pre))
    out = pd.DataFrame(
        {
            "resume_id": raw["resume_id"],
            "source_id": raw["source_id"],
            "resume_text": text,
            "resume_category": category,
            "text_source": source,
        }
    )
    out["n_words"] = out["resume_text"].map(word_count).astype("int64")

    min_words = int(pre.get("min_resume_words", 1))
    reason = pd.Series(None, index=out.index, dtype=object)
    reason = reason.mask(out["n_words"] < min_words, "too_short")
    reason = reason.mask(out["n_words"].eq(0), "empty")
    reason = reason.mask(out["resume_category"].eq(""), "missing_category")
    dropped = out.loc[reason.notna(), ["resume_id", "source_id", "n_words"]].assign(reason=reason[reason.notna()])
    kept = out.loc[reason.isna()].copy()
    kept["text_sha1"] = kept["resume_text"].map(text_sha1)
    return kept.reset_index(drop=True), dropped.reset_index(drop=True)


def deduplicate_resumes(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop exact duplicates of the (case-folded) cleaned text, keeping the lowest source ID.

    Returns ``(kept, dropped)``; ``dropped`` has ``resume_id, duplicate_of, reason``.
    """
    ordered = _source_order(df)
    key = ordered["resume_text"].str.casefold()
    first_id = ordered.groupby(key, sort=False)["resume_id"].transform("first")
    is_dup = key.duplicated(keep="first")
    dropped = pd.DataFrame(
        {"resume_id": ordered.loc[is_dup, "resume_id"], "duplicate_of": first_id[is_dup], "reason": "exact_duplicate"}
    )
    kept = ordered.loc[~is_dup].sort_index()
    return kept.reset_index(drop=True), dropped.reset_index(drop=True)


def subsample_resumes(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    """Seeded subset of ``n`` resumes, round-robin over categories (used by smoke tests)."""
    if n is None or n >= len(df):
        return df.copy()
    groups = {cat: grp["resume_id"].tolist() for cat, grp in df.groupby("resume_category")}
    chosen = set(_round_robin(groups, n, id_priority(seed)))
    return df[df["resume_id"].isin(chosen)].reset_index(drop=True)


# --------------------------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------------------------

_JOB_TEXT_FIELDS = ("job_title", "job_description", "company", "location", "employment_type")


def clean_jobs(raw: pd.DataFrame, cfg: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean job text fields and drop jobs with no title or a too-short description.

    Returns ``(kept, dropped)``; ``dropped`` has ``job_id, source_id, reason, n_words``.
    """
    pre = cfg["preprocessing"]
    out = raw.copy()
    for col in _JOB_TEXT_FIELDS:
        if col in out.columns:
            cleaned = [clean_text(t, pre) for t in out[col]]
            if col not in ("job_title", "job_description"):
                cleaned = [v or None for v in cleaned]  # blank optional fields become None
            # Explicit object dtype: pandas 3 would otherwise infer a string dtype and turn None into NaN.
            out[col] = pd.Series(cleaned, index=out.index, dtype=object)
    out["n_words_description"] = out["job_description"].map(word_count).astype("int64")

    min_words = int(pre.get("min_job_description_words", 1))
    reason = pd.Series(None, index=out.index, dtype=object)
    reason = reason.mask(out["n_words_description"] < min_words, "short_description")
    reason = reason.mask(out["n_words_description"].eq(0), "missing_description")
    reason = reason.mask(out["job_title"].eq(""), "missing_title")
    dropped = out.loc[reason.notna(), ["job_id", "source_id", "n_words_description"]].assign(
        reason=reason[reason.notna()]
    )
    kept = out.loc[reason.isna()]
    return kept.reset_index(drop=True), dropped.rename(columns={"n_words_description": "n_words"}).reset_index(drop=True)


def job_dedup_key(title: pd.Series, description: pd.Series) -> pd.Series:
    """Normalised (title, description) key used for exact job deduplication."""
    return title.map(normalize_title) + "\x1f" + description.map(normalize_title)


def deduplicate_jobs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop jobs repeating an earlier normalised (title, description), keeping the lowest job ID.

    Returns ``(kept, aliases)``; ``aliases`` maps each dropped ``alias_job_id`` to the kept ``job_id``
    so labels or judgments that reference a dropped copy can be redirected.
    """
    ordered = _source_order(df)
    key = job_dedup_key(ordered["job_title"], ordered["job_description"])
    first_id = ordered.groupby(key, sort=False)["job_id"].transform("first")
    is_dup = key.duplicated(keep="first")
    aliases = pd.DataFrame({"alias_job_id": ordered.loc[is_dup, "job_id"], "job_id": first_id[is_dup]})
    kept = ordered.loc[~is_dup].sort_index()
    return kept.reset_index(drop=True), aliases.reset_index(drop=True)


def near_duplicate_groups(
    ids: Sequence[str],
    texts: Sequence[str],
    threshold: float,
    method: str = "tfidf_cosine",
    chunk_size: int = 1000,
) -> tuple[pd.Series, pd.DataFrame]:
    """Group documents whose TF-IDF cosine similarity is at least ``threshold`` (transitively).

    Returns ``(groups, pairs)``: ``groups`` maps each ID to its group ID (the smallest ID in its
    connected component, so a singleton's group is itself); ``pairs`` lists every pair above the
    threshold with its similarity. ``method="none"`` returns singleton groups and no pairs.
    """
    ids = list(ids)
    if method == "none" or len(ids) < 2:
        return pd.Series(ids, index=ids, dtype=object), pd.DataFrame(columns=["id_a", "id_b", "similarity"])
    if method != "tfidf_cosine":
        raise ConfigError(f"Unknown near-duplicate method '{method}'; use 'tfidf_cosine' or 'none'")

    from sklearn.feature_extraction.text import TfidfVectorizer

    matrix = TfidfVectorizer(sublinear_tf=True, stop_words="english", dtype=np.float32).fit_transform(texts)
    parent = list(range(len(ids)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    rows: list[tuple[str, str, float]] = []
    transposed = matrix.T.tocsc()
    for start in range(0, len(ids), chunk_size):
        block = (matrix[start : start + chunk_size] @ transposed).toarray()
        local, other = np.nonzero(block >= threshold)
        for i_local, j in zip(local, other):
            i = start + int(i_local)
            if j <= i:
                continue
            rows.append((ids[i], ids[j], float(block[i_local, j])))
            root_i, root_j = find(i), find(int(j))
            if root_i != root_j:
                parent[max(root_i, root_j)] = min(root_i, root_j)

    components: dict[int, list[int]] = {}
    for i in range(len(ids)):
        components.setdefault(find(i), []).append(i)
    group_of = {}
    for members in components.values():
        label = min(ids[m] for m in members)
        for m in members:
            group_of[ids[m]] = label
    pairs = pd.DataFrame(rows, columns=["id_a", "id_b", "similarity"])
    return pd.Series(group_of, dtype=object).reindex(ids), pairs


def id_priority(seed: int) -> Callable[[str], tuple[int, str]]:
    """Sampling priority of a record: a hash of (seed, ID), with the ID as tie-breaker.

    A record's priority depends only on its own ID, never on which other records are present,
    so samples drawn "lowest priority first" change minimally when the candidate set changes.
    """
    return lambda record_id: (stable_seed(seed, str(record_id)), str(record_id))


def _round_robin(groups: Mapping[Any, list[str]], n: int, priority: Callable[[str], Any]) -> list[str]:
    """Pick ``n`` distinct IDs cycling over groups (sorted keys), each group ordered by ``priority``."""
    queues = {key: sorted(items, key=priority) for key, items in sorted(groups.items())}
    chosen: list[Any] = []
    seen: set[Any] = set()
    while len(chosen) < n and any(queues.values()):
        for key in list(queues):
            queue = queues[key]
            while queue and queue[0] in seen:
                queue.pop(0)
            if queue:
                item = queue.pop(0)
                seen.add(item)
                chosen.append(item)
                if len(chosen) == n:
                    break
    return chosen


def protected_families(rules: LabelRules, protect: str) -> set[str]:
    if protect == "mapped_families":
        return {
            f
            for cat in rules.included_categories
            for f in rules.categories[cat].primary + rules.categories[cat].related
        }
    if protect == "all":
        return set(rules.families)
    if protect == "none":
        return set()
    raise ConfigError(f"sampling.protect must be mapped_families, all or none (got {protect!r})")


def sample_jobs(
    jobs: pd.DataFrame, rules: LabelRules, cfg: Mapping[str, Any], seed: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Seeded job sample that keeps every job of rare protected families (``data.yaml: sampling``).

    ``jobs`` must have a ``job_families`` column. A protected family is *rare* if it has at most
    ``rare_family_max_jobs`` jobs. If all rare-family jobs fit in ``n_jobs`` they are all taken and
    the rest of the budget is filled with the other jobs of lowest :func:`id_priority` (a seeded
    hash of the job ID, i.e. a uniform random draw); otherwise jobs are taken round-robin across the
    rare families, in priority order, so each is represented.

    Stability: because a job's priority depends only on its own ID, a mapping change that adds
    or removes ``d`` rare-family jobs changes the sample by those jobs plus ``d`` fill jobs at the
    priority margin; every other job stays.
    """
    n_jobs = cfg.get("n_jobs")
    rare_max = int(cfg.get("rare_family_max_jobs", 0))
    ordered = jobs.sort_values("job_id").reset_index(drop=True)
    counts = ordered["job_families"].explode().value_counts()
    priority = id_priority(seed)
    stats: dict[str, Any] = {"n_available": int(len(ordered)), "n_requested": n_jobs}

    if n_jobs is None or n_jobs >= len(ordered):
        stats.update(mode="all", n_sampled=int(len(ordered)), rare_families=[])
        sampled = ordered
    else:
        protected = protected_families(rules, cfg.get("protect", "mapped_families"))
        rare = sorted(f for f in protected if 0 < counts.get(f, 0) <= rare_max)
        members = {
            f: ordered.loc[ordered["job_families"].map(lambda fs, f=f: f in fs), "job_id"].tolist() for f in rare
        }
        rare_ids = {job_id for ids in members.values() for job_id in ids}
        if len(rare_ids) <= n_jobs:
            rest = sorted((j for j in ordered["job_id"] if j not in rare_ids), key=priority)
            chosen = rare_ids | set(rest[: n_jobs - len(rare_ids)])
            mode = "rare_families_then_hash_priority"
        else:
            chosen = set(_round_robin(members, n_jobs, priority))
            mode = "round_robin_rare_families"
        sampled = ordered[ordered["job_id"].isin(chosen)]
        stats.update(mode=mode, n_sampled=int(len(sampled)), rare_families=rare, n_rare_family_jobs=len(rare_ids))

    after = sampled["job_families"].explode().value_counts()
    stats["family_counts"] = {
        f: {"available": int(counts.get(f, 0)), "sampled": int(after.get(f, 0))} for f in rules.families
    }
    stats["n_without_family"] = {
        "available": int(ordered["job_families"].map(len).eq(0).sum()),
        "sampled": int(sampled["job_families"].map(len).eq(0).sum()),
    }
    return sampled.reset_index(drop=True), stats


# --------------------------------------------------------------------------------------------
# Text templates
# --------------------------------------------------------------------------------------------


def template_fields(template: str) -> list[str]:
    return [field for _, field, _, _ in string.Formatter().parse(template) if field]


def _as_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value)


def render_text(record: Mapping[str, Any], template: str) -> str:
    """Render ``template`` (e.g. ``"{job_title}. {job_description}"``) for one record; nulls become ''."""
    fields = template_fields(template)
    missing = [f for f in fields if f not in record]
    if missing:
        raise ConfigError(f"Text template {template!r} uses fields missing from the data: {missing}")
    rendered = template.format(**{f: _as_text(record[f]) for f in fields})
    return _WS_RE.sub(" ", rendered).strip()


def build_texts(df: pd.DataFrame, template: str) -> pd.Series:
    """Render ``template`` for every row of ``df`` (same rules as :func:`render_text`)."""
    fields = template_fields(template)
    missing = [f for f in fields if f not in df.columns]
    if missing:
        raise ConfigError(f"Text template {template!r} uses fields missing from the data: {missing}")
    records = df[fields].astype(object).where(df[fields].notna(), None).to_dict("records")
    return pd.Series([render_text(r, template) for r in records], index=df.index, dtype=object)
