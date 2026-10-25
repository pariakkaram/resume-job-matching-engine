"""Weak-supervision rules: assign job families from titles and grade (category, job) pairs.

This module is pure: it works on titles, family names and function codes, and does no file I/O.
``src.data.build_labels`` applies it to the processed datasets to write labelled pairs.
The rules themselves are configured in ``configs/labels.yaml``; the header of that file
documents the grading procedure implemented in :meth:`LabelRules.grade`.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Collection, Iterable, Mapping

from src.config import ConfigError

GRADE_NOT_RELEVANT = 0
GRADE_PARTIAL = 1
GRADE_HIGH = 2
VALID_GRADES = (GRADE_NOT_RELEVANT, GRADE_PARTIAL, GRADE_HIGH)
CONFIDENCE_LEVELS = ("high", "medium", "low")

# Rule names, in the order LabelRules.grade checks them. labels.yaml must define each one.
RULES = (
    "title_primary_code_match",
    "title_primary_family_uncoded",
    "title_primary_no_job_codes",
    "title_primary_code_mismatch",
    "title_related",
    "code_only",
    "title_unrelated_code_match",
    "title_unrelated",
    "no_evidence",
)
CATEGORY_STATUSES = ("included", "excluded")

_WHITESPACE = re.compile(r"\s+")


def normalize_title(title: str | None) -> str:
    """Lowercase, NFKC-normalise and collapse whitespace in a job title."""
    if not title:
        return ""
    text = unicodedata.normalize("NFKC", str(title)).lower()
    return _WHITESPACE.sub(" ", text).strip()


def _compile(patterns: Iterable[str], where: str) -> tuple[re.Pattern[str], ...]:
    compiled = []
    for pattern in patterns or ():
        try:
            compiled.append(re.compile(pattern, re.IGNORECASE))
        except re.error as exc:
            raise ConfigError(f"Bad regex in {where}: {pattern!r} ({exc})") from exc
    return tuple(compiled)


@dataclass(frozen=True)
class JobFamily:
    name: str
    include: tuple[re.Pattern[str], ...]
    exclude: tuple[re.Pattern[str], ...] = ()
    function_codes: frozenset[str] = frozenset()
    description: str = ""

    def matches(self, normalized_title: str) -> bool:
        return any(p.search(normalized_title) for p in self.include) and not any(
            p.search(normalized_title) for p in self.exclude
        )


@dataclass(frozen=True)
class CategoryMapping:
    category: str
    status: str
    primary: tuple[str, ...]
    related: tuple[str, ...]
    note: str = ""

    @property
    def included(self) -> bool:
        return self.status == "included"


@dataclass(frozen=True)
class Judgment:
    """A weak relevance label. ``grade`` is None when the pair is uncertain (left unlabelled)."""

    grade: int | None
    confidence: str
    rule: str


class LabelRules:
    def __init__(
        self,
        families: Mapping[str, JobFamily],
        categories: Mapping[str, CategoryMapping],
        outcomes: Mapping[str, tuple[int | None, str]],
        uninformative_codes: Collection[str] = (),
    ) -> None:
        self.families = dict(families)
        self.categories = dict(categories)
        self.outcomes = dict(outcomes)
        self.uninformative_codes = frozenset(uninformative_codes)
        self._validate()

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "LabelRules":
        """Build rules from the parsed ``labels.yaml``."""
        families = {}
        for name, spec in (cfg.get("job_families") or {}).items():
            spec = spec or {}
            if not spec.get("include"):
                raise ConfigError(f"Job family '{name}' has no include patterns")
            families[name] = JobFamily(
                name=name,
                include=_compile(spec["include"], f"job_families.{name}.include"),
                exclude=_compile(spec.get("exclude") or (), f"job_families.{name}.exclude"),
                function_codes=frozenset(spec.get("function_codes") or ()),
                description=spec.get("description", ""),
            )

        categories = {}
        for name, spec in (cfg.get("categories") or {}).items():
            spec = spec or {}
            categories[name] = CategoryMapping(
                category=name,
                status=spec.get("status", "included"),
                primary=tuple(spec.get("primary") or ()),
                related=tuple(spec.get("related") or ()),
                note=spec.get("note", ""),
            )

        outcomes = {}
        for rule, spec in (cfg.get("grading") or {}).items():
            spec = spec or {}
            outcomes[rule] = (spec.get("grade"), spec.get("confidence", "low"))
        return cls(families, categories, outcomes, cfg.get("uninformative_codes") or ())

    def _validate(self) -> None:
        missing_rules = [rule for rule in RULES if rule not in self.outcomes]
        unknown_rules = [rule for rule in self.outcomes if rule not in RULES]
        if missing_rules or unknown_rules:
            raise ConfigError(
                f"grading must define exactly {list(RULES)}; "
                f"missing {missing_rules}, unknown {unknown_rules}"
            )
        for rule, (grade, confidence) in self.outcomes.items():
            if grade is not None and grade not in VALID_GRADES:
                raise ConfigError(f"grading.{rule}.grade must be one of {VALID_GRADES} or null")
            if confidence not in CONFIDENCE_LEVELS:
                raise ConfigError(f"grading.{rule}.confidence must be one of {CONFIDENCE_LEVELS}")

        for mapping in self.categories.values():
            if mapping.status not in CATEGORY_STATUSES:
                raise ConfigError(
                    f"categories.{mapping.category}.status must be one of {CATEGORY_STATUSES}"
                )
            if mapping.included and not mapping.primary:
                raise ConfigError(f"Included category '{mapping.category}' has no primary family")
            unknown = [f for f in mapping.primary + mapping.related if f not in self.families]
            if unknown:
                raise ConfigError(f"Category '{mapping.category}' references unknown families {unknown}")
            overlap = set(mapping.primary) & set(mapping.related)
            if overlap:
                raise ConfigError(
                    f"Category '{mapping.category}' lists {sorted(overlap)} as both primary and related"
                )

    @property
    def included_categories(self) -> tuple[str, ...]:
        return tuple(name for name, m in self.categories.items() if m.included)

    def assign_families(self, title: str | None) -> tuple[str, ...]:
        """Return every family whose patterns match the title, in config order."""
        normalized = normalize_title(title)
        if not normalized:
            return ()
        return tuple(name for name, family in self.families.items() if family.matches(normalized))

    def primary_codes(self, category: str) -> frozenset[str]:
        mapping = self._mapping(category)
        return frozenset().union(*(self.families[f].function_codes for f in mapping.primary))

    def grade(
        self,
        category: str,
        job_families: Collection[str],
        job_codes: Collection[str],
        *,
        allow_excluded: bool = False,
    ) -> Judgment:
        """Grade one (resume category, job) pair. See the header of labels.yaml for the rules.

        Excluded categories raise ConfigError unless ``allow_excluded`` (used by review reports).
        """
        mapping = self._mapping(category)
        if not mapping.included and not allow_excluded:
            raise ConfigError(f"Category '{category}' is excluded and cannot be graded")
        families = set(job_families)
        codes = set(job_codes) - self.uninformative_codes
        primary_hits = [f for f in mapping.primary if f in families]

        if primary_hits:
            if any(codes & self.families[f].function_codes for f in primary_hits):
                return self._judge("title_primary_code_match")
            if any(not self.families[f].function_codes for f in primary_hits):
                return self._judge("title_primary_family_uncoded")
            if not codes:
                return self._judge("title_primary_no_job_codes")
            return self._judge("title_primary_code_mismatch")
        if families & set(mapping.related):
            return self._judge("title_related")

        code_match = bool(codes & self.primary_codes(category))
        if not families:
            return self._judge("code_only" if code_match else "no_evidence")
        return self._judge("title_unrelated_code_match" if code_match else "title_unrelated")

    def _mapping(self, category: str) -> CategoryMapping:
        try:
            return self.categories[category]
        except KeyError:
            raise ConfigError(f"Resume category '{category}' is not in labels.yaml") from None

    def _judge(self, rule: str) -> Judgment:
        grade, confidence = self.outcomes[rule]
        return Judgment(grade=grade, confidence=confidence, rule=rule)
