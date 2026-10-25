"""Weak-label rules: job-family assignment and (category, job) grading."""

import pandas as pd
import pytest

from src.config import ConfigError, load_config
from src.data.label_rules import RULES, LabelRules, normalize_title

from .conftest import RAW_RESUMES


@pytest.fixture
def rules(labels_cfg):
    return LabelRules.from_config(labels_cfg)


def test_normalize_title():
    assert normalize_title("  Senior\tSoftware   ENGINEER ") == "senior software engineer"
    assert normalize_title("Ｃｈｅｆ") == "chef"  # NFKC folds full-width characters
    assert normalize_title(None) == ""


def test_assign_families_is_multi_label_and_respects_excludes(rules):
    assert rules.assign_families("Software Sales Representative") == ("software", "sales")
    assert rules.assign_families("Security Officer") == ("security",)
    assert rules.assign_families("Security Engineer") == ()
    assert rules.assign_families("Warehouse Associate") == ()
    assert rules.assign_families("") == ()


@pytest.mark.parametrize(
    "category, families, codes, expected",
    [
        ("IT", ["software"], ["IT"], (2, "high", "title_primary_code_match")),
        ("CHEF", ["kitchen"], ["SALE"], (2, "medium", "title_primary_family_uncoded")),
        ("IT", ["software"], [], (2, "medium", "title_primary_no_job_codes")),
        ("IT", ["software"], ["SALE"], (1, "medium", "title_primary_code_mismatch")),
        ("IT", ["data"], ["ANLS"], (1, "medium", "title_related")),
        ("IT", [], ["IT"], (1, "low", "code_only")),
        ("IT", ["sales"], ["IT", "SALE"], (None, "low", "title_unrelated_code_match")),
        ("IT", ["sales"], ["SALE"], (0, "high", "title_unrelated")),
        ("IT", [], ["SALE"], (0, "low", "no_evidence")),
    ],
)
def test_every_grading_rule(rules, category, families, codes, expected):
    judgment = rules.grade(category, families, codes)
    assert (judgment.grade, judgment.confidence, judgment.rule) == expected


def test_primary_match_beats_related_and_unrelated_families(rules):
    # A job in a primary AND an unrelated family is still graded on the primary family.
    assert rules.grade("SALES", ["software", "sales"], ["SALE"]).grade == 2


def test_uninformative_codes_are_ignored(rules):
    # OTHR alone is "no codes", not a contradiction of the title.
    assert rules.grade("IT", ["software"], ["OTHR"]).rule == "title_primary_no_job_codes"
    assert rules.grade("IT", [], ["OTHR"]).rule == "no_evidence"


def test_excluded_and_unknown_categories(rules):
    assert "FARM" not in rules.included_categories
    with pytest.raises(ConfigError, match="excluded"):
        rules.grade("FARM", ["kitchen"], [])
    assert rules.grade("FARM", ["kitchen"], [], allow_excluded=True).grade == 2
    with pytest.raises(ConfigError, match="not in labels.yaml"):
        rules.grade("ASTRONAUT", [], [])


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda c: c["categories"]["IT"].update(related=["nope"]), "unknown families"),
        (lambda c: c["categories"]["IT"].update(related=["software"]), "both primary and related"),
        (lambda c: c["categories"]["IT"].update(primary=[]), "no primary family"),
        (lambda c: c["categories"]["IT"].update(status="maybe"), "status"),
        (lambda c: c["job_families"]["sales"].update(include=["(unclosed"]), "Bad regex"),
        (lambda c: c["job_families"]["sales"].update(include=[]), "no include patterns"),
        (lambda c: c["grading"].pop("no_evidence"), "missing"),
        (lambda c: c["grading"]["no_evidence"].update(grade=3), "grade must be"),
        (lambda c: c["grading"]["no_evidence"].update(confidence="certain"), "confidence must be"),
    ],
)
def test_invalid_configs_are_rejected(labels_cfg, mutate, message):
    mutate(labels_cfg)
    with pytest.raises(ConfigError, match=message):
        LabelRules.from_config(labels_cfg)


def test_rule_order_matches_config_documentation():
    assert RULES[0] == "title_primary_code_match" and RULES[-1] == "no_evidence"


# ---- The real configs/labels.yaml -------------------------------------------------------------


@pytest.fixture(scope="module")
def real_rules():
    return LabelRules.from_config(load_config("labels"))


@pytest.mark.parametrize(
    "title, must_have, must_not_have",
    [
        ("Registered Nurse - ICU", {"healthcare"}, set()),
        ("Staff Accountant", {"accounting"}, set()),
        ("Retail Sales Associate", {"sales", "retail"}, set()),
        ("Business Development Manager", {"business_development"}, set()),
        ("Line Cook", {"culinary"}, set()),
        ("IT Support Specialist", {"software_it"}, set()),
        ("Senior Data Scientist", {"data_analytics"}, set()),
        ("Group Fitness Instructor", {"fitness"}, set()),
        ("Paralegal", {"legal"}, set()),
        ("Security Officer", {"security_public_safety"}, {"software_it"}),
        ("Travel Nurse RN - Labor and Delivery", {"healthcare"}, {"logistics_warehouse"}),
        ("Assistant Manager $15.90 / HR + FT Hours", set(), {"hr"}),
        ("Senior Business Project Manager (non-IT)", {"project_management"}, {"software_it"}),
        ("State Farm Sales Representative", {"sales"}, {"agriculture"}),
        ("Salesforce Communication Cloud", {"software_it"}, {"public_relations"}),
        ("Leadership Coach", set(), {"fitness"}),
        ("Software Engineer III", {"software_it"}, {"engineering"}),
    ],
)
def test_real_mapping_on_known_titles(real_rules, title, must_have, must_not_have):
    families = set(real_rules.assign_families(title))
    assert must_have <= families, f"{title!r} -> {families}"
    assert not (must_not_have & families), f"{title!r} -> {families}"


def test_real_mapping_every_family_is_reachable_or_negative_only(real_rules):
    used = {f for m in real_rules.categories.values() for f in m.primary + m.related}
    assert used <= set(real_rules.families)


@pytest.mark.integration
@pytest.mark.skipif(not RAW_RESUMES.is_file(), reason="data/raw/resume/Resume.csv not present")
def test_real_mapping_covers_every_resume_category(real_rules):
    categories = set(pd.read_csv(RAW_RESUMES, usecols=["Category"])["Category"].dropna())
    assert categories, "no categories read"
    assert categories <= set(real_rules.categories), categories - set(real_rules.categories)
