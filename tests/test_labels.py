"""Weak-label rules: job-family assignment and (category, job) grading."""

import pandas as pd
import pytest

from src.config import ConfigError, load_config
from src.data import build_labels as bl
from src.data.label_rules import RULES, LabelRules, normalize_title
from src.errors import LeakageError

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


@pytest.mark.parametrize(
    "category, title, codes, grade",
    [
        # Approved changes of 2026-10-26 (DECISIONS D47-D51).
        ("CONSULTANT", "SAP Consultant", ["IT"], 2),  # title alone; CNSL code no longer required
        ("HEALTHCARE", "Travel Nurse RN - Telemetry", ["MNFC"], 2),  # clinical title beats unrelated code
        ("HEALTHCARE", "Physical Therapist", [], 2),
        ("HEALTHCARE", "Nurse Recruiter", ["HR"], 1),  # recruiters are excluded from clinical_care
        ("HEALTHCARE", "Pharmaceutical Sales Specialist - Primary Care", ["SALE"], 1),
        ("AVIATION", "A&P Mechanic", [], 2),
        ("AVIATION", "Aircraft Maintenance Technician", [], 2),
        ("AVIATION", "Diesel Mechanic", [], 1),
        ("AVIATION", "Maintenance Technician", [], 1),
        ("AUTOMOBILE", "Aircraft Mechanic", [], 0),  # aircraft mechanics are no longer automotive
        ("AUTOMOBILE", "Automotive Technician", [], 2),
        ("BUSINESS-DEVELOPMENT", "Account Executive", ["SALE"], 2),
        ("BUSINESS-DEVELOPMENT", "Sales Development Representative", ["SALE"], 2),
        ("BUSINESS-DEVELOPMENT", "Retail Sales Associate", ["SALE"], 1),
        ("SALES", "Account Executive", ["SALE"], 2),
        ("FITNESS", "Athletic Trainer", [], 1),
        ("FITNESS", "Physical Therapy Aide", [], 1),
        ("FITNESS", "Personal Trainer", [], 2),
        # Follow-up of 2026-10-26 (D54, D55).
        ("FITNESS", "Line Cook - San Diego KOA Resort", [], 0),
        ("FITNESS", "Front Desk Hospitality - Temecula KOA at Vail Lake Resort", [], 0),
        ("FITNESS", "Aquatics Coordinator", [], 1),
        ("FITNESS", "Spa Supervisor", [], 1),
        ("HEALTHCARE", "Physician Group Negotiator", ["HCPR"], 1),
    ],
)
def test_real_mapping_approved_changes(real_rules, category, title, codes, grade):
    families = real_rules.assign_families(title)
    assert real_rules.grade(category, families, codes).grade == grade, (title, families)


def test_real_mapping_is_approved():
    cfg = load_config("labels")
    assert cfg["status"] == "approved" and str(cfg["approved_on"]) == "2026-10-26"


def test_real_mapping_every_family_is_reachable_or_negative_only(real_rules):
    used = {f for m in real_rules.categories.values() for f in m.primary + m.related}
    assert used <= set(real_rules.families)


@pytest.mark.integration
@pytest.mark.skipif(not RAW_RESUMES.is_file(), reason="data/raw/resume/Resume.csv not present")
def test_real_mapping_covers_every_resume_category(real_rules):
    categories = set(pd.read_csv(RAW_RESUMES, usecols=["Category"])["Category"].dropna())
    assert categories, "no categories read"
    assert categories <= set(real_rules.categories), categories - set(real_rules.categories)


# ---- Applying the rules to a corpus (src/data/build_labels.py) --------------------------------


@pytest.fixture
def corpus(rules):
    jobs = pd.DataFrame(
        {
            "job_id": ["J1", "J2", "J3", "J4", "J5", "J6"],
            "job_title": ["Software Developer", "Data Analyst", "Sales Rep", "Line Cook", "Security Officer", "Clerk"],
            "function_codes": [["IT"], ["ANLS"], ["SALE"], [], [], ["IT"]],
        }
    )
    return bl.assign_job_families(jobs, rules)


def test_assign_job_families_adds_list_and_category_string(corpus):
    assert corpus["job_families"].tolist() == [["software"], ["data"], ["sales"], ["kitchen"], ["security"], []]
    assert corpus["job_category"].tolist() == ["software", "data", "sales", "kitchen", "security", None]


def test_build_category_labels_grades_every_pair_of_included_categories(rules, corpus):
    labels = bl.build_category_labels(corpus, rules, mapping_version="v1")
    assert list(labels.columns) == bl.LABEL_COLUMNS
    assert len(labels) == 3 * len(corpus)  # IT, SALES, CHEF; FARM is excluded
    it = bl.category_grades(labels, "IT")
    assert it == {"J1": 2, "J2": 1, "J3": 0, "J4": 0, "J5": 0, "J6": 1}  # J6: code_only
    assert set(labels["label_source"]) == {"weak:labels.yaml"} and set(labels["mapping_version"]) == {"v1"}
    summary = bl.label_summary(labels)
    assert summary["IT"] == {"grade_2": 1, "grade_1": 2, "uncertain": 0, "grade_0_high": 3, "grade_0_low": 0}


def test_min_support_excludes_thin_categories(rules, corpus):
    labels = bl.build_category_labels(corpus, rules, "v1")
    kept, excluded = bl.apply_min_support(labels, min_highly_relevant=2)
    assert excluded == {"IT": 1, "SALES": 1, "CHEF": 1}
    assert kept.empty
    kept, excluded = bl.apply_min_support(labels, min_highly_relevant=1)
    assert excluded == {} and len(kept) == len(labels)


def _pairs_inputs(rules):
    jobs = pd.DataFrame(
        {
            "job_id": [f"J{i}" for i in range(10)],
            "job_title": ["Software Developer"] * 8 + ["Sales Rep", "Line Cook"],
            "function_codes": [["IT"]] * 6 + [["SALE"]] * 2 + [["SALE"], []],
        }
    )
    labels = bl.build_category_labels(bl.assign_job_families(jobs, rules), rules, "v1")
    resumes = pd.DataFrame(
        {"resume_id": ["R1", "R2", "R3", "R4"], "resume_text": ["a b", "c d", "e f", " "],
         "resume_category": ["IT", "IT", "IT", "IT"]}
    )
    splits = pd.DataFrame(
        {"resume_id": ["R1", "R2", "R3", "R4"], "resume_category": ["IT"] * 4, "split": ["train", "val", "test", "train"]}
    )
    return resumes, splits, labels


def test_positive_pairs_use_train_only_cap_and_confidence(rules):
    resumes, splits, labels = _pairs_inputs(rules)
    cfg = {"positive_min_grade": 2, "positive_min_confidence": "high", "max_positives_per_resume": 4,
           "validation_check": {"require_nonempty_text": True, "forbid_test_resumes": True}}
    pairs = bl.build_positive_pairs(resumes, splits, labels, cfg, seed=7)
    assert list(pairs.columns) == bl.PAIR_COLUMNS
    assert set(pairs["resume_id"]) == {"R1"}  # R2 is val, R3 test, R4 has empty text
    assert len(pairs) == 4  # capped; 6 high-confidence grade-2 jobs were eligible
    assert set(pairs["job_id"]) <= {f"J{i}" for i in range(6)}  # J6/J7: code mismatch -> grade 1
    assert set(pairs["confidence"]) == {"high"} and (pairs["label"] == 2).all()
    again = bl.build_positive_pairs(resumes.iloc[::-1], splits.iloc[::-1], labels, cfg, seed=7)
    assert sorted(again["job_id"]) == sorted(pairs["job_id"]), "sampling must not depend on row order"


def test_positive_pairs_refuse_test_resumes(rules):
    resumes, splits, labels = _pairs_inputs(rules)
    with pytest.raises(LeakageError):
        bl.build_positive_pairs(resumes, splits, labels, {"validation_check": {"forbid_test_resumes": True}}, 0,
                                use_splits=("train", "test"))
