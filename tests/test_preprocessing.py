"""Text cleaning, deduplication, near-duplicate grouping, sampling and text templates."""

import pandas as pd
import pytest

from src.config import ConfigError, load_config
from src.data import preprocess as pp
from src.data.label_rules import LabelRules

PRE = {"unicode_form": "NFKC", "strip_html": True, "collapse_whitespace": True, "keep_case": True}


def test_strip_html_keeps_text_breaks_blocks_and_drops_scripts():
    html = "<div>Senior <b>Chef</b></div><ul><li>Menus</li><li>Teams</li></ul><script>alert(1)</script>&amp; more"
    text = pp.strip_html(html)
    assert "alert" not in text
    assert pp.clean_text(text, PRE) == "Senior Chef Menus Teams & more"
    assert pp.strip_html("R&amp;D lead") == "R&D lead"  # entities alone are decoded
    assert pp.strip_html("5 < 7 and plain text") == "5 < 7 and plain text"


def test_clean_text_normalises_unicode_whitespace_and_junk():
    raw = "Ｐｙｔｈｏｎ dev​\t\n  with � C++\x07"
    assert pp.clean_text(raw, PRE) == "Python dev with C++"
    assert pp.clean_text(None, PRE) == "" and pp.clean_text(float("nan"), PRE) == ""
    assert pp.clean_text("MiXeD", {**PRE, "keep_case": False}) == "mixed"


def _data_cfg():
    return load_config("data")


def test_clean_resumes_uses_html_fallback_and_drops_short_or_uncategorised():
    long_text = " ".join(["word"] * 60)
    raw = pd.DataFrame(
        {
            "resume_id": ["R1", "R2", "R3", "R4"],
            "source_id": [1, 2, 3, 4],
            "resume_text": [long_text, "", "too short", long_text],
            "resume_category": ["HR", "HR", "HR", " "],
            "resume_text_fallback": [None, f"<p>{long_text}</p>", None, None],
        }
    )
    kept, dropped = pp.clean_resumes(raw, _data_cfg())
    assert kept["resume_id"].tolist() == ["R1", "R2"]
    assert kept.set_index("resume_id").loc["R2", "text_source"] == "Resume_html"
    assert kept.set_index("resume_id").loc["R1", "text_source"] == "Resume_str"
    assert dict(zip(dropped["resume_id"], dropped["reason"])) == {"R3": "too_short", "R4": "missing_category"}
    assert kept["text_sha1"].str.len().eq(40).all()


def test_deduplicate_resumes_keeps_lowest_source_id_case_insensitively():
    df = pd.DataFrame(
        {
            "resume_id": ["R9", "R2", "R5"],
            "source_id": [9, 2, 5],
            "resume_text": ["Same Text", "same text", "other"],
            "resume_category": ["HR"] * 3,
        }
    )
    kept, dropped = pp.deduplicate_resumes(df)
    assert sorted(kept["resume_id"]) == ["R2", "R5"]
    assert dropped.to_dict("records") == [{"resume_id": "R9", "duplicate_of": "R2", "reason": "exact_duplicate"}]


def test_clean_and_deduplicate_jobs_record_reasons_and_aliases():
    desc = " ".join(["cook"] * 30)
    raw = pd.DataFrame(
        {
            "job_id": ["J30", "J10", "J20", "J40"],
            "source_id": [30, 10, 20, 40],
            "job_title": ["Line Cook", "line  cook", "Chef", ""],
            "job_description": [desc, desc.upper(), "Apply now", desc],
            "company": ["A", "", None, "B"],
            "location": ["X"] * 4,
            "employment_type": ["Full-time"] * 4,
        }
    )
    kept, dropped = pp.clean_jobs(raw, _data_cfg())
    assert dict(zip(dropped["job_id"], dropped["reason"])) == {"J20": "short_description", "J40": "missing_title"}
    assert kept.set_index("job_id").loc["J10", "company"] is None  # blank optional field -> None
    deduped, aliases = pp.deduplicate_jobs(kept)
    assert deduped["job_id"].tolist() == ["J10"]
    assert aliases.to_dict("records") == [{"alias_job_id": "J30", "job_id": "J10"}]


def test_near_duplicate_groups_are_transitive_and_threshold_bound():
    texts = [
        "python developer building data pipelines in the cloud",
        "python developer building data pipelines in the cloud today",
        "python developer building data pipelines on the cloud today",
        "executive chef running a busy kitchen",
    ]
    groups, pairs = pp.near_duplicate_groups(["d", "b", "c", "a"], texts, threshold=0.8)
    assert groups["d"] == groups["b"] == groups["c"] == "b"  # smallest ID in the component
    assert groups["a"] == "a"
    assert set(map(tuple, pairs[["id_a", "id_b"]].to_numpy())) >= {("d", "b")}
    assert (pairs["similarity"] >= 0.8).all()
    none_groups, none_pairs = pp.near_duplicate_groups(["x", "y"], ["a", "a"], 0.5, method="none")
    assert none_groups.to_dict() == {"x": "x", "y": "y"} and none_pairs.empty
    with pytest.raises(ConfigError):
        pp.near_duplicate_groups(["x", "y"], ["a", "b"], 0.5, method="minhash")


@pytest.fixture
def family_jobs(labels_cfg):
    rules = LabelRules.from_config(labels_cfg)
    titles = ["Software Developer"] * 20 + ["Sales Rep"] * 20 + ["Line Cook"] * 3 + ["Warehouse Associate"] * 7
    jobs = pd.DataFrame({"job_id": [f"J{i:03d}" for i in range(len(titles))], "job_title": titles})
    jobs["job_families"] = jobs["job_title"].map(lambda t: list(rules.assign_families(t)))
    return rules, jobs


def test_sample_jobs_keeps_rare_families_and_is_seeded(family_jobs):
    rules, jobs = family_jobs
    cfg = {"n_jobs": 10, "rare_family_max_jobs": 5, "protect": "mapped_families"}
    sample, stats = pp.sample_jobs(jobs, rules, cfg, seed=1)
    assert len(sample) == 10 and sample["job_id"].is_unique
    assert stats["mode"] == "rare_families_then_hash_priority"
    assert stats["family_counts"]["kitchen"] == {"available": 3, "sampled": 3}  # all rare jobs kept
    again, _ = pp.sample_jobs(jobs.sample(frac=1, random_state=3), rules, cfg, seed=1)
    assert again["job_id"].tolist() == sample["job_id"].tolist(), "input order must not matter"
    other, _ = pp.sample_jobs(jobs, rules, cfg, seed=2)
    assert other["job_id"].tolist() != sample["job_id"].tolist()


def test_sample_jobs_is_stable_when_the_mapping_changes(labels_cfg):
    """Adding a rare family must change the sample only by its jobs plus as many marginal fill jobs."""
    titles = ["Software Developer"] * 300 + ["Sales Rep"] * 300 + ["Line Cook"] * 10 + ["Security Officer"] * 8
    titles += ["Warehouse Associate"] * 382
    jobs = pd.DataFrame({"job_id": [f"J{i:04d}" for i in range(len(titles))], "job_title": titles})
    cfg = {"n_jobs": 400, "rare_family_max_jobs": 20, "protect": "mapped_families"}

    def sample(config):
        rules = LabelRules.from_config(config)
        frame = jobs.assign(job_families=jobs["job_title"].map(lambda t: list(rules.assign_families(t))))
        return set(pp.sample_jobs(frame, rules, cfg, seed=3)[0]["job_id"])

    before = sample(labels_cfg)
    labels_cfg["categories"]["IT"]["related"] = ["data", "security"]  # security becomes a protected rare family
    after = sample(labels_cfg)
    security_ids = {f"J{i:04d}" for i in range(610, 618)}
    newly_forced = security_ids - before
    assert security_ids <= after
    assert after - before == newly_forced
    assert len(before - after) == len(newly_forced), "only marginal fill jobs may drop out"


def test_sample_jobs_round_robins_when_rare_jobs_exceed_budget(family_jobs):
    rules, jobs = family_jobs
    sample, stats = pp.sample_jobs(jobs, rules, {"n_jobs": 9, "rare_family_max_jobs": 1000}, seed=0)
    assert stats["mode"] == "round_robin_rare_families"
    counts = stats["family_counts"]
    assert counts["software"]["sampled"] == counts["sales"]["sampled"] == counts["kitchen"]["sampled"] == 3
    full, stats_all = pp.sample_jobs(jobs, rules, {"n_jobs": None}, seed=0)
    assert len(full) == len(jobs) and stats_all["mode"] == "all"


def test_subsample_resumes_spreads_across_categories():
    df = pd.DataFrame(
        {"resume_id": [f"R{i}" for i in range(30)], "resume_category": ["A"] * 20 + ["B"] * 8 + ["C"] * 2}
    )
    sub = pp.subsample_resumes(df, 6, seed=0)
    assert sub["resume_category"].value_counts().to_dict() == {"A": 2, "B": 2, "C": 2}
    assert len(pp.subsample_resumes(df, None, seed=0)) == 30


def test_templates_render_rows_and_reject_unknown_fields():
    df = pd.DataFrame({"job_title": ["Chef", None], "job_description": ["Cook  well", "Desc"]})
    assert pp.build_texts(df, "{job_title}. {job_description}").tolist() == ["Chef. Cook well", ". Desc"]
    assert pp.render_text({"resume_text": "hi"}, "{resume_text}") == "hi"
    with pytest.raises(ConfigError, match="missing from the data"):
        pp.build_texts(df, "{job_title} {salary}")
