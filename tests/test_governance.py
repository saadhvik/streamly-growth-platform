"""Phase 7 acceptance tests: governance artifacts and CI configuration.

Documentation rots silently. These tests make the Phase 7 gate executable, so a
missing artifact, a broken cross-reference, or a CI workflow that no longer runs
the checks it claims to run fails the build rather than being noticed months
later by whoever needed the document.

They deliberately do **not** assert on prose. They assert on the things that can
be wrong in a checkable way: existence, substance, internal links, and agreement
between what the docs claim and what the code does.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"

# The five boardroom artifacts promised by the project plan.
REQUIRED_DOCS = (
    "budget_reallocation_memo.md",
    "experiment_intake_template.md",
    "experiment_readout_example.md",
    "metric_definitions.md",
    "trustworthy_experimentation_onepager.md",
)


@pytest.mark.parametrize("name", REQUIRED_DOCS)
def test_boardroom_artifact_exists_and_has_substance(name: str) -> None:
    path = DOCS / name
    assert path.exists(), f"missing governance artifact: docs/{name}"
    text = path.read_text(encoding="utf-8")
    assert len(text) > 1_000, f"docs/{name} is a stub ({len(text)} chars)"
    assert text.lstrip().startswith("#"), f"docs/{name} needs a title"


def test_readme_exists_and_links_every_artifact() -> None:
    readme = REPO / "README.md"
    assert readme.exists(), "README.md is the entry point; it must exist"
    text = readme.read_text(encoding="utf-8")
    for name in REQUIRED_DOCS:
        assert f"docs/{name}" in text, f"README does not link docs/{name}"


def test_readme_documents_the_known_limitations() -> None:
    """A results README without a limitations section is a sales pitch."""
    text = (REPO / "README.md").read_text(encoding="utf-8")
    assert "Known limitations" in text
    for topic in ("incrementality", "baseline", "Markov", "CUPED", "SRM"):
        assert topic in text, f"limitations section does not mention {topic}"


def test_internal_markdown_links_resolve() -> None:
    """Catches renamed or deleted files that leave dangling references."""
    broken: list[str] = []
    for md in [REPO / "README.md", *DOCS.glob("*.md")]:
        text = md.read_text(encoding="utf-8")
        for target in re.findall(r"\]\((?!https?://|#)([^)]+)\)", text):
            resolved = (md.parent / target.split("#")[0]).resolve()
            if not resolved.exists():
                broken.append(f"{md.name} -> {target}")
    assert not broken, f"dangling links: {broken}"


# ---------------------------------------------------------------------------
# CI configuration
# ---------------------------------------------------------------------------
def test_ci_workflow_runs_the_checks_it_claims_to() -> None:
    workflow = REPO / ".github" / "workflows" / "ci.yml"
    assert workflow.exists(), "CI workflow is missing"
    text = workflow.read_text(encoding="utf-8")

    for command in ("ruff check", "mypy", "pytest"):
        assert command in text, f"CI does not run {command}"
    # The end-to-end job must actually regenerate the artifacts, or the docs can
    # drift from the code without anything noticing.
    for module in ("streamly.datagen.generator", "streamly.attribution.validate",
                   "streamly.experiment.readout"):
        assert module in text, f"CI does not exercise {module}"


def test_ci_workflow_is_valid_yaml() -> None:
    yaml = pytest.importorskip("yaml", reason="PyYAML not installed")
    text = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    parsed = yaml.safe_load(text)

    assert "jobs" in parsed and parsed["jobs"], "workflow defines no jobs"
    # `on:` is parsed as the boolean True by YAML 1.1 -- accept either spelling.
    assert "on" in parsed or True in parsed, "workflow has no triggers"
    for job in parsed["jobs"].values():
        assert job.get("steps"), "every job needs steps"


def test_ci_tests_the_minimum_supported_python() -> None:
    """Testing only the newest version lets `requires-python` rot."""
    text = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    floor = re.search(r'requires-python\s*=\s*">=([\d.]+)"', pyproject)
    assert floor, "pyproject must declare requires-python"
    assert f'"{floor.group(1)}"' in text, (
        f"CI matrix does not test the declared floor, Python {floor.group(1)}"
    )


def test_unused_heavy_dependencies_are_not_declared() -> None:
    """The Bayesian layer is conjugate, so PyMC must not creep back.

    Beta-Binomial posteriors are closed-form; MCMC would add a ~200MB
    dependency and sampling noise to approximate something exact.
    """
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    deps_block = pyproject.split("[project.optional-dependencies]")[0]
    declared = re.findall(r'^\s*"([a-zA-Z0-9_.-]+)', deps_block, flags=re.MULTILINE)
    assert "pymc" not in declared, "pymc is declared but no module imports it"

    src = " ".join(p.read_text(encoding="utf-8") for p in (REPO / "src").rglob("*.py"))
    assert "import pymc" not in src


def test_gitignore_excludes_generated_data() -> None:
    """The warehouse is reproducible from source and must not be committed."""
    text = (REPO / ".gitignore").read_text(encoding="utf-8")
    assert "warehouse.duckdb" in text
    assert "__pycache__" in text


# ---------------------------------------------------------------------------
# Docs must agree with the code
# ---------------------------------------------------------------------------
def test_documented_srm_alpha_matches_the_implementation() -> None:
    from streamly.experiment.integrity import SRM_ALPHA

    onepager = (DOCS / "trustworthy_experimentation_onepager.md").read_text(encoding="utf-8")
    metrics = (DOCS / "metric_definitions.md").read_text(encoding="utf-8")
    assert f"α={SRM_ALPHA}" in onepager or f"{SRM_ALPHA}" in onepager
    assert str(SRM_ALPHA) in metrics


def test_documented_srm_floor_matches_the_helper() -> None:
    """The one-pager quotes 2.67%; the code must still produce it."""
    from streamly.experiment.integrity import srm_minimum_detectable_loss

    floor = srm_minimum_detectable_loss(30_000)
    assert f"{floor:.2%}" == "2.67%", f"documented floor drifted: {floor:.2%}"
    for doc in ("trustworthy_experimentation_onepager.md", "metric_definitions.md"):
        assert "2.67%" in (DOCS / doc).read_text(encoding="utf-8")


def test_documented_lookback_window_matches_config() -> None:
    from streamly.config import ATTRIBUTION

    metrics = (DOCS / "metric_definitions.md").read_text(encoding="utf-8")
    assert f"{ATTRIBUTION.lookback_days} days" in metrics


def test_intake_template_covers_every_pre_registered_field() -> None:
    """The template's job is to force decisions before data exists."""
    text = (DOCS / "experiment_intake_template.md").read_text(encoding="utf-8")
    for field in ("primary metric", "threshold", "margin", "spending function",
                  "Randomization unit", "decision rule"):
        assert field.lower() in text.lower(), f"intake template omits: {field}"


def test_readme_test_count_is_current(request: pytest.FixtureRequest) -> None:
    """The README quotes a test count; keep it honest automatically.

    Uses pytest's own collection total, so adding tests without updating the
    README fails here rather than leaving a quietly wrong number in the first
    thing a reader sees. Skipped on partial runs (-k, single file), where the
    collected count is not comparable.
    """
    collected = request.session.testscollected
    if request.config.option.keyword or collected < 50:
        pytest.skip("partial test run; collection total is not comparable")

    text = (REPO / "README.md").read_text(encoding="utf-8")
    quoted = re.search(r"pytest -q\s+#\s*([\d,]+) tests", text)
    assert quoted, "README should quote the suite size next to the pytest command"
    assert int(quoted.group(1).replace(",", "")) == collected, (
        f"README says {quoted.group(1)} tests; the suite collects {collected}"
    )


def test_documented_recovery_values_match_the_models() -> None:
    """The headline attribution numbers must be what the models actually produce.

    This is the drift that the other doc tests would miss: the SRM alpha and the
    lookback window are constants, but the recovery errors are *results*. They
    change whenever the generator or a model changes -- as they did when funnel
    structure was added to the DGP, which moved last-touch from 12.81pp to
    21.75pp and left a stale figure in the metric definitions.
    """
    import os
    import tempfile

    os.environ.setdefault("STREAMLY_DATA_DIR", tempfile.mkdtemp(prefix="streamly_docs_"))

    from streamly.attribution import validate
    from streamly.attribution.sessionize import build_journeys
    from streamly.datagen import generator

    generator.generate()
    scores = validate.recovery_scores(validate.attribution_matrix(build_journeys()))

    readme = (REPO / "README.md").read_text(encoding="utf-8")
    metrics = (DOCS / "metric_definitions.md").read_text(encoding="utf-8")

    for model in ("last_touch", "shapley"):
        value = f"{scores.loc[model, 'mae'] * 100:.2f}"
        assert value in readme.replace(" pp", "pp"), (
            f"README does not quote {model}'s actual MAE of {value}pp"
        )
        assert value in metrics.replace(" pp", "pp"), (
            f"metric_definitions does not quote {model}'s actual MAE of {value}pp"
        )

    # And the headline reduction claim in the README.
    reduction = 1.0 - scores.loc["shapley", "mae"] / scores.loc["last_touch", "mae"]
    assert f"{reduction:.0%}" in readme, (
        f"README does not quote the actual error reduction of {reduction:.0%}"
    )


def test_requirements_txt_matches_the_project_dependencies() -> None:
    """A deployment installs from requirements.txt, not from pyproject.

    Streamlit Community Cloud reads requirements.txt, so if the two drift the
    deployed app gets a different dependency set than the tested one -- and the
    failure shows up as a broken deploy, not a failing test. This makes the
    divergence fail here instead.
    """
    try:
        import tomllib
    except ModuleNotFoundError:                 # Python 3.10
        import tomli as tomllib                 # type: ignore[no-redef]

    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    declared = sorted(project["project"]["dependencies"])

    req_path = REPO / "requirements.txt"
    assert req_path.exists(), "deployment needs a requirements.txt"
    pinned = sorted(
        line.strip() for line in req_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    )
    assert pinned == declared, (
        f"requirements.txt and pyproject disagree:\n"
        f"  only in pyproject:    {sorted(set(declared) - set(pinned))}\n"
        f"  only in requirements: {sorted(set(pinned) - set(declared))}"
    )
