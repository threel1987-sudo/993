from pathlib import Path


def test_requirements_do_not_include_unused_scikit_learn():
    requirements = Path("requirements.txt").read_text(encoding="utf-8")

    assert "scikit-learn" not in requirements


def test_codebase_does_not_import_sklearn():
    source_files = [
        path
        for path in Path(".").glob("*.py")
        if path.name != "tests"
    ] + list(Path("scripts").glob("*.py"))

    for path in source_files:
        text = path.read_text(encoding="utf-8")
        assert "import sklearn" not in text
        assert "from sklearn" not in text
