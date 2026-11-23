import importlib.util
import sys
import unittest
from pathlib import Path


def load_tests(
    loader: unittest.TestLoader,
    standard_tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    """Include every workspace package's `tests/` suite when running `unittest discover -s tests`."""
    suite = unittest.TestSuite()
    suite.addTests(standard_tests)
    repo_root = Path(__file__).resolve().parent.parent
    pkg_pattern = pattern or "test_*.py"
    for pkg_dir in sorted((repo_root / "packages").glob("*")):
        tests_dir = pkg_dir / "tests"
        if not tests_dir.is_dir():
            continue
        pkg_prefix = pkg_dir.name.replace("-", "_")
        for test_file in sorted(tests_dir.rglob(pkg_pattern)):
            rel = test_file.relative_to(tests_dir).with_suffix("")
            mod_name = "_".join((pkg_prefix, "tests", *rel.parts))
            spec = importlib.util.spec_from_file_location(mod_name, test_file)
            if spec is not None and spec.loader is not None:
                module = importlib.util.module_from_spec(spec)
                sys.modules[mod_name] = module
                spec.loader.exec_module(module)
                suite.addTests(loader.loadTestsFromModule(module))
    return suite
