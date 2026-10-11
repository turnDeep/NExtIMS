"""Install the current commands and legacy import-compatible modules."""
from pathlib import Path
from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent
requirements = [line.strip() for line in (ROOT / "requirements.txt").read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")]
setup(
    name="nextims", version="1.1.0", author="turnDeep",
    description="Graph-based EI mass spectrum prediction with neutral-loss fusion",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/turnDeep/NExtIMS",
    packages=find_packages(include=["src", "src.*", "scripts", "scripts.*"]),
    install_requires=requirements, python_requires=">=3.10",
    entry_points={"console_scripts": [
        "nextims=scripts.nextims:main",
        "nextims-train=scripts.nextims:train_main",
        "nextims-evaluate=scripts.nextims:evaluate_main",
        "nextims-predict=scripts.nextims:predict_main",
    ]},
)
