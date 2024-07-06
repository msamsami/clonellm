from __future__ import annotations

import codecs
from os import path

from setuptools import find_packages, setup

from clonellm import __version__


def get_requirements(filename: str) -> list[str]:
    return [p for line in open(filename).readlines() if (p := line.replace("\n", "").strip()) and not p.startswith("#")]


requirements = get_requirements("requirements.txt")
requirements_dev = list(set(get_requirements("requirements_dev.txt")) - set(requirements))

setup(
    name="clonellm",
    version=__version__,
    description="Python package to create an AI clone of yourself using LLMs.",
    keywords=["llm", "language models", "nlp", "rag", "ai", "ai clone"],
    author="Mehdi Samsami",
    author_email="mehdisamsami@live.com",
    license="MIT License",
    url="https://github.com/msamsami/clonellm",
    long_description=codecs.open(path.join(path.abspath(path.dirname(__file__)), "README.md"), encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    include_package_data=True,
    package_data={"clonellm": ["py.typed"]},
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.9,<3.13",
    install_requires=requirements,
    extras_require={"chroma": ["langchain-chroma"], "dev": requirements_dev},
)
