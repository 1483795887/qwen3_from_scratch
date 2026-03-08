from setuptools import find_packages, setup

setup(
    name="qwen3-from-scratch",
    version="0.0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
)
