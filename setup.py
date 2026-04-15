from setuptools import setup, find_packages

setup(
    name="orlock",
    version="0.1.0",
    description="ORLOCK Backend - LLM Integration Server",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
)
