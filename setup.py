"""Package setup for AI_Powered_Last_Mile_Delivery_Automation."""
from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="AI_Powered_Last_Mile_Delivery_Automation",
    version="0.1.0",
    author="Juan David Valderrama Artunduaga",
    description="An end-to-end AI and LLMOps pipeline for Last Mile delivery company.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.13",
)
