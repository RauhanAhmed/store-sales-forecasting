from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = "-e ."
def get_requirements(requirements_path:str) -> List[str]:
    with open(requirements_path) as file:
        requirements = file.read().split("\n")
    requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
name = "store sales forecasting",
version = 0.0.1,
author = "Rauhan Ahmed Siddiqui",
author_email = "rauhaan.siddiqui@gmail.com",
packages = find_packages(),
requires = get_requirements(r"requirements.txt")
)