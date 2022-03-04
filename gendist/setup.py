from setuptools import find_packages, setup

setup(
    name="gendist",
    packages=find_packages(),
    install_requires=[
        "jaxlib",
        "jax"
    ]
)