from setuptools import find_packages, setup

setup(
    name="boss",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pymongo",
        "kafka-python",
        "openai",
        "pydantic",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "boss=start:main",
        ],
    },
)
