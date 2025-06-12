from setuptools import setup, find_packages

setup(
    name="esm-mutscan",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "tqdm",
        "transformers>=4.40.0",
        "torch>=2.0.0",
        "tokenizers>=0.14.0"
    ],
    entry_points={
        "console_scripts": [
            "esm-mutscan=esm_mutscan.__main__:main"
        ]
    },
    author="Your Name",
    description="Modular mutation scanning and ML toolkit using ESM models",
    python_requires=">=3.8",
)
