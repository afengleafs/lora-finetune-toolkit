from setuptools import setup, find_packages

setup(
    name="lora-finetune-toolkit",
    version="0.1.0",
    description="A modular toolkit for LLM fine-tuning with LoRA",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "peft>=0.8.0",
        "trl>=0.7.0",
        "datasets>=2.16.0",
        "bitsandbytes>=0.42.0",
        "accelerate>=0.25.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "lora-train=scripts.train:main",
            "lora-infer=scripts.inference:main",
        ],
    },
)
