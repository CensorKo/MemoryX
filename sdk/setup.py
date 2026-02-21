from setuptools import setup, find_packages

setup(
    name="t0ken-memoryx",
    version="1.1.0",
    description="MemoryX Python SDK - 让 AI Agents 轻松拥有持久记忆",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="MemoryX Team",
    author_email="support@t0ken.ai",
    url="https://t0ken.ai",
    packages=find_packages(),
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="memory ai agent llm cognitive memoryx",
    license="MIT",
)
