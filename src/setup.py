from setuptools import setup, find_packages

setup(
    name="mypackage",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[],  # Add dependencies here
    author="Ahmadsho Akdodshoev",
    author_email="akdodshoev@gmail.com",
    description="JAX implementation of Funk SVD using SGD and BCD.",
    long_description=open("../README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/simply-pleb/HDDA-Recommendation-systems",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)