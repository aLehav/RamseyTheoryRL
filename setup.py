from setuptools import setup, find_packages

setup(
    name="RamseyTheoryRL",
    version="0.62",
    packages=find_packages(),
    include_package_data=True,
    author="Adam Lehavi, Steve Vott",
    author_email="svott03@gmail.com, alehavi@usc.edu",
    description="Ramsey Number Explorer",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license="MIT",
    url="https://github.com/aLehav/RamseyTheoryRL",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.10",
    ],
)
