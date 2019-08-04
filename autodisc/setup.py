import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="autodisc",
    version="0.0.1",
    author="Chris Reinke, Mayalen Etcheverry",
    author_email="chris.reinke@inria.fr",
    description="Packages for automated discovery experiments by Flowers, Inria.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
