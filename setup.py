import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="data-logger-seblee97",  # Replace with your own username
    version="0.0.1",
    author="Sebastian Lee",
    author_email="sebastianlee.1997@yahoo.co.uk",
    description="Data Logger Util",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seblee97/data_loggeer",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=["numpy", "pandas"],
)
