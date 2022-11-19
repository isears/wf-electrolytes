import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="wflytes",
    version="0.0.1",
    author="Isaac Sears",
    author_email="isaac.j.sears@gmail.com",
    description="Waveform Electrolytes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/isears/wf-electrolytes",
    project_urls={
        "Bug Tracker": "https://github.com/isears/wf-electrolytes/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    install_requires=["dask", "pandas", "numpy"],
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
