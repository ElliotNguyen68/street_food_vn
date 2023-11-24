import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name="sfivn",
    version="0.0.1",
    author="locnt",
    author_email="elliotnguyen68@gmail.com",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ElliotNguyen68/street_food_vn/tree/dev",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    package_data={"": ["*.txt","*.npy","*.csv"]},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.5",
    install_requires=required
)
