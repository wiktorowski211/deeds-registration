
import setuptools

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="deeds",
    version="0.0.1",
    author="Marcin Wiktorowski",
    author_email="wiktorowski211@gmail.com",
    description="Python wrapper around efficient 3D discrete deformable registration for medical images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wiktorowski211/deeds-registration",
    packages=setuptools.find_packages(exclude=("tests",)),
    package_data={"deeds.libs": ["*"]},
    install_requires=[
        'SimpleITK',
        'importlib-resources'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix"
    ],
    python_requires='>=3.6',
)