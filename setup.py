
import setuptools
import numpy

from Cython.Build import cythonize
from distutils.extension import Extension

sourcefiles = ['deeds/registration.pyx', 'deeds/libs/deedsBCV0.cpp']
extensions = [Extension(name='deeds.registration', sources=sourcefiles, language="c++",
                        include_dirs=[numpy.get_include()], extra_compile_args=["-O3", "-fopenmp", "-mavx2", "-msse4.2", "-std=c++11"], extra_link_args=['-fopenmp'])]

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="deeds",
    version="1.0.0",
    author="Marcin Wiktorowski",
    author_email="wiktorowski211@gmail.com",
    description="Python wrapper around efficient 3D discrete deformable registration for medical images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wiktorowski211/deeds-registration",
    packages=setuptools.find_packages(exclude=("tests",)),
    ext_modules=cythonize(extensions),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.6',
)
