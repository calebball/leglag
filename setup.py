from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import Cython.Build
import numpy

setup(
    ext_modules=[
        Extension(
            "leglag.two_e_integrals",
            ["leglag/two_e_integrals.pyx"],
            include_dirs=[numpy.get_include()],
        ),
        Extension(
            "leglag.moller_plesset",
            ["leglag/moller_plesset.pyx"],
            include_dirs=[numpy.get_include()],
        ),
        Extension(
            "leglag.utilities",
            ["leglag/utilities.pyx"],
            include_dirs=[numpy.get_include()],
        ),
    ],
    cmdclass={"build_ext": Cython.Build.build_ext},
)
