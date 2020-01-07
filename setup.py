from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import Cython.Compiler.Options
import numpy

Cython.Compiler.Options.annotate = True
Cython.Compiler.Options.get_directive_defaults()['linetrace'] = True
Cython.Compiler.Options.get_directive_defaults()['binding'] = True

extensions = [
    Extension("leglag.two_e_integrals", ["leglag/two_e_integrals.pyx"],
        define_macros=[('CYTHON_TRACE', '1')],
        include_dirs=[numpy.get_include()]),
    Extension("leglag.utilities", ["leglag/utilities.pyx"],
        define_macros=[('CYTHON_TRACE', '1')],
        include_dirs=[numpy.get_include()]),
    Extension("leglag.moller_plesset", ["leglag/moller_plesset.pyx"],
        define_macros=[('CYTHON_TRACE', '1')],
        include_dirs=[numpy.get_include()])
    ]

setup(
        name = "LegLag",
        packages = ["leglag"],
        package_data = {"leglag": ['integral_data/*.dat']},
        install_requires = ['numpy', 'scipy', 'Cython'],
        ext_modules = cythonize(extensions),
        include_dirs = [numpy.get_include()]
)
