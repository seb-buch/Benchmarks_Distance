from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import os

os.environ["CFLAGS"] = "-w"

ext_modules = [
    Extension(
        "core_ns",
        ["core_ns.pyx"],
        libraries=["m"],
        extra_compile_args=['-fopenmp', "-I"+numpy.get_include()],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    name='core_ns',
    ext_modules=cythonize(ext_modules, gdb_debug=True),
)

