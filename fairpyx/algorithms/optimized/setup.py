from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("optimized.pyx"),  # This should match your .pyx file name
    script_args=['build_ext', '--inplace']
)

