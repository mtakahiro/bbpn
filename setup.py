from setuptools import setup, find_packages
from setuptools.extension import Extension

import os

try:
    from Cython.Build import cythonize
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False

if USE_CYTHON:
    cext = '.pyx'
else:
    cext = '.c'

print('C extension: {0}'.format(cext))


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


readme = open('README.rst').read()
doclink = """


Documentation
-------------

The full documentation can be generated with Sphinx"""

history = open('HISTORY.rst').read().replace('.. :changelog:', '')
desc = open("README.rst").read()

PACKAGE_PATH = os.path.abspath(os.path.join(__file__, os.pardir))
print(PACKAGE_PATH)

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

PACKAGE_PATH = os.path.abspath(os.path.join(__file__, os.pardir))

setup(
    name = "bbpn",
    author = "Takahiro Morishita",
    author_email = "takahiro@ipac.caltech.edu",
    description = "Bye Bye Pink Noise - 1/f noise subtraction tool for JWST cal.fits files",
    license = "",
    url = "https://github.com/mtakahiro",
    download_url = "https://github.com/",
    packages=['bbpn'],#,'example'
    package_dir={'bbpn': 'bbpn'},
    requires=['astropy', 'numpy'],
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    classifiers=[
        "Development Status :: 1 - Planning",
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],
    zip_safe=False,
)
