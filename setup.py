import os
import numpy as np
from setuptools import setup, find_packages

DISTNAME = 'doubt'
DESCRIPTION = 'Bringing back uncertainty to machine learning'
URL = 'https://github.com/saattrupdan/doubt'
MAINTAINER = 'Dan Saattrup Nielsen'
MAINTAINER_EMAIL = 'saattrupdan@gmail.com'
LICENSE = 'new BSD'
VERSION = 'alpha'

if __name__ == '__main__':
    setup(
        name = DISTNAME,
        maintainer = MAINTAINER,
        maintainer_email = MAINTAINER_EMAIL,
        packages = find_packages(),
        include_package_data = True,
        description = DESCRIPTION,
        license = LICENSE,
        url = URL,
        version = VERSION,
        zip_safe = False,
        classifiers = [
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            'License :: OSI Approved',
            'Programming Language :: C',
            'Programming Language :: Python',
            'Topic :: Software Development',
            'Topic :: Scientific/Engineering',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX',
            'Operating System :: UNIX',
            'Operating System :: MacOS'
        ],
        install_requires = ['numpy', 'scipy', 'scikit-learn'],
    )
