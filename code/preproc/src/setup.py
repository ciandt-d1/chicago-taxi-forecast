# -*- coding: utf-8 -*-
"""
Package module
"""

import setuptools as setuptools

REQUIRED_PACKAGES = [
    'apache-beam[gcp]==2.11.0',
    'tensorflow-transform==0.13.0',
    'tensorflow==2.6.4'
]

setuptools.setup(
    name='chicago-taxi-trips-forecast',
    version='0.0.1',
    include_package_data=True,
    install_requires=REQUIRED_PACKAGES,
    packages=setuptools.find_packages(),
    author='CI&T'
)
