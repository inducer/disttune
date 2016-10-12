#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.rst", "rt") as inf:
    readme = inf.read()

setup(name="disttune",
      version="2015.1",
      description="Distributed Autotuning with PostgreSQL",
      long_description=readme,
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Developers',
          'Intended Audience :: Other Audience',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Natural Language :: English',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'Topic :: Scientific/Engineering :: Visualization',
          'Topic :: Software Development :: Libraries',
          'Topic :: Utilities',
          ],

      author="Andreas Kloeckner",
      url="http://pypi.python.org/pypi/datapyle",
      author_email="inform@tiker.net",
      license="MIT",
      packages=find_packages(),

      install_requires=[
          "psycopg2",
          ])
