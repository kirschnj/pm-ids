#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name='pm',
      version='0.1',
      description='partial monitoring games',
      author='Johannes Kirschner',
      author_email='jkirschner@inf.ethz.ch',
      packages=find_packages(include=['pm']),
      entry_points={
            'console_scripts': [
                  'pm2=pm.main:main',
                  'pm2-aggr=pm.aggregate:main'
            ]
      }
      , install_requires=[
            'numpy',
            'scipy',
            'pandas',
            'osqp'
      ]
      )
