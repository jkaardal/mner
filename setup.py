import pip
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

""" To install package mner and its dependencies, run:

        python setup.py install

"""

pip.main(['install', 'GPy>=0.6'])

setup(name='mner',
      version='0.1',
      description='Solve low-rank second-order maximum noise entropy problems',
      url='https://github.com/jkaardal/mner/',
      author='Joel T. Kaardal',
      license='MIT',
      packages=['mner', 'mner.solvers', 'mner.util', 'mner.presets'],
      install_requires=[
          'numpy>=1.7.1',
          'scipy>=0.11',
          'theano>=0.8.2',
          'GPy',
          'GPyOpt',
          'pyipm',
      ],
      dependency_links=[
          'https://github.com/jkaardal/GPyOpt/tarball/master#egg=GPyOpt', # as of Aug. 10, 2017, this fork is necessary; optional, only needed for Bayesian optimization
          'https://github.com/jkaardal/pyipm/tarball/master#egg=pyipm', # optional, only needed for interior-point solver
      ])
      
