# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os

if not os.getenv('TOXTEST'):
    with open('README.md') as f:
        readme = f.read()

    with open('LICENSE') as f:
        license = f.read()
else:
    readme = ''
    license = ''

setup(
    name='parallel_tf_tests',
    version='0.1.0',
    description='Speed tests for TF parallelisation',
    long_description=readme,
    author='Tom Nicholson',
    author_email='tfwnicholson@gmail.com',
    url='https://github.com/tfwnicholson/parallel_tf_tests',
    license=license,
    packages=find_packages(exclude=('parallel_tf_tests')),
    install_requires=['numpy==1.11.1',
                      'scipy==0.17.1',
                      'pandas==0.18.1',
                      'tensorflow==0.12.1']
)
