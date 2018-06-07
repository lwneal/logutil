#!/usr/bin/env python

from setuptools import setup

setup(name='logutil',
        version='0.1.0',
        description='Utilities for logging statistics over time',
        author='Larry Neal',
        author_email='nealla@lwneal.com',
        packages=[
            'logutil',
        ],
      install_requires=[
          "pytz",
          "tqdm",
          "tensorboard_logger",
      ],
)
