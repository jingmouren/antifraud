from __future__ import print_function
from setuptools import setup, find_packages
import sys
import io

setup(
    name="antifraud",
    version="0.1.0",
    author="",
    author_email="",
    description="fraud detection.",
    long_description=io.open("README.md", encoding="UTF-8").read(),
    license="MIT",
    url="https://github.com/daweicheng/antifraud",
    packages=find_packages(),
    entry_points={
        # 'console_scripts': [
        #     'ternary = ternary.__main__:main'
        # ]
    },
    include_package_data=True,
    classifiers=[
        "Environment :: Web Environment",
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: Chinese',
        'Operating System :: MacOS',
        'Operating System :: Microsoft',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Topic :: NLP',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    install_requires=[
            'pytest>=3.6.3',
            'scikit-learn>=0.20.0',
            'pandas>=0.23.3',
            'numpy>=1.14.3',
            'matplotlib>=2.0.2'
        ],
    zip_safe=True,
)
