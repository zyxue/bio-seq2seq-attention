#!/usr/bin/env python
# -*- coding: utf-8 -*-


from setuptools import setup, find_packages


requirements = [
    # 'pytorch',
    'tqdm>=4.23.4',
    'biopython>=1.72',
    'pandas>=0.23.1',
    'numpy>=1.14.5',
    'matplotlib>=2.2.2',
    'scipy>=1.1.0',
]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="Zhuyi Xue",
    author_email='zxue@bcgsc.ca',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        # 'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
    ],
    description="seq2seq+attention",
    entry_points={
        'console_scripts': [
            'seq2seq=seq2seq.seq2seq:main',
        ],
    },
    install_requires=requirements,
    # license="MIT license",
    long_description='TODO',
    include_package_data=True,
    keywords='seq2seq attention',
    name='seq2seq',
    packages=find_packages(),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/zyxue/bio-seq2seq-attention',
    version='0.0.1',
    zip_safe=False,
)
