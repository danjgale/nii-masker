from setuptools import setup, find_packages

setup(
    name='niimasker',
    version='0.0.1',
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*",
                                    "tests"]),
    license='MIT',
    author='Dan Gale',
    long_description=open('README.md').read(),
    url='https://github.com/danjgale/roi-extractor',
    install_requires=[
        'numpy',
        'pandas',
        'nibabel',
        'nilearn>=0.5.0',
        'natsort'
    ],
    tests_require=[
        'pytest',
        'pytest-cov'
    ],
    setup_requires=['pytest-runner'],
    entry_points={
        'console_scripts': [
            'niimasker=niimasker.cli:main'
            ]
        }
)

