from setuptools import setup, find_packages

setup(
    name='rextract',
    version='0.0.1',
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*",
                                    "tests"]),
    license='MIT',
    author='Dan Gale',
    long_description=open('README.md').read(),
    url='https://github.com/danjgale/roi-extractor',
    tests_require=[
        'pytest',
        'pytest-cov'
    ],
    setup_requires=['pytest-runner']
)