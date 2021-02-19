from setuptools import setup, find_packages
import re
import io

__version__ = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
    io.open('niimasker/__init__.py', encoding='utf_8_sig').read()
    ).group(1)

test_deps = ['pytest-cov',
             'pytest']

extras = {
    'test': test_deps,
}

setup(
    name='niimasker',
    version=__version__,
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*",
                                    "tests"]),
    package_data={'niimasker': ['templates/*.html']},
    license='MIT',
    author='Dan Gale',
    long_description=open('README.md').read(),
    url='https://github.com/danjgale/roi-extractor',
    install_requires=[
        'numpy',
        'pandas',
        'nibabel',
        'nilearn>=0.5.0',
        'natsort',
        'scipy',
        'scikit-learn',
        'matplotlib',
        'jinja2',
        'load_confounds'
    ],
    tests_require=test_deps,
    extras_require=extras,
    setup_requires=['pytest-runner'],
    entry_points={
        'console_scripts': [
            'niimasker=niimasker.cli:main'
            ]
        }
)
