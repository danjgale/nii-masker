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
    url='https://github.com/danjgale/nii-masker',
    python_requires='>=3.6.0',
    install_requires=[
        'numpy>=1.16.5',
        'pandas>=1.1.0',
        'nibabel>=3.2.0',
        'nilearn>=0.5.0',
        'natsort>=7.1.1',
        'scipy>=1.5.0',
        'scikit-learn>=0.24.1',
        'matplotlib>=3.3.0',
        'jinja2>=2.11.3',
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
