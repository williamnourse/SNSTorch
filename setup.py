from setuptools import setup, find_packages


def readme_file_contents():
    with open('README.md') as f:
        data = f.read()
    return data

setup(
    name='snstorch',
    version='0.1.1',
    packages=find_packages(),
    url='https://github.com/wnourse05/SNSTorch',
    license='Apache v2.0',
    author='William Nourse',
    author_email='nourse@case.edu',
    description='',
    long_description='',
    setup_requires=['wheel'],
    python_requires='>=3.5',
    install_requires=['torch'],

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: GPU',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ]
)