from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

requires = [
    'geopy',
    'shapely',
    'numpy',
    'sympy',
    'transforms3d',
    'gooey',
]

setup(
    name='UAV_Processing',
    version='0.1.0',
    description='A metadata based approach for analyzing UAV datasets',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Abhijit-Dhanda/UAV_Processing',
    author='Abhijit Dhanda',
    author_email='abhijitdhanda@cmail.carleton.ca',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Scientists',
        'License :: MIT License',
        'Programming Language :: Python :: 2.7',
    ],
    keywords='UAV photogrammetry filtration metadata',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=requires,
)