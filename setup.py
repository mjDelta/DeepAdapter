from setuptools import setup, find_packages

requirements = [i.strip() for i in open("./requirements.txt").readlines()]
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
	name='deepadapter',
	version='1.0.0',
	author='MJ Zhang',
	author_email='mengji.zhang0809@gmail.com',
	description=long_description,
	packages=(
        find_packages() +
        find_packages(where="./deepadapter/deepadapter") +
        find_packages(where="./deepadapter/models") +
        find_packages(where="./deepadapter/params") +
        find_packages(where="./deepadapter/utils")
    ),
    license='BSD-3',
	classifiers=[
	'Programming Language :: Python :: 3',
	'Operating System :: OS Independent',
	],
	python_requires='>=3.9',
	install_requires=requirements,
)
