# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as fr:
    requirements = fr.read().splitlines()

setup(
    name="sign_text_to_pose",
    packages=find_packages(),
    version="0.0.1",
    description="Library for generating pose files from text",
    author="Amit Moryossef",
    author_email="amitmoryossef@gmail.com",
    url="https://github.com/sign/text-to-pose",
    keywords=["Sign Language"],
    install_requires=requirements,
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=["Programming Language :: Python :: 3.8"],
)
