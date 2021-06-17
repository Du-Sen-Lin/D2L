"""
Copyright (C) 2021 WD Co., Ltd.
"""
import setuptools

exec(open('WDcv/version.py').read())

setuptools.setup(
    name="WDcv",
    version=__version__,
    author="Wood",
    author_email="dusl0713@thundersoft.com",
    description="WDcv",
    long_description="WDcv.",
    url="https://github.com/Wood/WDcv",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
