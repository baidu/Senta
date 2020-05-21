# -*- coding: UTF-8 -*-
################################################################################
#
#   Copyright (c) 2019  Baidu.com, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

"""
Setup script.

"""
import setuptools
from io import open 

with open("README.md", "r", encoding="UTF-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Senta",
    version="2.0.0",
    author="Baidu NLP",
    author_email="gaocan01@baidu.com",
    description="A sentiment classification tools made by Baidu NLP.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/baidu/senta",
    packages = setuptools.find_packages(),
    #packages = ['Senta'],
    package_dir = {'senta':'senta'},
    package_data = {'senta':['config/*']},
    platforms = "any",
    license='Apache 2.0',
    install_requires=[
        "nltk == 3.4.5", 
        "numpy == 1.14.5",
        "six == 1.11.0",
        "scikit-learn == 0.20.4",
        "sentencepiece == 0.1.83"],
    python_requires='>=3.7',
    classifiers = [
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.7',
          ],
)
