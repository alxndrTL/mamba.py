from setuptools import setup, find_packages

setup(
    name='mambapy',
    version='1.2.0',
    packages=find_packages(),
    install_requires=[
        'torch',
    ],
    author='Alexandre TL',
    author_email='alexandretl3434@gmail.com',
    description='A simple and efficient Mamba implementation in pure PyTorch.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/alxndrTL/mamba.py',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)