import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='basic-ml',
    version='1.0.0',
    author='Alan Bi',
    author_email='alan.bi326@gmail.com',
    description='A lightweight package for basic machine learning needs',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/alankbi/basic-ml',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
