from setuptools import setup, find_packages

with open("README.md", "r") as fh: long_description = fh.read()

setup(name='pytorch_lognormal_distribution',
      version='0.0.1',
      description='A pip installable version of the lognormal mixture distribution from https://github.com/shchur/ifl-tpp/tree/master/code',
      long_description=long_description,
      author='Matthew McDermott; Original: Oleksandr Shchur, Marin Bilos, Stephan Guennemann',
      author_email='mattmcdermott8@gmail.com; Original: shchur@in.tum.de',
      install_requires=['torch'],
      packages=find_packages('.'),
      zip_safe=False)
