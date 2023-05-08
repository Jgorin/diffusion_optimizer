from setuptools import setup, find_packages

setup(
    name='diffusion_optimizer',
    version='1.0.0',
    author='Joshua Gorin',
    url='N/A',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    description='Evolutionary voronoi neighborhood optimization library'
)