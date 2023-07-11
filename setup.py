from setuptools import setup
from Cython.Build import cythonize

setup(
    name='xraysim',
    version='0.6',
    package_dir={'': 'src/pkg'},
    packages=['readgadget', 'readgadget.modules', 'pygadgetreader', 'sphprojection', 'gadgetutils', 'specutils'],
    ext_modules=cythonize('src/pkg/sphprojection/*.pyx'),
    zip_safe=False,
    url='https://github.com/mroncarelli/xraysim',
    license='',
    author='Mauro Roncarelli',
    author_email='mauro.roncarelli@inaf.it',
    description='End-to-end X-ray simulator'
)
