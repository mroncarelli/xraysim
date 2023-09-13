from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("sphprojection.kernel", sources=["xraysim/sphprojection/kernel.pyx"]),
    Extension("sphprojection.mapping_loops", sources=["xraysim/sphprojection/mapping_loops.pyx"])
]

setup(
    name='xraysim',
    version='0.6',
    package_dir={'': 'xraysim'},
    packages=['readgadget', 'readgadget.modules', 'pygadgetreader', 'sphprojection', 'gadgetutils', 'specutils'],
    ext_modules=cythonize(extensions, force=True),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
    url='https://github.com/mroncarelli/xraysim',
    license='',
    author='Mauro Roncarelli',
    author_email='mauro.roncarelli@inaf.it',
    description='End-to-end X-ray simulator'
)
