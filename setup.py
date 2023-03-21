from setuptools import setup

setup(
    name='xraysim',
    version='0.2',
    package_dir={'': 'src/pkg'},
    packages=['readgadget', 'readgadget.modules', 'pygadgetreader', 'proj2d'],
    url='https://github.com/mroncarelli/xraysim',
    license='',
    author='Mauro Roncarelli',
    author_email='mauro.roncarelli@inaf.it',
    description='End-to-end X-ray simulator'
)
