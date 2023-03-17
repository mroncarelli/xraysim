from setuptools import setup

setup(
    name='xraysim',
    version='0.1',
    packages=['src.pkg.readgadget', 'src.pkg.readgadget.modules', 'src.pkg.pygadgetreader', 'src.pkg.proj2d'],
    url='https://github.com/mroncarelli/xraysim',
    license='',
    author='Mauro Roncarelli',
    author_email='mauro.roncarelli@inaf.it',
    description='End-to-end X-ray simulator'
)
