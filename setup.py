from setuptools import setup

setup(name='NO3sis',
    version=0.1,
    url='https://github.com/SimJeg/NO3sis.git',
    author='Simon Jegou',
    description='Playing with faces on videos',
    license='MIT',
    install_requires = ['opencv-python'],
    long_description=open('README.md').read()
    )