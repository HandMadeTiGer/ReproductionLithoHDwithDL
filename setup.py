from setuptools import setup
import glob
import os.path as osp
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension

ROOT_DIR = osp.dirname(osp.abspath(__file__))

include_dirs = [osp.join(ROOT_DIR)+'/include', osp.join(ROOT_DIR)+'/gds']

libraries = ["cairo", "boost_system", "pthread"]
sources = glob.glob(ROOT_DIR+'/**/*.cpp', recursive=True)

# extra_compile_args = ['-g']
# extra_link_args = ['-g']

setup(
    name='Layout2ImageE2E',
    version='1.0',
    author='?',
    author_email='?',
    description='extract image data infomation from layout',
    long_description='extract image data infomation from layout',
    packages=find_packages(),
    install_requires=[
        'torch',
        # other dependencies
    ],
    ext_modules=[
        CppExtension(
            name='Layout2ImageE2E',
            sources=sources,
            include_dirs=include_dirs,
            libraries=libraries,
            # extra_compile_args=extra_compile_args,
            # extra_link_args=extra_link_args
            )
    ],
    cmdclass={
        'build_ext' : BuildExtension
    }

)