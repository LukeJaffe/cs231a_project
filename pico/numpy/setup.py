#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from distutils.core import setup, Extension
import numpy as np

ext_modules = [
    Extension('cam', 
        sources = ['cammodule.cpp'], 
        include_dirs = ['/home/username/pico/include'],
        library_dirs = ['/home/username/pico/bin'],
        libraries = ['royale', 'royaleCAPI', 'uvc', 'spectre3'],
        runtime_library_dirs = ['/home/username/pico/bin'],
        extra_compile_args = ['-std=c++11'],
        extra_link_args = ['-Wl,--export-dynamic']
    ) 
]
        

setup(
        name = 'Cam',
        version = '1.0',
        include_dirs = [np.get_include()], #Add Include path of numpy
        ext_modules = ext_modules,
      )
