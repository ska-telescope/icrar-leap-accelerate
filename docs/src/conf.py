# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Doxygen Generate --------------------------------------------------------

import os
import sys
import subprocess
import shutil

def configureDoxyfile(input_dir: str, output_dir: str):
    with open('Doxyfile.in', 'r') as file:
        file_data = file.read()

    file_data = file_data.replace('@DOXYGEN_INPUT_DIR@', input_dir)
    file_data = file_data.replace('@DOXYGEN_OUTPUT_DIR@', output_dir)

    with open('Doxyfile', 'w') as file:
        file.write(file_data)

# Check if we're running on Read the Docs' servers
read_the_docs_build = os.environ.get('READTHEDOCS', None) == 'True'

doxygen_xml = ""
breathe_projects = {}

if read_the_docs_build:
    input_dir = '../LeapAccelerate'
    output_dir = 'build'
    configureDoxyfile(input_dir, output_dir)
    subprocess.call('doxygen', shell=True)
    breathe_projects['LeapAccelerate'] = output_dir + '/doxygen/xml'
    doxygen_xml = output_dir + '/doxygen/xml'

# -- Project information -----------------------------------------------------

project = 'LeapAccelerate'
copyright = '2021, Callan Gray'
author = 'Callan Gray'

# The full version, including alpha/beta/rc tags
with open('../../version.txt') as f:
    version = f.read().strip()
release = version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'breathe'
]

source_suffix = ".rst"

# Automatically generate autodoc_doxygen targets
autodoc_default_flags = ['members']

# Automatically generate stub pages
# autosummary_generate = True

breathe_default_project = "LeapAccelerate"
breathe_default_members = ("members", "undoc-members")
breathe_separate_member_pages = True

breathe_projects_source = {
    "LeapAccelerate": ("../../src/icrar", [
        "leap-accelerate/core/stream_out_type.h",
        "leap-accelerate/core/compute_implementation.h",
        "leap-accelerate/algorithm/ILeapCalibrator.h",
        "leap-accelerate/algorithm/cpu/CpuLeapCalibrator.h",
        "leap-accelerate/algorithm/cuda/CudaLeapCalibrator.h",
        "leap-accelerate/math/math_conversion.h",
        "leap-accelerate/math/complex_extensions.h",
        "leap-accelerate/math/vector_extensions.h"
    ])
}

breathe_doxygen_config_options = { '__host__': '' }

breathe_domain_extension = {
    "h": "cpp",
    "cc": "cpp",
    "cu": "cpp"
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']