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


# -- Project information -----------------------------------------------------

project = 'Leap Accelerate'
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
    'breathe',
    'sphinx.ext.autodoc',
    'sphinxcontrib.autodoc_doxygen',
    'sphinx.ext.autosummary',
    'sphinx_autopackagesummary',
    'recommonmark'
]

# Automatically generate autodoc_doxygen targets
autodoc_default_flags = ['members']
doxygen_xml = '/home/calgray/Code/icrar/leap-accelerate/build/Release/docs/doxygen/xml'

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
        "leap-accelerate/algorithm/cpu/CpuILeapCalibrator.h",
        "leap-accelerate/algorithm/cuda/CudaILeapCalibrator.h"
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