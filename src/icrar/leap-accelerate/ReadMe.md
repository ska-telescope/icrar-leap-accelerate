# leap-accelerate-lib

A generic library containing core functionality for leap calibration.

Functionality in this project is split into three independant namespaces/implementations:

* cpu
* cuda

## Compute Implementations

### cpu

Provides computation using Eigen3 math libraries and layout that provides trivial
data layouts compatible with raw buffer copying to acceleration libraries and compatible 
with kernel compilers.

See [http://eigen.tuxfamily.org/index.php?title=Main_Page](Eigen) for more details.

### cuda
Provides Cuda computation using native cuda libraries and buffer objects.

## Components

**core/** - contains files required by other components

**common/** - contains leap specific files used by other components

**exception/** - contains exceptions used by other components 

**math/** - contains generic math extensions

**cuda/** - contains cuda specific classes and helpers

**ms/** - contains abstraction layers for measurement set objects

**model/** - contains data structures for leap calibration

**algorithm/** - contains utility classes and functions for performing leap calibration
