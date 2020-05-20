# OpenCL (oclcl) on Petalisp

This is a slow but functioning backend for [Petalisp](https://github.com/marcoheisig/Petalisp)
that can utilise a graphics card to theoretically run Petalisp code which deals only in real 
numerical arrays faster than on a CPU.

## Dependencies

These can, and probably should, be obtained from Quicklisp:

- [Petalisp](https://github.com/marcoheisig/Petalisp) for some really weird reason
- [oclcl](https://github.com/guicho271828/eazy-opencl) for generating OpenCL code

This cannot be obtained from Quicklisp:

- [eazy-opencl](https://github.com/guicho271828/eazy-opencl) for running OpenCL code

Cloning eazy-opencl into `~/quicklisp/local-projects` will allow you to load it with Quicklisp.

## Bugs

- I have witnessed a few "random" memory faults over many computations, which is scary.
  This might have to do with how eazy-opencl deals with device memory, and how it depends on
  finalisers which a garbage collector with little work to do will seldom run. It is also
  unclear how to manually free said device memory.
- ~~Reduction is supposed to occur over a binary tree, instead of linearly like CL:REDUCE.
  In "concrete" terms, this means we should compute `f(f(x1, x2), f(x3, x4))` instead of
  `f(f(f(x1, x2), x3), x4)`, for a function `f` and values `x1` through `x4`.~~ haha no
  🦎REDUCTIONS🦎ARE🦎GONE🦎🦎🦎
