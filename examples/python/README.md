## you have to copy the .pyd to the tests.py folder for it to work
## also needs python,numpy,numba,pyplot etc etc and pybind11
## some tests to compare vs numba and referece 
- element wise fusion VS numpy raw 
![element wise fusion VS numpy raw](ewise_vs_raw.png)
- element wise fusion VS numba jit
![element wise fusion VS numba jit](ewise_vs_numba.png)
- map reduce VS numpy raw
![map reduce VS numpy raw](reduce_vs_raw.png)
- map reduce VS numba jit
![map reduce VS numba jit](reduce_vs_numba.png)