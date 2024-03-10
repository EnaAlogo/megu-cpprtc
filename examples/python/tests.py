import cpprtc
import numpy as np
import inspect
from time import time
import os
from numba import njit
from template import TEMPLATE

def ptr_decl(ins,outs):
    str = '\n'.join([f'const auto* {ins[i]}_ = reinterpret_cast<float const*>(data[{i}]);' for i in range(len(ins))])+'\n'
    str += '\n'.join([f'auto* {outs[i]}_ = reinterpret_cast<float*>(data[{i+len(ins)}]);' for i in range(len(outs))])+'\n'
    return str

def impl_and_vec_code(ins , outs , rexps, str : str):
    assert len(outs) == len(rexps)
    str_s = '\n'.join([f'auto const {ins[i]} = {ins[i]}_[i];' for i in range(len(ins))]) 
    str_s +='\n'+str
    assign_str = '\n'.join([f'{outs[i]}_[i]={rexps[i]};' for i in range(len(outs))])
    
    vec_str = '\n'.join([f'auto const {ins[i]} = Vec::loadu({ins[i]}_+i);' for i in range(len(ins))])
    vec_str += '\n' + str

    store_str = '\n'.join([f'({rexps[i]}).storeu({outs[i]}_+i);' for i in range(len(outs))])
    
    return str_s , assign_str,vec_str,store_str


def code_template(decls , impl , vimpl):
    return TEMPLATE.replace('${decl}',decls).replace('${impl}',impl).replace('${vimpl}',vimpl).replace("${type}","EWISE")


def get_arg_names(func):
    signature = inspect.signature(func)
    return [param.name for param in signature.parameters.values()]

def get_function_body(func):
    source_lines = inspect.getsource(func).split('def ')[1].split('\n')
    source_lines = source_lines[1:]
    source_lines = [line.lstrip() for line in source_lines]
    return '\n'.join(source_lines)

def autos(str,ins,outs):

    lines = str.split('\n')
    s = set(ins).union(set(outs))
    str = ""
    for line in lines:
        if '=' in line and line.split('=')[0].replace(' ' ,'') not in s:
            s.add(line.split('=')[0].replace(' ',''))
            str +='auto '
        str += line
        if str[-1] != ';':
           str += ';'
    return str


def codegen(tst):
    src = get_function_body(tst).replace('np.','')
    rexrps = [ s.replace(' ','').replace('\n','') for s in src.split('return')[1].split(',')]
    ins = get_arg_names(tst)
    outs = [f'__x_{i}' for i in range(len(rexrps))]
    src = autos(src.split('return')[0],ins,outs)
    impl , assign , vec , store = impl_and_vec_code(ins,outs,rexrps,src)
    return code_template(ptr_decl(ins,outs),impl+assign,vec+store),len(outs)

class ReductionKernel:
    def __init__(self,
                 reduction :str ="(x,y)->x+y",
                 identity : str= "0",
                 map : str = "(x)->x",
                 wapr_shfl_down = "return x.reduce_add();"
                 ):
        code = TEMPLATE.replace("${type}","REDUCE")
        code = code.replace("${rx}",reduction.replace("->"," ")) 
        code = code.replace("${pm}",map.replace("->"," ")) 
        code = code.replace("${ident}",identity)
        code = code.replace("${warp_shfl_down}",wapr_shfl_down)
        #print('\n'.join([f'{i}: {s}' for i,s in enumerate(code.split('\n'))]))
        #exit()
        arg = cpprtc.CompilerConfig(f'kernel_{int(time())}',code)
        arg.arch = 'AVX2'
        arg.lang_standard = 'c++20'
        arg.include_dirs = ['../../../compute']
        self.program = cpprtc.Program(arg)
        self.kernel = self.program.function('Kernel')

    def __call__(self , x :np.array):
        assert x.dtype == np.float32
        return cpprtc.reduce(self.kernel,x)


class Jit:
     def __init__(self,fn) -> None:
        code , outnum = codegen(fn)
        arg = cpprtc.CompilerConfig(f'kernel_{int(time())}',code)
        arg.arch = 'AVX2'
        arg.lang_standard = 'c++20'
        #print('\n'.join([f'{i}: {s}' for i,s in enumerate(code.split('\n'))]))
        #exit()
        arg.include_dirs = ['../compute']
        self.program = cpprtc.Program(arg)
        self.kernel = self.program.function('Kernel')
        self.outn = outnum

     def __call__(self,*args, **kwargs):
        lst = [arg for arg in args]
        lst += [arg for (k,arg) in kwargs.items()]
        s = np.broadcast_shapes(*[l.shape for l in lst])
        lst = [np.ascontiguousarray(np.broadcast_to(l,s)) for l in lst]
        lst += [np.empty(s,np.float32) for _ in range(self.outn)]
        for a in lst:
            a.flags['WRITEABLE'] = True
        cpprtc.ewise(self.kernel,lst)
        return tuple([out for out in lst[-self.outn:]])

#just wierd windows stuff
if os.name == 'nt':  
    cwd = os.getcwd()
    try:
        os.chdir('C:')
    finally:
        os.chdir(cwd)


def testrdjit(x,op):
    return op(x)

def testrd(x):
    return (np.sqrt(x*x)-(x*x)).sum()

@njit(nogil=True,cache=True,parallel=True)
def testrdnumba(x):
    return (np.sqrt(x*x)-(x*x)).sum()


@njit(nogil=True,cache=True,parallel=True)
def testnumba(x,y):
    z = np.sqrt((x*y)-(x*y)) * y + x
    f = y*y+np.maximum(x,z)
    d = f * z / np.minimum(x,y)
    return x*y, z ,np.sqrt(f *z) , np.sqrt(d) / f

def testraw(x,y):
    z = np.sqrt((x*y)-(x*y)) * y + x
    f = y*y+np.maximum(x,z)
    d = f * z / np.minimum(x,y)
    return x*y, z ,np.sqrt(f *z) , np.sqrt(d) / f

@Jit
def test(x,y):
    z = np.sqrt((x*y)-(x*y)) * y + x
    f = y*y+np.max(x,z)
    d = f * z / np.min(x,y)
    return x*y, z ,np.sqrt(f *z) , np.sqrt(d) / f


def ewise():
    times = []
    shapes = [[1<<13],[1<<15],[1<<17],[1<<20],[1<<23],[1<<25]]
    for shape in shapes:
        x = np.ones(*shape,dtype=np.float32)
        y = np.ones(*shape,dtype=np.float32)
        assert all([np.all(f == z) for f,z in zip(test(x,y),testraw(x,y))])
        for i in range(3):
            n  = time()*1e+6
            testraw(x,y)
            f = time()*1e+6
            if i == 1:
                times.append([f-n])
            n  = time()*1e+6
            test(x,y)
            f = time()*1e+6
            if i == 1:
                times[-1].append(f-n)
                
            n  = time()*1e+6
            testnumba(x,y)
            f = time()*1e+6
            if i == 1:
                times[-1].append(f-n)
    
    
    import matplotlib.pyplot as plt
    
    plt.plot(shapes,[p[0] for p in times],label='Raw')
    plt.plot(shapes,[p[1] for p in times],label='Custom jit')
    plt.plot(shapes,[p[2] for p in times],label='Numba')
    
    plt.xlabel('number of elements')
    plt.ylabel('time(us)')
    plt.title('Runtime')
    plt.legend()
    plt.grid(True)
    
    plt.show()    

def redux():
    times = []
    shapes = [[1<<13],[1<<15],[1<<17],[1<<20],[1<<23],[1<<25]]
    
    op =  ReductionKernel( map= "(x)->sqrt(x*x)-(x*x)")

    for shape in shapes:
        x = np.ones(*shape,dtype=np.float32)
        assert testrd(x) == testrdjit(x,op)
        for i in range(3):
            n  = time()*1e+6
            testrd(x)
            f = time()*1e+6
            if i == 1:
                times.append([f-n])
            n  = time()*1e+6
            testrdjit(x,op)
            f = time()*1e+6
            if i == 1:
                times[-1].append(f-n)
                
            n  = time()*1e+6
            testrdnumba(x)
            f = time()*1e+6
            if i == 1:
                times[-1].append(f-n)
    
    
    import matplotlib.pyplot as plt
    
    plt.plot(shapes,[p[0] for p in times],label='Raw')
    plt.plot(shapes,[p[1] for p in times],label='Custom jit')
    plt.plot(shapes,[p[2] for p in times],label='Numba')
    
    plt.xlabel('number of elements')
    plt.ylabel('time(us)')
    plt.title('Runtime')
    plt.legend()
    plt.grid(True)
    
    plt.show()  

