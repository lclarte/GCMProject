# cython_functions.cpython-39-darwin.so
import cython
cimport numpy as np
from libc cimport math 

cdef float threshold_l = -10
cdef float threshold_V = 1e-3
cdef float bound       = 10.0
cdef int   lim         = 500
cdef float spacing     = 2.0 * bound / lim

cdef float pseudobayes_likelihood_c(float z, float beta):
    if z > threshold_l:
        return math.exp(-beta * math.log(1. + math.exp(-z)))
    # maybe simplifies the expression when z is very small ? 
    return math.exp(beta * z)

cpdef float pseudobayes_likelihood(float z, float beta):
    return pseudobayes_likelihood_c(z, beta)

# === 

cdef float pseudobayes_Z0_quad_argument_c(float z, int y,float w, float sqrtV, float beta = 1.0):
    cdef float localfield = y * (z * sqrtV + w)
    if localfield > threshold_l:
        return math.exp(-beta * math.log(1. + math.exp(-(localfield)))) * math.exp(- z*z / 2.0)
    else:
        return math.exp(beta * localfield) * math.exp(- z*z / 2.0)

cpdef float pseudobayes_Z0_quad_argument(float z, int y,float w, float sqrtV, float beta = 1.0):
    return pseudobayes_Z0_quad_argument_c(z, y, w, sqrtV, beta)

# === 

cdef float pseudobayes_dZ0_quad_argument_c(float z, int y, float w, float sqrtV, float beta = 1.0):
    cdef float localfield = y * (z * sqrtV + w)
    if localfield > threshold_l:
        return z * math.exp(-beta * math.log(1. + math.exp(-localfield))) * math.exp(- z*z / 2)
    else:
        return z * math.exp(beta * localfield) * math.exp(- z*z / 2)

cpdef float pseudobayes_dZ0_quad_argument(float z, int y, float w, float sqrtV, float beta = 1.0):
    return pseudobayes_dZ0_quad_argument_c(z, y, w, sqrtV, beta)

# === 

cpdef float pseudobayes_ddZ0_quad_argument(float z, int y, float w, float sqrtV, float beta = 1.0):
    return (z*z) * pseudobayes_likelihood(y * (z * sqrtV + w), beta) * math.exp(-z*z / 2.) / math.sqrt(2 * math.pi)

# ===