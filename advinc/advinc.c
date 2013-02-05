#include <Python.h>
/*#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION*/
#include "numpy/arrayobject.h"


typedef void (*inplace_map_binop)(PyArrayMapIterObject *, PyArrayIterObject *);

static void npy_float64_inplace_add(PyArrayMapIterObject *mit, PyArrayIterObject *it)
{
    int index = mit->size;
    while (index--) {
        ((npy_float64*)mit->dataptr)[0] = ((npy_float64*)mit->dataptr)[0] + ((npy_float64*)it->dataptr)[0];

        PyArray_MapIterNext(mit);
        PyArray_ITER_NEXT(it);
    }
}

inplace_map_binop addition_funcs[] = {
npy_float64_inplace_add,
NULL};

int type_numbers[] = {
NPY_FLOAT64,
-1000};



static int
map_increment(PyArrayMapIterObject *mit, PyObject *op, inplace_map_binop add_inplace)
{
    PyArrayObject *arr = NULL;
    PyArrayIterObject *it;
    PyArray_Descr *descr;
    if (mit->ait == NULL) {
        return -1;
    }
    descr = PyArray_DESCR(mit->ait->ao);
    Py_INCREF(descr);
    arr = (PyArrayObject *)PyArray_FromAny(op, descr,
                                0, 0, NPY_ARRAY_FORCECAST, NULL);
    if (arr == NULL) {
        return -1;
    }
    if ((mit->subspace != NULL) && (mit->consec)) {
        if (mit->iteraxes[0] > 0) {  
            PyArray_MapIterSwapAxes(mit, (PyArrayObject **)&arr, 0);
            if (arr == NULL) {
                return -1;
            }
        }
    }
    it = (PyArrayIterObjectPy*)
            Array_BroadcastToShape(arr, mit->dimensions, mit->nd);
    if (it  == NULL) {
        Py_DECREF(arr);	
        
        return -1;
    }

    (*add_inplace)(mit, it);

    Py_DECREF(arr);
    Py_DECREF(it);
    return 0;
}


static PyObject *
inplace_increment(PyObject *dummy, PyObject *args)
{
    PyObject *arg_a = NULL, *index=NULL, *inc=NULL;
    PyArrayObject *a;
    inplace_map_binop add_inplace = NULL; 
    int type_number = -1;
    int i =0;
    PyArrayMapIterObject * mit;

    if (!PyArg_ParseTuple(args, "OOO", &arg_a, &index,
            &inc)) { 
        return NULL;
    }
    if (!PyArray_Check(arg_a)) {
         PyErr_SetString(PyExc_ValueError, "needs an ndarray as first argument");
         return NULL;
    }

    a = (PyArrayObject *) arg_a;
    
    if (PyArray_FailUnlessWriteable(a, "input/output array") < 0) {
        return NULL;
    }   

    if (PyArray_NDIM(a) == 0) {
        PyErr_SetString(PyExc_IndexError, "0-d arrays can't be indexed.");
        return NULL; 
    }
    type_number = PyArray_TYPE(a);  



    while (type_numbers[i] >= 0 && addition_funcs[i] != NULL){    
        if (type_number == type_numbers[i]) {
            add_inplace = addition_funcs[i];
            break;
        }
        i++ ;
    }
    
    if (add_inplace == NULL) {
        PyErr_SetString(PyExc_TypeError, "unsupported type for a"); 
        return NULL;
    }
    mit = (PyArrayMapIterObject *) PyArray_MapIterArray(a, index);
    if (mit == NULL) {
        goto fail;
    }
    if (map_increment(mit, inc, add_inplace) != 0) {
        goto fail;
    }
    
    Py_DECREF(mit);
    
    Py_INCREF(Py_None);
    return Py_None;

fail:
    Py_XDECREF(mit);

    return NULL;
}

static PyMethodDef mymethods[] = {
    { "inplace_increment",inplace_increment,
      METH_VARARGS,
      "increments a numpy array on a set of indexes"},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

PyMODINIT_FUNC
initadvinc(void)
{
   (void)Py_InitModule("advinc", mymethods);
   import_array();
}

