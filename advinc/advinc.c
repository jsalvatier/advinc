#include <Python.h>


PyMODINIT_FUNC
initadvinc(void)
{
   (void)Py_InitModule("advinc", mymethods);
   import_array();
}

static PyMethodDef mymethods[] = {
    { "advinc",array_ass_sub,
      METH_VARARGS,
      "increments a numpy array on a set of indexes"},
    {NULL, NULL, 0, NULL} /* Sentinel */
}


static int
PyArray_SetMap(PyArrayMapIterObject *mit, PyObject *op)
{
    PyObject *arr = NULL;
    PyArrayIterObject *it;
    int index;
    int swap;
    PyArray_Descr *descr;

    /* Unbound Map Iterator */
    if (mit->ait == NULL) {
        return -1;
    }
    descr = mit->ait->ao->descr;
    Py_INCREF(descr);
    arr = PyArray_FromAny(op, descr, 0, 0, FORCECAST, NULL);
    if (arr == NULL) {
        return -1;
    }
    if ((mit->subspace != NULL) && (mit->consec)) {
        if (mit->iteraxes[0] > 0) {  /* then we need to swap */
            _swap_axes(mit, (PyArrayObject **)&arr, 0);
            if (arr == NULL) {
                return -1;
            }
        }
    }

    /* Be sure values array is "broadcastable"
       to shape of mit->dimensions, mit->nd */

    if ((it = (PyArrayIterObject *)\
         PyArray_BroadcastToShape(arr, mit->dimensions, mit->nd))==NULL) {
        Py_DECREF(arr);
        return -1;
    }

    index = mit->size;

    PyArray_MapIterReset(mit);

    while(index--) {
        memmove(mit->dataptr, it->dataptr, PyArray_ITEMSIZE(arr));
        if (swap) {
        	(double)mit->dataptr = (double)mit->dataptr + (double)it->dataptr; 
        }
        PyArray_MapIterNext(mit);
        PyArray_ITER_NEXT(it);
    }
    Py_DECREF(arr);
    Py_DECREF(it);
    return 0;
}


PyObject* index_increment(PyObject *dummy, PyObject *args)
{
	PyArrayObject *a;
	PyOjbect* index;
	PyObject *inc;
	
	if (!PyArg_ParseTuple(args, "O&OO", 
						&a, PyArray_Converter,
	                    &index,
	                    &inc)) return NULL;
                      
    PyArrayMapIterObject *mit;
    intp vals[MAX_DIMS];

    if (inc == NULL) {
        PyErr_SetString(PyExc_ValueError,
                        "cannot delete array elements");
        return NULL;
    }
    if (!PyArray_ISWRITEABLE(a)) {
        PyErr_SetString(PyExc_RuntimeError,
                        "array is not writeable");
        return NULL;
    }
    
    if (a->nd == 0) {
        PyErr_SetString(PyExc_IndexError, "0-d arrays can't be indexed.");
        return NULL;
    }

    PyErr_Clear();


    mit = (PyArrayMapIterObject *) PyArray_MapIterNew(index, 0, 1);
    if (mit == NULL) {
        return NULL;
    }

    PyArray_MapIterBind(mit, a);
    if (PyArray_SetMap(mit, inc) != 0)
    {
    	PyErr_SetString(PyExc_RuntimeError, "error during mapping");
    	return NULL;
    }
    Py_DECREF(mit);
    
    PyDECREF(a);
    PyDECREF(index);
    PyDECREF(inc);
    
    Py_INCREF(Py_None);
    return Py_None;
    
}


static PyObject *
example_wrapper(PyObject *dummy, PyObject *args)
{
    PyObject *arg_a, *index=NULL, *inc=NULL;
	PyArrayObject *a;
	
    if (!PyArg_ParseTuple(args, "OOO", &arg_a, &index,
        &inc)) return NULL;

    a = PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_INOUT_ARRAY);
    if (a == NULL) return NULL;
    
    if (a->nd == 0) {
        PyErr_SetString(PyExc_IndexError, "0-d arrays can't be indexed.");
        goto fail; 
    }
    
    if (!PyArray_ISWRITEABLE(a)) {
        PyErr_SetString(PyExc_RuntimeError,
                        "array is not writeable");
        goto fail;
    }

	
	//body
	mit = (PyArrayMapIterObject *) PyArray_MapIterNew(index, 0, 1);
    if (mit == NULL) goto fail;

    PyArray_MapIterBind(mit, a);
    if (PyArray_SetMap(mit, inc) != 0)
    {
    	PyErr_SetString(PyExc_RuntimeError, "error during mapping");
    	goto fail;
    }
    
    //endbody
	
	Py_DECREF(mit);

    Py_DECREF(a);
    Py_DECREF(index);
    Py_DECREF(inc);
    
    Py_INCREF(Py_None);
    return Py_None;

 fail:
 	Py_XDECREF(mit);
    Py_XDECREF(a);
    Py_XDECREF(index);
    Py_XDECREF(inc);
    return NULL;
}