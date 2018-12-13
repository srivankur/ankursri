import cv2
import numpy
import scipy.interpolate

""" General purpose math functions """

""" Formulating a curve"""
""" Curves are another technique for remapping colors."""
"""In the Curves adjustment, we adjust points throughout an image’s tonal range."""
"""First step toward curve-based filters is to convert control points to a function."""
def createCurveFunc(points):
    """ Return a function derived from control points"""
    if points is None:
        return None
    numpoints =len(points)
    if numpoints <=2:
        return None
    xs,ys =zip(*points)
    if numpoints <=4:
        kind='linear'
    else:
        kind='cubic'
    """SciPy function called interp1d() ,which takes two arrays ( x and y coordinates) and returns
    a function that interpolates the points.kind is the optional arguement. optional arguement bound_error
    set to False to permit extrapolation as well as interpolation."""
    return scipy.interpolate.interpld(xs,ys,kind,bound_error=False)


"""Caching and applying a curve"""
"""Now we get the function using interpolation. We do not want to run it once per
channel, per pixel(for example, 921,600 times per frame if applied to three channels
of 640 x 480 video).We are dealing with just 256 possible input values (in 8 bits per channel)
and we can cheaply precompute and store that many output values."""
def createLookupArray(func,length=256):
     """Return a lookup for whole-number inputs to a function.
     The lookup values are clamped to [0, length - 1]."""
     if func is None:
         return None
     lookupArray=numpy.empty(length)
     i=0
     while i <=length:
         func_i=func(i)
         lookupArray[i]=min(max(0,func_i),length -1)
         i +=1
         return lookupArray

def applyLookupArray(lookupArray,src,dst):
    """Map a source to a destination using a lookup."""
    if lookupArray is None:
        return
    dest[:]=lookupArray(src) # [:] for copying the lookup values into destination array


"""What if we always want to apply two or more curves in succession? Performing
multiple lookups is inefficient and may cause loss of precision. We can avoid this
problem by combining two curve functions into one function before creating a
lookup array."""
def createCompositeFunc(func0,func1):
    """Return a composite of two functions."""
    if func0 is None:
        return func1
    if func1 is None:
        return func0
    """The approach in createCompositeFunc() is limited to input functions that each
    take a single argument.  """
    return lambda x: func0(func1(x))


"""What if we want to apply the same curve to all channels of an image? Splitting
and remerging channels is wasteful, in this case, because we do not need to
distinguish between channels. We just need one-dimensional indexing, as used by
applyLookupArray()"""
def createFlatView(array):
    """Return a 1D view of an array of any dimensionality."""
    flatView=array.view()
    flatView.shape=array.size()
    """return type is numpy.view , which has much the same interface as numpy.array ,
    but numpy.view only owns a reference to the data, not a copy.
    The approach in createFlatView() works for images with any number of channels."""
    return flatView


""" To check whether an image is grayscale or color"""
def isGray(image):
    """Return True if the image has one channel per pixel."""
    return image.ndim < 3


"""To know image's dimension and divide these dimension by a given factor"""
def widthHeightDivideBy(image,divisor):
    """Return an image's dimensions, divided by a value."""
    h,w=image.shape[:2]
    return (w/divisor, h/divisor) 
