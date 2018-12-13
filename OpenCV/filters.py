import cv2
import numpy
import utils

""" Filter functions and classes"""

def recolorRC(src,dst):
    """ Simulate conversion from BGR to RC (red, cyan).
cyan=(255,255,0)[b,g,r]
By averaging the B and G channels and storing the result in both B and G, we effectively
collapse these two channels into one, C.
The source and destination images must both be in BGR format.
Blues and greens are replaced with cyans.
Pseudocode:
dst.b = dst.g = 0.5 * (src.b + src.g)
dst.r = src.r
"""
    b,g,r=cv2.split(src)
    cv2.addWeighted(b,0.5,g,0.5,0,b) #arguements(in order):first src array,a weight applied
                                    # to array, scnd src array, a weight applied to array
                                    # a constant added to the result and a destination array
    cv2.merge((b,b,r),dest)        #replace b and g with modified b(which has both and g)


def recolorRGV(src,dst):
    """Simulate conversion from BGR to RGV (red, green, value).
    yellow=(0,255,255)[b,g,r]
    grey=(211,211,211)[b,g,r] any number between 0-255 for all b ,g,r
    So we can not make b completely zero as it will turn the pixel into yellow
    We want grey remain grey and pale blue become grey
    So we should reduce B values to the per-pixel minimum of B, G, and R
    The source and destination images must both be in BGR format.
    Pseudocode:
    dst.b = min(src.b, src.g, src.r)
    dst.g = src.g
    dst.r = src.r
    """
    b,g,r=cv2.split(src)
    cv2.min(b,g,b) # min() function computes the per-element minimum of the first two arguments
                   # and writes them to the third argument
    cv2.min(b,r,b)
    cv2.merge((b,g,r),dest) # b is modified to the minimum of b,g,r at every pixel


def recolorCMV(src,dst):
    """Simulate conversion from BGR to CMV (cyan, magenta, value).
    cyan=(255,255,0)[b,g,r]
    magenta=(255,0,255)[b,g,r]
    We can see that we can make g and r zero
    instead we will make b values to the per pixel maximum of b,g,r
    The source and destination images must both be in BGR format.
    Pseudocode:
    dst.b = max(src.b, src.g, src.r)
    dst.g = src.g
    dst.r = src.r
    """
    b,g,r=cv2.split(src)
    cv2.max(b,g,b) # max() function computes the per-element maximum of the first two arguments
                   # and writes them to the third argument
    cv2.max(b,r,b)
    cv2.merge((b,g,r),dest) # b is modified to the maximum of b,g,r at every pixel

class VFuncFilter(object):
    """A filter that applies a function to V i.e (value) channel (or all of BGR)."""
    def __init__(self,vFunc=None,dtype=numpy.uint8):
        length=numpy.iinfo(dtype).max +1
        self._vLooupArray=utils.createLookupArray(vFunc,length)

    def apply(self,src,dst):
        """Apply the filter with a BGR or gray source/destination."""
        srcFlatView=utils.flatView(src)
        dstFlatView=utils.flatView(dst)
        utils.applyLookupArray(self._vLooupArray,srcFlatView,dstFlatView)


class VCurveFilter(VFuncFilter):
    """A filter that applies a curve to V (or all of BGR)."""
    def __init__(self,vPoints,dtype=numpy.uint8):
        VFuncFilter.__init__(self,utils.createCurveFunc(vPoints),dtype)


class BGRFuncFilter(object):
    """A filter that applies different functions to each of BGR."""

    def __init__(self,vFunc=None,bFunc=None,gFunc=None, rFunc=None,dtype=numpy.uint8):
        length=numpy.iinfo(dtype).max+1
        self._bLookupArray=utils.createLookupArray(utils.createCompositeFunc(bFunc,vFunc),length)
        self._gLookupArray=utils.createLookupArray(utils.createCompositeFunc(gFunc,vFunc),length)
        self._rLookupArray=utils.createLookupArray(utils.createCompositeFunc(rFunc,vFunc),length)

    def apply(self,src,dst):
        """Apply the filter with a BGR source/destination."""
        b,g,r=cv2.split(src)
        utils.applyLookupArray(self._bLookupArray,b,b)
        utils.applyLookupArray(self._gLookupArray,g,g)
        utils.applyLookupArray(self._rLookupArray,r,r)
        cv2.merge([b,g,r],dst)

class BGRCurveFilter(BGRFuncFilter):
    """A filter that applies different curves to each of BGR."""

    def __init__(self,vPoints=None,bPoints=None,gPoints=None,rPoints=None,dtype=numpy.uint8):
        BGRFuncFilter.__init__(self,utils.createCurveFunc(vPoints),
                                    utils.createCurveFunc(bPoints),
                                    utils.createCurveFunc(gPoints),
                                    utils.createCurveFunc(rPoints),dtype)

"""The choice of control points is based on recommendations by photographer Petteri Sulonen. See
his article on film-like curves at http://www.prime-junta.net/pont/How_to/100_
Curves_and_Films/_Curves_and_films.html"""

"""Emulating Kodak Porta """
class BGRPortaCurveFilter(BGRCurveFilter):
    """A filter that applies Portra-like curves to BGR."""
    def __init__(self,dtype=numpy.uint8):
        BGRCurveFilter.__init__(self,vPoints=[(0,0),(23,20),(157,173),(255,255)],
                                     bPoints=[(0,0),(41,46),(157,173),(255,255)],
                                     gPoints=[(0,0),(23,20),(157,173),(255,255)],
                                     rPoints=[(0,0),(23,20),(157,173),(255,255)],
                                     dtype=dtype)

"""Emulating Fuji Provia """
class BGRProviaCurveFilter(BGRCurveFilter):
    """A filter that applies Provia-like curves to BGR."""
    def __init__(self,dtype=numpy.uint8):
        BGRCurveFilter.__init__(self,bPoints=[(0,0),(35,25),(205,227),(255,255)],
                                     gPoints=[(0,0),(27,21),(196,207),(255,255)],
                                     rPoints=[(0,0),(59,54),(202,210),(255,255)],
                                     dtype=dtype)

"""Emulating Fuji Velvia """
class BGRVelviaCurveFilter(BGRCurveFilter):
    """A filter that applies Velvia-like curves to BGR."""
    def __init__(self,dtype=numpy.uint8):
        BGRCurveFilter.__init__(self,vPoints=[(0,0),(128,118),(221,215),(255,255)],
                                     bPoints=[(0,0),(25,21),(122,153),(165,206),(255,255)],
                                     gPoints=[(0,0),(25,21),(95,102),(181,208),(255,255)],
                                     rPoints=[(0,0),(41,28),(183,209),(255,255)],
                                     dtype=dtype)

"""Emulating Cross Processing """
class BGRCrossProcessCurveFilter(BGRCurveFilter):
    """A filter that applies cross-proces-like curves to BGR."""
    def __init__(self,dtype=numpy.uint8):
        BGRCurveFilter.__init__(self,bPoints=[(0,20),(255,235)],
                                     gPoints=[(0,0),(56,39),(208,226),(255,255)],
                                     rPoints=[(0,0),(56,22),(211,255),(255,255)],
                                     dtype=dtype)

""" Highlighting the edges"""

def strokeEdges(src,dst,blurKsize=7,edgeKsize=5):
    if blurKsize >=3:
        blurredSrc=cv2.medianBlur(src,blurKsize) # medianBlur is preffered for edges preserving
        graySrc=cv2.cvtColor(blurredSrc,cv2.COLOR_BGR2GRAY)
    else:
        graySRc=cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    cv2.Laplacian(graySrc,cv2.CV_8U,graySrc,ksize=edgeKsize)# argu(in order): src image, desired depth
                                                            #of the destination image, destination image, kernel size
    normalizedInverseAlpha=(1.0/255) * (255 - graySrc) #To get black edges on white background invert it
                                                        # and then normalize it
    channels=cv2.split(src)                           # then we can multiply src to normalizedInverseAlpha
    for channel in channels:
        channel[:]=channel * normalizedInverseAlpha
    cv2.merge(channels,dst)


""" Convolution filter: custom kernel"""

class VConvolutionFilter(object):
    """A filter that applies a convolution to V (or all of BGR)."""
    def __init__(self,kernel):
        self._kernel=kernel

    def apply(self,src,dst):
        """Apply the filter with a BGR or gray source/destination."""
        cv2.filter2D(src,-1,self._kernel,dst) #The second argument specifies the per-channel depth of the destination image
                                              #(such as cv2.CV_8U for 8 bits per channel). A negative value (as used here) means
                                              #that the destination image has the same depth as the source image.
"""Sharpening, edge detection, and blur filters use kernels that are highly
    symmetric"""

class SharpenFilter(VConvolutionFilter):
    """A sharpen filter with a 1-pixel radius."""
    def __init__(self):
        """The weights of below kernel sum to 1 . This should be the case whenever we want to leave
            the image's overall brightness unchanged."""
        kernel=numpy.array([[-1, -1, -1],
                            [-1, 9, -1],
                            [-1, -1, -1]])
        VConvolutionFilter.__init__(self,kernel)

class FindEdgesFilter(VConvolutionFilter):
    """An edge-finding filter with a 1-pixel radius."""
    def __init__(self):
        """If we modify a sharpening kernel slightly, so that its weights sum to 0 instead,
        then we have an edge detection kernel that turns edges white and non-edges black."""
        kernel=numpy.array([[-1, -1, -1],
                            [-1, 8, -1],
                            [-1, -1, -1]])
        VConvolutionFilter.__init__(self,kernel)

class BlurFilter(VConvolutionFilter):
    """A blur filter with a 2-pixel radius."""
    def __init__(self):
        """For a blur effect, the weights should sum to 1 and should be positive
        throughout the neighborhood."""
        kernel=numpy.array([[0.4, 0.4, 0.4, 0.4, 0.4],
                            [0.4, 0.4, 0.4, 0.4, 0.4],
                            [0.4, 0.4, 0.4, 0.4, 0.4],
                            [0.4, 0.4, 0.4, 0.4, 0.4],
                            [0.4, 0.4, 0.4, 0.4, 0.4]])
        VConvolutionFilter.__init__(self,kernel)

class EmbossFilter(VConvolutionFilter):
    """An emboss filter with a 1-pixel radius."""
    def __init__(self):
        """Kernels with less symmetry produce an interesting effect. Let's consider
        a kernel that blurs on one side (with positive weights) and sharpens on the
        other (with negative weights). It will produce a ridged or embossed effect."""
        kernel=numpy.array([[-2, -1, 0],
                            [-1, 1, 1],
                            [0, 1, 2]])
        VConvolutionFilter.__init__(self,kernel)
