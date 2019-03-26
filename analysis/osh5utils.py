"""Provide basic operations for H5Data"""

import osh5def
import numpy as np
import copy
import re
from functools import wraps, partial, reduce
import warnings
import osh5io
import glob
from scipy import signal


def metasl(func=None, unit=None):
    """save meta data before calling the function and restore them to output afterwards
    It has the following limitations:
        It saves the metadata of the first argument and only supports function that return one quantity
    """
    if func is None:
        return partial(metasl, unit=unit)

    @wraps(func)
    def sl(*args, **kwargs):
        saved = None
        # save meta data into a list
        if hasattr(args[0], 'meta2dict'):
            saved = args[0].meta2dict()
        # execute user function
        out = func(*args, **kwargs)
        # load saved meta data into specified positions
        try:
            if isinstance(out, osh5def.H5Data):
                out.__dict__.update(saved)
            else:
                out = osh5def.H5Data(out, **saved)
        except:
            raise TypeError('Output is not/cannot convert to H5Data')
        # Update unit if necessary
        if unit is not None:
            out.data_attrs['UNITS'] = osh5def.OSUnits(unit)
        return out
    return sl


class __NameNotFound(object):
    pass


def override_num_indexing_kw(kw, name, default_val=__NameNotFound, h5data_pos=0):
    """
        inject one more keyword to the wrapped function. If kw == name users are forced to use string indexing
        the value of kw (usually named 'axis' or 'axes') now would be replace by the value of H5Data.index_of(name). 
    :param kw: str, name of the keyword to be overridden
    :param name: str, name of the new keyword accepting string as index
    :param default_val: default value of parameter 'name'. Using mutable such as string as default value is not adviced
    :param h5data_pos: position of h5data in the args list
    :return: the decorator
    """
    def _decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            val = kwargs.pop(name, default_val)
            if val is not __NameNotFound:
                kwargs[kw] = args[h5data_pos].index_of(val)
            return f(*args, **kwargs)
        return wrapper
    return _decorator


def enhence_num_indexing_kw(kw, h5data_pos=0):
    """
        kw (usually named 'axis' or 'axes') now should be able to handle both number and string
        index (but not mixing the two)
    :param kw: str, name of the keyword to be overridden
    :param h5data_pos: position of h5data in the args list
    :return: the decorator
    """
    def _decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            val = kwargs.pop(kw, __NameNotFound)
            if val is not __NameNotFound:
                if not (isinstance(val, int) or isinstance(val[0], int)):
                    # print(kwargs[kw])
                    kwargs[kw] = args[h5data_pos].index_of(val)
                else:  # put the value back
                    # kwargs.update({kw: val})
                    kwargs[kw] = val
            return f(*args, **kwargs)
        return wrapper
    return _decorator


def metasl_map(mapping=(0, 0)):  # not well tested
    def _my_decorator(func):
        """save meta data before calling the function and restore them to output afterwards
        The input to output mapping is specified in the keyword "mapping" where (i, o) is the
        position of i-th input parameters for which meta data will be saved and the o-th output
        return values that the meta data will be restored to. Multiple save/restore should be
        written as ((i1,o1), (i2,o2), (i3,o3)) etc. The counting start from 0
        """
        @wraps(func)
        def sl(*args, **kwargs):
            # search for all specified input arguments
            saved, kl = [], []
            if isinstance(mapping[0], tuple):  # multiple save/load
                kl = sorted(list(mapping), key=lambda a: a[0])  # sort by input param. pos.
            else:
                kl = [mapping]
            if len(args) < kl[-1][0]:  # TODO(1) they can be in the kwargs
                raise Exception('Cannot find the ' + str(kl[-1][0]) + '-th argument for meta data saving')
            # save meta data into a list
            for tp in kl:
                if hasattr(args[tp[0]], 'meta2dict'):
                    saved.insert(0, (args[tp[0]].meta2dict(), tp[1]))
            if not iter(saved):
                raise ValueError('Illegal mapping parameters')
            # execute user function
            out = func(*args, **kwargs)
            # load saved meta data into specified positions
            ol, tl = [], out if isinstance(out, tuple) else (out, )
            try:
                for tp in saved:
                    if isinstance(tl[tp[1]], osh5def.H5Data):
                        ol.append(osh5def.H5Data.__dict__.update(tp[0]))
                    else:
                        aaa = tl[tp[1]]
                        ol.append(osh5def.H5Data(aaa, **tp[0]))
            except IndexError:
                raise IndexError('Output does not have ' + str(tp[1]) + ' elements')
            except:
                raise TypeError('Output[' + str(tp[1]) + '] is not/cannot convert to H5Data')
            tmp = tuple(ol) if len(ol) > 1 else ol[0] if ol else out
            return tmp
        return sl
    return _my_decorator


def stack(arr, axis=0, axesdata=None):
    """Similar to numpy.stack. Arr is the list of H5Data to be stacked. By default the newly created dimension
    will be labeled as time axis. Other meta data will be copied from the last element of arr
    """
    try:
        if not isinstance(arr[-1], osh5def.H5Data):
            raise TypeError('Input array must contain H5Data objects')
    except (TypeError, IndexError):   # not an array or an empty array, just return what ever passed in
        return arr
    md = arr[-1]
    ax = copy.deepcopy(md.axes)
    if axesdata:
        if axesdata.size != len(arr):
            raise ValueError('Number of points in axesdata is different from the new dimension to be created')
        ax.insert(axis, axesdata)
    else:  # we assume the new dimension is time
        taxis_attrs = {'UNITS': "\omega_p^{-1}", 'LONG_NAME': "time", 'NAME': "t"}
        ax.insert(axis, osh5def.DataAxis(arr[0].run_attrs['TIME'],
                                         arr[-1].run_attrs['TIME'], len(arr), attrs=taxis_attrs))
    r = np.stack(arr, axis=axis)
    return osh5def.H5Data(r, md.timestamp, md.data_attrs, md.run_attrs, axes=ax)


def combine(dir_or_filelist, prefix=None, file_slice=slice(None,), preprocess=None, axesdata=None, save=None):
    """
    stack a directory of grid data and optionally save the result to a file
    :param dir_or_filelist: name of the directory
    :param prefix: string, match file names with specific prefix
    :param file_slice: a slice that applies to the file name list, useful for skipping every other files or start
                        in the middle of the list for example
    :param preprocess: a list of callable (and their args and kwargs if any) that will act on the data before stacking
                        happens. It has to accept one H5Data objects as argument and return one H5Data objects.
                        Note that it has to be a list, not tuple or any other type, aka if isinstance(preprocess, list)
                        returns False then this parameter will be ignored entirely.
    :param axesdata: user difined axes, see stack for more detail
    :param save: name of the save file. user can also set it to true value and the output will use write_h5 defaults
    :return: combined grid data, one dimension more than the preprocessed original data
    Usage of preprocess:
    The functino list should look like:
    [(func1, arg11, arg21, ..., argn1, {'kwarg11': val11, 'kwarg21': val21, ..., 'kwargn1', valn1}),
     (func2, arg12, arg22, ..., argn2, {'kwarg12': val12, 'kwarg22': val22, ..., 'kwargn2', valn2}),
     ...,
     (func2, arg1n, arg2n, ..., argnn, {'kwarg1n': val1n, 'kwarg2n': val2n, ..., 'kwargnn', valnn})] where 
     any of the *args and/or **args can be omitted. see also __parse_func_param for limitations.
        if preprocess=[(numpy.power, 2), (numpy.average, {axis=0}), numpy.sqrt], then the data to be stacked is
        numpy.sqrt( numpy.average( numpy.power( read_h5(file_name), 2 ), axis=0 ) )
    """
    prfx = str(prefix).strip() if prefix else ''

    if isinstance(dir_or_filelist, str):
        flist = sorted(glob.glob(dir_or_filelist + '/' + prfx + '*.h5'))[file_slice]
    else:  # dir_or_filelist is a list of file names
        flist = dir_or_filelist[file_slice]
    if isinstance(preprocess, list) and preprocess:
        func_list = [__parse_func_param(item) for item in preprocess]
        tmp = [reduce(lambda x, y: y[0](x, *y[1], **y[2]), func_list, osh5io.read_h5(fn)) for fn in flist]
    else:
        tmp = [osh5io.read_h5(f) for f in flist]
    res = stack(tmp, axesdata=axesdata)
    if save:
        if not isinstance(save, str):
            save = dir_or_filelist if isinstance(dir_or_filelist, str) else './' + res.name + '.h5'
        osh5io.write_h5(res, save)
    return res


def __parse_func_param(item):
    """
    The limitation here is that
    1) (solvable by user defined functions) it can not handle a function with the following signature:
        function f has only positional arguments, i.e. f(arg1, arg2, ..., argn), and argn is expecting a dict.
        User can define a wrapper function to switch the argument positions or add a dummy argument at the end.
    2) (solvable within our notation) user wants to pass a dict as the last positional argument to the
        following function: g(arg1, arg2, ..., argn, kwarg1=val1, ..., kwargn=valn) where argn is expecting a
        dict. The problem can be avoid by adding an empty dict to the end (g, arg1, ..., argn, {}).
        It may confuse users though.
    """
    # we don't return functools.partial here because it will break the ufunc behavior
    try:
        iter(item)
        # assuming if last element is not a dictionary then it is one of the positional arguments
        if isinstance(item[-1], dict):
            args, kwargs = item[1:-1], item[-1]
        else:
            args, kwargs = item[1:], {}
    except TypeError:
        # unary function, nothing to do
        return item, (), {}
    return item[0], args, kwargs


# #----------------------------------- FFT Wrappers ----------------------------------------
# sfunc: for shifting; ffunc: for calculating frequency; ftfunc: for fft the data; uafunc: for updating axes
#
def __idle(a, *_args, **_kwargs):
    return a


def __try_update_axes(updfunc):
    def update_axes(a, idx, shape, sfunc=__idle, ffunc=__idle):
        if not hasattr(a, 'axes'):  # no axes found
            return
        try:
            iter(idx)
        except TypeError:
            idx = tuple(range(len(a.axes))) if idx is None else (idx,)
        try:
            iter(shape)
        except TypeError:
            shape = (shape,)
        # The data size can change due to the s (or n) keyword. We have to force axes update somehow.
        updfunc(a.axes, idx, shape, sfunc, ffunc=ffunc)
    return update_axes


def __update_axes_label(axes, i):
    if axes[i].attrs['NAME'] == 't' or axes[i].attrs['LONG_NAME'] == 'time' or axes[i].attrs['UNITS'].is_time():
        axes[i].attrs['LONG_NAME'] = '\omega'
        axes[i].attrs['NAME'] == 'w'
    else:
        axes[i].attrs['LONG_NAME'] = 'K(' + axes[i].attrs['LONG_NAME'] + ')'
        axes[i].attrs['NAME'] = 'k' + axes[i].attrs['NAME']
    try:
        axes[i].attrs['UNITS'] **= -1
    except TypeError:
        pass


@__try_update_axes
def _update_fft_axes(axes, idx, shape, sfunc, ffunc):
    for i in idx:
        __update_axes_label(axes, i)
        axes[i].attrs['shift'] = axes[i].min  # save lower bound. value of axes
        axes[i].ax = sfunc(ffunc(shape[i], d=axes[i].increment)) * 2 * np.pi


@__try_update_axes
def _update_ifft_axes(axes, idx,  shape, _sfunc, ffunc):
    warned = False
    for i in idx:
        try:
            xmin = axes[i].attrs['shift']
        except KeyError:
            xmin = 0
            if not warned:
                warnings.warn('Maybe doing IFFT on non-FFT data. '
                              'Make sure to use our FFT routine for forward FFT',
                              RuntimeWarning)
                warned = True
        axes[i].ax = ffunc(shape[i], d=axes[i].increment, min=xmin)
        if axes[i].attrs['LONG_NAME'] == '\omega' or axes[i].attrs['UNITS'].is_frequency():
            axes[i].attrs['LONG_NAME'] = 'time'
            axes[i].attrs['NAME'] = 't'
        else:
            axes[i].attrs['LONG_NAME'] = axes[i].attrs['LONG_NAME'][2:-1]
            axes[i].attrs['NAME'] = axes[i].attrs['NAME'][1:]
        try:
            axes[i].attrs['UNITS'] **= -1
        except TypeError:
            pass


def _get_ihfft_axis(n, d=1.0, min=0.0):
    length = 2 * np.pi / d
    return np.arange(min, length + min, length / n)[0: n//2+1]


def _get_ifft_axis(n, d=1.0, min=0.0):
    length = 2 * np.pi / d
    return np.arange(min, length + min, length / n)


def __ft_interface(ftfunc, sfunc):
    @metasl
    def ft_interface(a, s, axes, norm):
        # call fft and shift the result
        return sfunc(ftfunc(sfunc(a, axes), s, axes, norm), axes)
    return ft_interface


def __shifted_ft_gen(ftfunc, sfunc, ffunc, uafunc):
    def shifted_fft(a, s=None, axes=None, norm=None):
        shape = s if s is not None else a.shape
        o = __ft_interface(ftfunc, sfunc=sfunc)(a, s=s, axes=axes, norm=norm)
        uafunc(o, axes, shape, sfunc=sfunc, ffunc=ffunc)
        return o
    return shifted_fft


# # ========  Normal FFT  ==========
__shifted_fft = partial(__shifted_ft_gen, sfunc=np.fft.fftshift, ffunc=np.fft.fftfreq, uafunc=_update_fft_axes)
__shifted_ifft = partial(__shifted_ft_gen, sfunc=np.fft.ifftshift, ffunc=_get_ifft_axis, uafunc=_update_ifft_axes)


@enhence_num_indexing_kw('axes')
def fftn(a, s=None, axes=None, norm=None):
    return __shifted_fft(np.fft.fftn)(a, s=s, axes=axes, norm=norm)


@enhence_num_indexing_kw('axes')
def fft2(a, s=None, axes=(-2, -1), norm=None):
    return __shifted_fft(np.fft.fft2)(a, s=s, axes=axes, norm=norm)


@enhence_num_indexing_kw('axis')
def fft(a, n=None, axis=-1, norm=None):
    return __shifted_fft(np.fft.fft)(a, s=n, axes=axis, norm=norm)


@enhence_num_indexing_kw('axes')
def ifftn(a, s=None, axes=None, norm=None):
    return __shifted_ifft(np.fft.ifftn)(a, s=s, axes=axes, norm=norm)


@enhence_num_indexing_kw('axes')
def ifft2(a, s=None, axes=(-2, -1), norm=None):
    return __shifted_ifft(np.fft.ifft2)(a, s=s, axes=axes, norm=norm)


@enhence_num_indexing_kw('axis')
def ifft(a, n=None, axis=-1, norm=None):
    # if axes is None:
    #     axes = -1
    return __shifted_ifft(np.fft.ifft)(a, s=n, axes=axis, norm=norm)


# # ========  real FFT  ==========
__shifted_rfft = partial(__shifted_ft_gen, sfunc=__idle, ffunc=np.fft.rfftfreq, uafunc=_update_fft_axes)
__shifted_irfft = partial(__shifted_ft_gen, sfunc=__idle, ffunc=_get_ifft_axis, uafunc=_update_ifft_axes)


def __save_space_shape(a, s):
    if isinstance(a, osh5def.H5Data):
        shape = s if s is not None else a.shape
        a.data_attrs.setdefault('oshape', shape)


def __restore_space_shape(xdfunc):
    def get_shape(a, s, axes):
        if s is not None:
            return s
        if isinstance(a, osh5def.H5Data):
            return xdfunc(a, s, axes)
    return get_shape


@__restore_space_shape
def __rss_1d(a, _s, axes):
    return a.data_attrs['oshape'][-1] if axes is None else a.data_attrs['oshape'][axes]


@__restore_space_shape
def __rss_2d(a, _s, axes):
    return a.data_attrs['oshape'][-2:] if axes is None else tuple([a.data_attrs['oshape'][i] for i in axes])


@__restore_space_shape
def __rss_nd(a, _s, axes):
    return a.data_attrs['oshape'] if axes is None else tuple([a.data_attrs['oshape'][i] for i in axes])


@enhence_num_indexing_kw('axes')
def rfftn(a, s=None, axes=None, norm=None):
    __save_space_shape(a, s)
    return __shifted_rfft(np.fft.rfftn)(a, s=s, axes=axes, norm=norm)


@enhence_num_indexing_kw('axes')
def rfft2(a, s=None, axes=(-2, -1), norm=None):
    __save_space_shape(a, s)
    return __shifted_rfft(np.fft.rfft2)(a, s=s, axes=axes, norm=norm)


@enhence_num_indexing_kw('axis')
def rfft(a, n=None, axis=-1, norm=None):
    __save_space_shape(a, n)
    return __shifted_rfft(np.fft.rfft)(a, s=n, axes=axis, norm=norm)


@enhence_num_indexing_kw('axes')
def irfftn(a, s=None, axes=None, norm=None):
    s = __rss_nd(a, s, axes)
    return __shifted_irfft(np.fft.irfftn)(a, s=s, axes=axes, norm=norm)


@enhence_num_indexing_kw('axes')
def irfft2(a, s=None, axes=(-2, -1), norm=None):
    s = __rss_2d(a, s, axes)
    return __shifted_irfft(np.fft.irfft2)(a, s=s, axes=axes, norm=norm)


@enhence_num_indexing_kw('axis')
def irfft(a, n=None, axis=-1, norm=None):
    n = __rss_1d(a, n, axis)
    return __shifted_irfft(np.fft.irfft)(a, s=n, axes=axis, norm=norm)


# # ========  Hermitian FFT  ==========
__shifted_hfft = partial(__shifted_ft_gen, sfunc=np.fft.fftshift, ffunc=np.fft.fftfreq, uafunc=_update_fft_axes)
__shifted_ihfft = partial(__shifted_ft_gen, sfunc=__idle, ffunc=_get_ihfft_axis, uafunc=_update_ifft_axes)


@enhence_num_indexing_kw('axis')
def hfft(a, n=None, axis=-1, norm=None):
    if n is None:
        n = a.shape[-1] if axis is None else a.shape[axis]
    nn = 2*n - 1 if n % 2 else 2*n - 2
    return __shifted_hfft(np.fft.hfft)(a, s=nn, axes=axis, norm=norm)


@enhence_num_indexing_kw('axis')
def ihfft(a, n=None, axis=-1, norm=None):
    return __shifted_ihfft(np.fft.ihfft)(a, s=n, axes=axis, norm=norm)
# ----------------------------------- FFT Wrappers ----------------------------------------


# ---------------------------------- SciPy Wrappers ---------------------------------------
@enhence_num_indexing_kw('axis')
@metasl
def hilbert(x, N=None, axis=-1):
    return signal.hilbert(x, N=N, axis=axis)


@enhence_num_indexing_kw('axis')
@metasl
def hilbert2(x, N=None):
    return signal.hilbert2(x, N=N)


@enhence_num_indexing_kw('axis')
def spectrogram(h5data, axis=-1, **kwargs):
    """
    A wrapper function of scipy.signal.spectrogram
    :param h5data:
    :param kwargs:
    :param axis:
    :return: h5data with one more dimension appended
    """
    meta = h5data.meta2dict()
    k, x, Sxx = signal.spectrogram(h5data.values, fs=1/h5data.axes[axis].increment, axis=axis, **kwargs)
    meta['axes'][axis].ax = x - x[0] + h5data.axes[axis].min
    meta['axes'].insert(axis, osh5def.DataAxis(attrs=copy.deepcopy(meta['axes'][axis].attrs), data=2*np.pi*k))
    __update_axes_label(meta['axes'], axis - 1 if axis < 0 else axis)
    return osh5def.H5Data(Sxx, **meta)
# ---------------------------------- SciPy Wrappers ---------------------------------------


# ---------------------------------- NumPy Wrappers ---------------------------------------
@metasl(unit='a.u.')
def angle(x, deg=0):
    return np.angle(x, deg=deg)


@enhence_num_indexing_kw('axis')
@metasl
def unwrap(x, discont=3.141592653589793, axis=-1):
    return np.unwrap(x, discont=discont, axis=axis)


@enhence_num_indexing_kw('axis')
def diff(x, n=1, axis=-1):
    dx_2 = 0.5 * x.axes[axis].increment
    
    @metasl
    def __diff(x, n=1, axis=-1):
        return np.diff(x, n=n, axis=axis)

    r = __diff(x, n=n, axis=axis)
    r.axes[axis] = osh5def.DataAxis(axis_min=x.axes[axis].min+n*dx_2, axis_max=x.axes[axis].max-n*dx_2, 
                                    axis_npoints=x.axes[axis].size-n, attrs=copy.deepcopy(x.axes[axis].attrs))
    return r

# ---------------------------------- NumPy Wrappers ---------------------------------------

def field_decompose(fldarr, ffted=True, idim=None, finalize=None, outquants=('L', 't')):
    """decompose a vector field into transverse and longitudinal direction
    fldarr: list of field components in the order of x, y, z
    ffted: If the input fields have been Fourier transformed
    finalize: what function to call after all transforms,
        for example finalize=abs will be converted the fields to amplitude
    idim: inverse fourier transform in idim direction(s)
    outquonts: output quantities: default=('L','t')
        'L': total amplitude square of longitudinal components
        'T': total amplitude square of transverse components
        't' or 't1', 't2', ...: transverse components, 't' means all transverse components
        'l' or 'l1', 'l2', ...: longitudinal components, 'l' means all longitudinal components
    return: list of field components in the following order (if some are not requested they will be simply omitted):
        ['L', 'T', 't', 'l']
    """
    dim = len(fldarr)
    if dim != fldarr[0].ndim:
        raise IndexError('Not enough field components for decomposition')
    if fldarr[0].ndim == 1:
        return copy.deepcopy(fldarr)
    if not finalize:
        finalize = __idle

    def wrap_up(data):
        if idim:
            return ifftn(data, axes=idim)
        else:
            return data

    def rename(fld, name, longname):
        if isinstance(fld, osh5def.H5Data):
            # replace numbers in the string
            fld.name = re.sub("\d+", fld.name, name)
            fld.data_attrs['NAME'] = re.sub("\d+", name, fld.data_attrs.get('NAME', fld.name))
            fld.data_attrs['LONG_NAME'] = re.sub("\d+", longname, fld.data_attrs.get('LONG_NAME', ''))
        return fld

    if ffted:
        fftfld = [copy.deepcopy(fi) for fi in fldarr]
    else:
        fftfld = [fftn(fi) for fi in fldarr]
    kv = np.meshgrid(*reversed([x.ax for x in fftfld[0].axes]), sparse=True)
    k2 = np.sum(ki**2 for ki in kv)  # |k|^2
    k2[k2 == 0.0] = float('inf')
    kdotfld = np.divide(np.sum(np.multiply(ki, fi) for ki, fi in zip(kv, fftfld)), k2)
    fL, fT, ft, fl = 0, 0, [], []
    for i, fi in enumerate(fftfld):
        tmp = kdotfld * kv[i]
        if 't' in outquants or 't' + str(i + 1) in outquants:
            ft.append((finalize(wrap_up(fftfld[i] - tmp)), '{t' + str(i + 1) + '}'))
            if 'T' in outquants:
                fT += np.abs(ft[-1][0])**2
        elif 'T' in outquants:
            fT += np.abs(wrap_up(fftfld[i] - tmp))**2

        if 'l' in outquants or 'l'+str(i+1) in outquants:
            fl.append((finalize(wrap_up(tmp)), '{l'+str(i+1) + '}'))
            if 'L' in outquants:
                fL += np.abs(fl[-1][0])**2
        elif 'L' in outquants:
            fL += np.abs(wrap_up(tmp))**2

    res = []
    if not isinstance(fL, int):
        res.append(rename(fL, 'L', 'L^2'))
    if not isinstance(fT, int):
        res.append(rename(fT, 'T', 'T^2'))
    if ft:
        for fi in ft:
            res.append(rename(fi[0], fi[1], fi[1]))
    if fl:
        for fi in fl:
            res.append(rename(fi[0], fi[1], fi[1]))
    return tuple(res)


# modified from SciPy cookbook
def rebin(a, fac):
    """
    rebin ndarray or H5Data into a smaller ndarray or H5Data of the same rank whose dimensions
    are factors of the original dimensions. If fac in some dimension is not whole divided by
    a.shape, the residual is trimmed from the last part of array.
    example usages:
     a=rand(6,4); b=rebin(a, fac=[3,2])
     a=rand(10); b=rebin(a, fac=[3])
    """
    index = [slice(0, u - u % fac[i]) if u % fac[i] else slice(0, u) for i, u in enumerate(a.shape)]
    a = a[index]
    # update axes first
    if isinstance(a, osh5def.H5Data):
        for i, x in enumerate(a.axes):
            x.ax = x.ax[::fac[i]]

    @metasl
    def __rebin(h, fac):
        # have to convert to ndarray otherwise sum will fail
        h = h.view(np.ndarray)
        shape = h.shape
        lenShape = len(shape)
        newshape = np.floor_divide(shape, np.asarray(fac))
        evList = ['h.reshape('] + \
                 ['newshape[%d],fac[%d],'%(i,i) for i in range(lenShape)] + \
                 [')'] + ['.sum(%d)'%(i+1) for i in range(lenShape)] + \
                 ['/fac[%d]'%i for i in range(lenShape)]
        return eval(''.join(evList))

    return __rebin(a, fac)


def log_Gabor_Filter_2d(w, w0, s0):
    return np.exp( - np.log(w/w0)**2 / (2 * np.log(s0)**2) )


def monogenic_signal(data, *args, filter_func=log_Gabor_Filter_2d, ffted=True, ifft=True):
    """
    Get the monogenic signal of 2D data. This implementation is better suited for intrisically 1D signals. 
    read the following articles for more details:
    [1] M. Felsberg and G. Sommer, IEEE Trans. Signal Process. 49, 3136 (2001).
    [2] C. P. Bridge, ArXiv:1703.09199 (2017).
    :param data: 2D H5Data
    :param *args: arguments which will be passthrough to the Filter function
    :param filter_func: filter function
    :param ffted: Set to True if the ft_data is the Fourier transform, default is False
    :param ifft: if True then inverse Fourier transform back to real space, default is True
    :return: mongenic signal as a tuple (f, f_R), where f is the filtered orginal signal and f_R is the
             Riesz transform of the filtered signal
    """
    ft_data = fft2(data) if not ffted else data
    w = np.meshgrid(*reversed([x.ax for x in ft_data.axes]), sparse=True)
    wamp = np.sqrt(np.sum(wi**2 for wi in w))
    origin = np.where(wamp==0)
    wamp[origin] = 1.
    flt = filter_func(wamp, *args)
    flt[origin] = 0.
    goc = ft_data * (w[0] * flt * 1j - w[1] * flt) / wamp
    ge = flt * ft_data
    if ifft:
        goc = ifft2(goc)
        ge = ifft2(ge)
    return np.real(ge), goc


def monogenic_local_phase(monogenic_signal, **kwargs):
    """
    get the local phase of a 2D monogenic signal
    :param monogenic_signal: the output of the monogenic_signal function
    :param unwrap: if True then unwrap the output angle, default is False
    :param kwargs: keyword arguments that will be passthrough to the unwrap function
    :return: local phase in h5Data format
    """
    theta = np.arctan2( np.abs(monogenic_signal[1])*np.sign(np.real(monogenic_signal[1])), monogenic_signal[0])
    unwrap_output = kwargs.pop('unwrap', False)
    if unwrap_output:
        return unwrap(theta, **kwargs)
    theta.data_attrs['UNITS'] = osh5def.OSUnits('a.u.')
    return theta


def monogenic_local_orientation(monogenic_signal):
    """
    get the local orientation of a 2D monogenic signal
    :param monogenic_signal: the output (or the second elements of the output) of the monogenic_signal function
    """
    if isinstance(monogenic_signal, (list, tuple)):
        return np.arctan(np.imag(monogenic_signal[1])/np.real(monogenic_signal[1]))
    else:
        return np.arctan(np.imag(monogenic_signal)/np.real(monogenic_signal))


def monogenic_local_amplitude(monogenic_signal):
    """
    get the local orientation of a 2D monogenic signal
    :param monogenic_signal: the output of the monogenic_signal function
    """
    return np.sqrt(monogenic_signal[0]**2 + np.abs(monogenic_signal[1])**2)


if __name__ == '__main__':
    fn = 'n0-123456.h5'
    d = osh5io.read_h5(fn)
    # d = subrange(d, ((0, 35.5), (0, 166)))
    # d = np.ones((7, 20))
    # d = rebin(d, fac=[3, 3])
    # d = d.subrange(bound=[None, (None, 23., -10.)])
    d.set_value(bound=(None, (None, 140., 10.)), val=2., inverse_select=True, method=np.multiply)
    print(repr(d.view(np.ndarray)))
    c = hfft(d)
    print('c = ', repr(c))
    b = ihfft(c)
    print('b is d? ', b is d)
    diff = d - b
    print('b - d = ', diff.view(np.ndarray))
    print(repr(b))
    b = np.sqrt(b)
    # print(repr(b))

