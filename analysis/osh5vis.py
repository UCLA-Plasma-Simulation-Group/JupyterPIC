# import osh5def
import matplotlib.pyplot as plt
import numpy as np
# try:
#     import osh5gui
#     gui_fname = osh5gui.gui_fname
# except ImportError:
#     print('Fail to import GUI routines. Check your PyQT installation')

def time_format(time=0.0, unit=None, convert_tunit=False, wavelength=0.351, **kwargs):
    if convert_tunit:
        t = wavelength * 5.31e-4 * time 
        unit = ' ps'
    else:
        t = time
    tmp = '$t = ' + "{:.2f}".format(t)
    if unit:
        tmp += '$ [$' + str(unit) + '$]'
    return tmp


def default_title(h5data, show_time=True, **kwargs):
    tmp = tex(h5data.data_attrs['LONG_NAME'])
    if show_time and not h5data.has_axis('t'):
        try:
            tmp += ', ' + time_format(h5data.run_attrs['TIME'][0], h5data.run_attrs['TIME UNITS'], **kwargs)
        except:  # most likely we don't have 'TIME' or 'TIME UNITS' in run_attrs
            pass
    return tmp


def tex(s):
    return '$' + s + '$' if s else ''


def axis_format(name=None, unit=None):
    s = '$' + str(name) + '$' if name else ''
    if unit:
        s += ' [$' + str(unit) + '$]'
    return s


def osplot(h5data, *args, **kwpassthrough):
    if h5data.ndim == 1:
        plot_object = osplot1d(h5data, *args, **kwpassthrough)
    elif h5data.ndim == 2:
        plot_object = osimshow(h5data, *args, **kwpassthrough)
    else:
        plot_object = None
    return plot_object


def __osplot1d(func, h5data, xlabel=None, ylabel=None, xlim=None, ylim=None, title=None, ax=None,
               convert_tunit=False, convert_xaxis=False, wavelength=0.351, *args, **kwpassthrough):
    if convert_xaxis:
        xaxis, xunit = h5data.axes[0].to_phys_unit()
    else:
        xaxis, xunit = h5data.axes[0], h5data.axes[0].attrs['UNITS']
    plot_object = func(xaxis, h5data.view(np.ndarray), *args, **kwpassthrough)
    if ax is not None:
        set_xlim, set_ylim, set_xlabel, set_ylabel, set_title = \
            ax.set_xlim, ax.set_ylim, ax.set_xlabel, ax.set_ylabel, ax.set_title
    else:
        set_xlim, set_ylim, set_xlabel, set_ylabel, set_title = \
            plt.xlim, plt.ylim, plt.xlabel, plt.ylabel, plt.title
    if xlabel is None:
        xlabel = axis_format(h5data.axes[0].attrs['LONG_NAME'], xunit)
    if ylabel is None:
        ylabel = axis_format(h5data.data_attrs['LONG_NAME'], str(h5data.data_attrs['UNITS']))
    if xlim is not None:
        set_xlim(xlim)
    if ylim is not None:
        set_ylim(ylim)
    set_xlabel(xlabel)
    set_ylabel(ylabel)
    if title is None:
        title = default_title(h5data, convert_tunit=convert_tunit, wavelength=wavelength)
    set_title(title)
    return plot_object


def osplot1d(h5data, *args, ax=None, **kwpassthrough):
    plot = plt.plot if ax is None else ax.plot
    return __osplot1d(plot, h5data, *args, **kwpassthrough)


def ossemilogx(h5data, *args, ax=None, **kwpassthrough):
    semilogx = plt.semilogx if ax is None else ax.semilogx
    return __osplot1d(semilogx, h5data, *args, **kwpassthrough)


def ossemilogy(h5data, *args, ax=None, **kwpassthrough):
    semilogy = plt.semilogy if ax is None else ax.semilogy
    return __osplot1d(semilogy, h5data, *args, **kwpassthrough)


def osloglog(h5data, *args, ax=None, **kwpassthrough):
    loglog = plt.loglog if ax is None else ax.loglog
    return __osplot1d(loglog, h5data, *args, **kwpassthrough)


def __osplot2d(func, h5data, *args, xlabel=None, ylabel=None, cblabel=None, title=None, xlim=None, ylim=None, clim=None,
               colorbar=True, ax=None, im=None, cb=None, convert_xaxis=False, convert_yaxis=False, fig=None,
               convert_tunit=False, wavelength=0.351, **kwpassthrough):
    if convert_xaxis:
        axis = h5data.axes[1].to_phys_unit()
        extx = axis[0].min(), axis[0].max()
        xunit = axis[1]
    else:
        extx = h5data.axes[1].min, h5data.axes[1].max
        xunit = h5data.axes[1].attrs['UNITS']

    if convert_yaxis:
        axis = h5data.axes[0].to_phys_unit()
        exty = axis[0].min(), axis[0].max()
        yunit = axis[1]
    else:
        exty = h5data.axes[0].min, h5data.axes[0].max
        yunit = h5data.axes[0].attrs['UNITS']

    extent_stuff = [extx[0], extx[1], exty[0], exty[1]]
    plot_object = func(h5data.view(np.ndarray), *args, extent=extent_stuff, **kwpassthrough)
    if ax is None:
        set_xlim, set_ylim, set_xlabel, set_ylabel, set_title = \
            plt.xlim, plt.ylim, plt.xlabel, plt.ylabel, plt.title
    else:
        set_xlim, set_ylim, set_xlabel, set_ylabel, set_title = \
            ax.set_xlim, ax.set_ylim, ax.set_xlabel, ax.set_ylabel, ax.set_title

    if xlim is not None:
        set_xlim(xlim)
    if ylim is not None:
        set_ylim(ylim)
    if xlabel is None:
        xlabel = axis_format(h5data.axes[1].attrs['LONG_NAME'], xunit)
    if ylabel is None:
        ylabel = axis_format(h5data.axes[0].attrs['LONG_NAME'], yunit)
    if title is None:
        title = default_title(h5data, convert_tunit=convert_tunit, wavelength=wavelength)
    set_xlabel(xlabel)
    set_ylabel(ylabel)
    set_title(title)

    if clim is not None:
        plot_object.set_clim(clim)

    if colorbar:
        if not cb:
            cb = plt.colorbar(plot_object) if fig is None else fig.colorbar(plot_object)
        if cblabel is None:
            cb.set_label(h5data.data_attrs['UNITS'].tex())
        else:
            cb.set_label(cblabel)
        return plot_object, cb
    return plot_object, None


def osimshow(h5data, *args, ax=None, cb=None, aspect=None, **kwpassthrough):
    imshow = ax.imshow if ax is not None else plt.imshow
    asp = 'auto' if aspect is None else aspect
    return __osplot2d(imshow, h5data, *args, cb=cb, aspect=asp, origin='lower', **kwpassthrough)


def oscontour(h5data, *args, ax=None, cb=None, **kwpassthrough):
    contour = ax.contour if ax is not None else plt.contour
    return __osplot2d(contour, h5data, *args, cb=cb, **kwpassthrough)


def oscontourf(h5data, *args, ax=None, cb=None, **kwpassthrough):
    contourf = ax.contourf if ax is not None else plt.contourf
    return __osplot2d(contourf, h5data, *args, cb=cb, **kwpassthrough)


def new_fig(h5data, *args, figsize=None, dpi=None, facecolor=None, edgecolor=None, linewidth=0.0, frameon=None,
            tight_layout=None, **kwpassthrough):
    plt.figure(figsize=figsize, dpi=dpi, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, frameon=frameon,
               tight_layout=tight_layout)
    osplot(h5data, *args, **kwpassthrough)
    plt.show()

