from __future__ import print_function
from ipywidgets import interact, Layout, Output
import ipywidgets as widgets
from IPython.display import display

import numpy as np

import osh5vis
import osh5io
import glob
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, PowerNorm, SymLogNorm
import threading


print("Importing osh5visipy. Please use `%matplotlib notebook' in your jupyter/ipython notebook")


def osimshow_w(data, *args, show=True, **kwargs):
    """
    2D plot with widgets
    :param data: 2D H5Data
    :param args: arguments passed to 2d plotting widgets. reserved for future use
    :param show: whether to show the widgets
    :param kwargs: keyword arguments passed to 2d plotting widgets. reserved for future use
    :return: if show == True return None otherwise return a list of widgets
    """
    wl = Generic2DPlotCtrl(data, *args, **kwargs).widgets_list
    if show:
        display(*wl)
    else:
        return wl


def slicer_w(data, *args, show=True, slider_only=False, **kwargs):
    """
    A slider for 3D data
    :param data: 3D H5Data or directory name (a string)
    :param args: arguments passed to plotting widgets. reserved for future use
    :param show: whether to show the widgets
    :param slider_only: if True only show the slider otherwise show also other plot control (aka 'the tab')
    :param kwargs: keyword arguments passed to 2d plotting widgets. reserved for future use
    :return: whatever widgets that are not shown
    """
    if isinstance(data, str):
        wl = DirSlicer(data, *args, **kwargs).widgets_list
        tab, slider = wl[0], widgets.HBox(wl[1:-2])
    else:
        wl = Slicer(data, *args, **kwargs).widgets_list
        tab, slider = wl[0], widgets.HBox(wl[1:-2])
    if show:
        if slider_only:
            display(slider, widgets.VBox(wl[-2:]))
            return tab
        else:
            display(tab, slider, widgets.VBox(wl[-2:]))
    else:
        return wl


def animation_w(data, *args, **kwargs):
    wl = Animation(data, *args, **kwargs).widgets_list
    display(widgets.VBox([wl[0], widgets.HBox(wl[1:4]), widgets.HBox(wl[4:-2]), widgets.VBox(wl[-2:])]))


class Generic2DPlotCtrl(object):
    tab_contents = ['Data', 'Labels', 'Axes', 'Lineout', 'Colormaps']
    eps = 1e-40
    colormaps_available = sorted(c for c in plt.colormaps() if not c.endswith("_r"))

    def __init__(self, data, slcs=(slice(None, ), ), title=None, norm=None, fig_handle=None,
                 time_in_title=True, **kwargs):

        self._data, self._slcs, self.im_xlt, self.time_in_title = data, slcs, None, time_in_title
        # # # -------------------- Tab0 --------------------------
        items_layout = Layout(flex='1 1 auto', width='auto')
        # normalization
        # general parameters: vmin, vmax, clip
        self.if_vmin_auto = widgets.Checkbox(value=True, description='Auto', layout=items_layout)
        self.if_vmax_auto = widgets.Checkbox(value=True, description='Auto', layout=items_layout)
        self.vmin_wgt = widgets.FloatText(value=np.min(data), description='vmin:', continuous_update=False,
                                          disabled=self.if_vmin_auto.value, layout=items_layout)
        self.vlogmin_wgt = widgets.FloatText(value=self.eps, description='vmin:', continuous_update=False,
                                             disabled=self.if_vmin_auto.value, layout=items_layout)
        self.vmax_wgt = widgets.FloatText(value=np.max(data), description='vmax:', continuous_update=False,
                                          disabled=self.if_vmin_auto.value, layout=items_layout)
        self.if_clip_cm = widgets.Checkbox(value=True, description='Clip', layout=items_layout)
        # PowerNorm specific
        self.gamma = widgets.FloatText(value=1, description='gamma:', continuous_update=False, layout=items_layout)
        # SymLogNorm specific
        self.linthresh = widgets.FloatText(value=self.eps, description='linthresh:', continuous_update=False,
                                           layout=items_layout)
        self.linscale = widgets.FloatText(value=1.0, description='linscale:', continuous_update=False,
                                          layout=items_layout)

        # build the widgets tuple
        ln_wgt = (LogNorm, widgets.VBox([widgets.HBox([self.vlogmin_wgt, self.if_vmin_auto]),
                                         widgets.HBox([self.vmax_wgt, self.if_vmax_auto]), self.if_clip_cm]))
        n_wgt = (Normalize, widgets.VBox([widgets.HBox([self.vmin_wgt, self.if_vmin_auto]),
                                          widgets.HBox([self.vmax_wgt, self.if_vmax_auto]), self.if_clip_cm]))
        pn_wgt = (PowerNorm, widgets.VBox([widgets.HBox([self.vmin_wgt, self.if_vmin_auto]),
                                           widgets.HBox([self.vmax_wgt, self.if_vmax_auto]), self.if_clip_cm,
                                           self.gamma]))
        sln_wgt = (SymLogNorm, widgets.VBox(
            [widgets.HBox([self.vmin_wgt, self.if_vmin_auto]),
             widgets.HBox([self.vmax_wgt, self.if_vmax_auto]), self.if_clip_cm, self.linthresh, self.linscale]))

        # find out default value for norm_selector
        norm_avail = {'Log': ln_wgt, 'Normalize': n_wgt, 'Power': pn_wgt, 'SymLog': sln_wgt}
        self.norm_selector = widgets.Dropdown(options=norm_avail,
                                              value=norm_avail.get(norm, n_wgt), description='Normalization:')
        # additional care for LorNorm()
        self.__handle_lognorm()
        # re-plot button
        self.norm_btn_wgt = widgets.Button(description='Apply', disabled=False, tooltip='set colormap', icon='refresh')
        tab0 = self.__get_tab0()

        # # # -------------------- Tab1 --------------------------
        # title
        if not title:
            title = osh5vis.default_title(data, show_time=self.time_in_title)
        self.if_reset_title = widgets.Checkbox(value=True, description='Auto')
        self.title = widgets.Text(value=title, placeholder='data', continuous_update=False,
                                  description='Title:', disabled=self.if_reset_title.value)
        # x label
        self.if_reset_xlabel = widgets.Checkbox(value=True, description='Auto')
        self.xlabel = widgets.Text(value=osh5vis.axis_format(data.axes[1].long_name, data.axes[1].units),
                                   placeholder='x', continuous_update=False,
                                   description='X label:', disabled=self.if_reset_xlabel.value)
        # y label
        self.if_reset_ylabel = widgets.Checkbox(value=True, description='Auto')
        self.ylabel = widgets.Text(value=osh5vis.axis_format(data.axes[0].long_name, data.axes[0].units),
                                   placeholder='y', continuous_update=False,
                                   description='Y label:', disabled=self.if_reset_ylabel.value)
        # colorbar
        self.if_reset_cbar = widgets.Checkbox(value=True, description='Auto')
        self.cbar = widgets.Text(value=data.units.tex(), placeholder='a.u.', continuous_update=False,
                                 description='Colorbar:', disabled=self.if_reset_cbar.value)

        tab1 = widgets.VBox([widgets.HBox([self.title, self.if_reset_title]),
                             widgets.HBox([self.xlabel, self.if_reset_xlabel]),
                             widgets.HBox([self.ylabel, self.if_reset_ylabel]),
                             widgets.HBox([self.cbar, self.if_reset_cbar])])

        # # # -------------------- Tab2 --------------------------
        self.setting_instructions = widgets.Label(value="Enter invalid value to reset", layout=items_layout)
        self.apply_range_btn = widgets.Button(description='Apply', disabled=False, tooltip='set range', icon='refresh')
        self.axis_lim_wgt = widgets.HBox([self.setting_instructions, self.apply_range_btn])
        # x axis
        self.x_min_wgt = widgets.FloatText(value=self._data.axes[1].min, description='xmin:', continuous_update=False,
                                           layout=items_layout)
        self.x_max_wgt = widgets.FloatText(value=self._data.axes[1].max, description='xmax:', continuous_update=False,
                                           layout=items_layout)
        self.x_step_wgt = widgets.FloatText(value=self._data.axes[1].increment, continuous_update=False,
                                            description='$\Delta x$:', layout=items_layout)
        self.xaxis_lim_wgt = widgets.HBox([self.x_min_wgt, self.x_max_wgt, self.x_step_wgt])
        # y axis
        self.y_min_wgt = widgets.FloatText(value=self._data.axes[0].min, description='ymin:', continuous_update=False,
                                           layout=items_layout)
        self.y_max_wgt = widgets.FloatText(value=self._data.axes[0].max, description='ymax:', continuous_update=False,
                                           layout=items_layout)
        self.y_step_wgt = widgets.FloatText(value=self._data.axes[0].increment, continuous_update=False,
                                            description='$\Delta y$:', layout=items_layout)
        self.yaxis_lim_wgt = widgets.HBox([self.y_min_wgt, self.y_max_wgt, self.y_step_wgt])
        tab2 = widgets.VBox([self.axis_lim_wgt, self.xaxis_lim_wgt, self.yaxis_lim_wgt])

        # # # -------------------- Tab3 --------------------------
        tab3 = self.if_lineout_wgt = widgets.Checkbox(value=False, description='X/Y Lineouts (incomplete feature)',
                                                      layout=items_layout)

        # # # -------------------- Tab4 --------------------------
        user_cmap = kwargs.pop('cmap', 'jet')
        self.cmap_selector = widgets.Dropdown(options=self.colormaps_available, value=user_cmap,
                                              description='Colormap:')
        self.cmap_reverse = widgets.Checkbox(value=False, description='Reverse', layout=items_layout)
        tab4 = widgets.HBox([self.cmap_selector, self.cmap_reverse])

        # construct the tab
        self.tab = widgets.Tab()
        self.tab.children = [tab0, tab1, tab2, tab3, tab4]
        [self.tab.set_title(i, tt) for i, tt in enumerate(self.tab_contents)]
        # display(self.tab)

        # link and activate the widgets
        self.if_reset_title.observe(self.__update_title, 'value')
        self.if_reset_xlabel.observe(self.__update_xlabel, 'value')
        self.if_reset_ylabel.observe(self.__update_ylabel, 'value')
        self.if_reset_cbar.observe(self.__update_cbar, 'value')
        self.norm_btn_wgt.on_click(self.update_norm)
        self.if_vmin_auto.observe(self.__update_vmin, 'value')
        self.if_vmax_auto.observe(self.__update_vmax, 'value')
        self.norm_selector.observe(self.__update_norm_wgt, 'value')
        self.cmap_selector.observe(self.update_cmap, 'value')
        self.cmap_reverse.observe(self.update_cmap, 'value')
        self.title.observe(self.update_title, 'value')
        self.xlabel.observe(self.update_xlabel, 'value')
        self.ylabel.observe(self.update_ylabel, 'value')
        self.cbar.observe(self.update_cbar, 'value')
        self.y_max_wgt.observe(self.__update_y_max, 'value')
        self.y_min_wgt.observe(self.__update_y_min, 'value')
        self.x_max_wgt.observe(self.__update_x_max, 'value')
        self.x_min_wgt.observe(self.__update_x_min, 'value')
        self.x_step_wgt.observe(self.__update_delta_x, 'value')
        self.y_step_wgt.observe(self.__update_delta_y, 'value')
        self.apply_range_btn.on_click(self.update_plot_area)
        self.if_lineout_wgt.observe(self.toggle_lineout, 'value')

        # plotting and then setting normalization colors
        self.out = Output()
        self.out_main = Output()
        self.observer_thrd, self.cb = None, None
        self.fig = plt.figure() if fig_handle is None else fig_handle
        self.ax = self.fig.add_subplot(111)
        with self.out_main:
            self.im, self.cb = self.plot_data()
            display(self.fig)

    @property
    def self(self):
        return self

    @property
    def widgets_list(self):
        return self.tab, self.out_main, self.out

    @property
    def widget(self):
        return widgets.VBox([self.tab, self.out_main, self.out])

    def update_data(self, data, slcs):
        self._data, self._slcs = data, slcs
        self.__update_title()
        self.__update_xlabel()
        self.__update_ylabel()

    def reset_plot_area(self):
        self.y_min_wgt.value, self.y_max_wgt.value, self.y_step_wgt.value = \
            self._data.axes[0].min, self._data.axes[0].max, self._data.axes[0].increment
        self.x_min_wgt.value, self.x_max_wgt.value, self.x_step_wgt.value = \
            self._data.axes[1].min, self._data.axes[1].max, self._data.axes[1].increment

    def redraw(self, data):
        """if the size of the data is the same we can just redraw part of figure"""
        self._data = data
        self.im.set_data(self.__pp(data[self._slcs]))
        self.fig.canvas.draw()

    def update_title(self, change):
        self.ax.axes.set_title(change['new'])

    def update_xlabel(self, change):
        self.ax.axes.xaxis.set_label_text(change['new'])

    def update_ylabel(self, change):
        self.ax.axes.yaxis.set_label_text(change['new'])

    def update_cbar(self, change):
        self.im.colorbar.set_label(change['new'])

    def update_cmap(self, _change):
        self.im.set_cmap(self.cmap_selector.value if not self.cmap_reverse.value else self.cmap_selector.value + '_r')

    def update_plot_area(self, *_):
        bnd = [(self.y_min_wgt.value, self.y_max_wgt.value, self.y_step_wgt.value),
               (self.x_min_wgt.value, self.x_max_wgt.value, self.x_step_wgt.value)]
        self._slcs = tuple(slice(*self._data.get_index_slice(self._data.axes[i], bd)) for i, bd in enumerate(bnd))
        self.cb.remove()
        self.im, self.cb = self.plot_data(im=self.im)
        # dirty hack
        if self.norm_selector.value[0] == LogNorm:
            self.cb.set_norm(LogNorm())

    def refresh_tab_wgt(self, update_list):
        """
        the tab.children is a tuple so we have to reconstruct the whole tab widget when 
        addition/deletion of children widgets happens
        """
        tmp = self.tab.children
        newtab = [tmp[i] if not t else t for i, t in enumerate(update_list)]
        self.tab.children = tuple(newtab)

    def plot_data(self, **passthrough):
        return osh5vis.osimshow(self.__pp(self._data[self._slcs]), cmap=self.cmap_selector.value,
                                norm=self.norm_selector.value[0](**self.__get_norm()), title=self.title.value,
                                xlabel=self.xlabel.value, ylabel=self.ylabel.value, cblabel=self.cbar.value,
                                ax=self.ax, fig=self.fig, **passthrough)

    def __get_tab0(self):
        return widgets.HBox([widgets.VBox([self.norm_selector, self.norm_selector.value[1]]), self.norm_btn_wgt])

    @staticmethod
    def _idle(data):
        return data

    def __new_lineout_plot(self):
        with self.out:
            self.fig, self.outline_ax = plt.subplots(1, 2, figsize=(3, 2))
            ldata = self._data[self._slcs]
            osh5vis.osplot1d(ldata[ldata.shape[0] // 2, :], ax=self.outline_ax[0])
            osh5vis.osplot1d(ldata[:, ldata.shape[0] // 2], ax=self.outline_ax[1], title='')
            plt.suptitle('X Axis={:.2f}'.format(ldata.axes[0][ldata.shape[0] // 2]) +
                         ', Y Axis={:.2f}'.format(ldata.axes[1][ldata.shape[1] // 2]))
            plt.title('Y lineout')
            plt.show()

    def toggle_lineout(self, change):
        if change['new']:
            # start a new thread so the interaction with original figure won't be blocked
            self.observer_thrd = threading.Thread(target=self.__new_lineout_plot)
            self.observer_thrd.daemon = True
            self.observer_thrd.start()
            # display(self.out)
        else:
            self.observer_thrd.join()  # kill the thread
            Output.clear_output(self.out)

    def __handle_lognorm(self):
        if self.norm_selector.value[0] == LogNorm:
            self.__pp = np.abs
            self.vmax_wgt.value = np.max(np.abs(self._data))
            self.__assgin_valid_vmin()
        else:
            self.vmax_wgt.value = np.max(self._data)
            self.__assgin_valid_vmin()
            self.__pp = self._idle

    def __update_norm_wgt(self, _change):
        """update tab1 (second tab) only and prepare _log_data if necessary"""
        tmp = [None] * len(self.tab_contents)
        tmp[0] = self.__get_tab0()
        self.refresh_tab_wgt(tmp)
        self.__handle_lognorm()

    def update_norm(self, *args):
        self.cb.remove()
        self.im.remove()
        self.im, self.cb = self.plot_data(im=self.im)
        #  update norm
        self.set_norm(*args)

    def __get_norm(self):
        vmin = None if self.if_vmin_auto.value else self.norm_selector.value[1].children[0].children[0].value
        vmax = None if self.if_vmax_auto.value else self.vmax_wgt.value
        param = {'vmin': vmin, 'vmax': vmax, 'clip': self.if_clip_cm.value}
        if self.norm_selector.value[0] == PowerNorm:
            param['gamma'] = self.gamma.value
        if self.norm_selector.value[0] == SymLogNorm:
            param['linthresh'] = self.linthresh.value
            param['linscale'] = self.linscale.value
        return param

    def set_norm(self, *_):
        param = self.__get_norm()
        self.cb.set_norm(self.norm_selector.value[0](**param))

    def __assgin_valid_vmin(self, v=None):
        # if it is log scale
        if self.norm_selector.value[0] == LogNorm:
            self.vlogmin_wgt.value = self.eps if v is None or v < self.eps else v
        else:
            self.vmin_wgt.value = np.min(self._data) if v is None else v

    def __update_vmin(self, _change):
        if self.if_vmin_auto.value:
            self.__assgin_valid_vmin()
            self.vmin_wgt.disabled = True
            self.vlogmin_wgt.disabled = True
        else:
            self.vmin_wgt.disabled = False
            self.vlogmin_wgt.disabled = False

    def __update_vmax(self, _change):
        if self.if_vmax_auto.value:
            self.vmax_wgt.value = np.max(self._data)
            self.vmax_wgt.disabled = True
        else:
            self.vmax_wgt.disabled = False

    def __update_title(self, *_):
        if self.if_reset_title.value:
            self.title.value = osh5vis.default_title(self._data, show_time=self.time_in_title)
            self.title.disabled = True
        else:
            self.title.disabled = False

    def __update_xlabel(self, *_):
        if self.if_reset_xlabel.value:
            self.xlabel.value = osh5vis.axis_format(self._data.axes[1].long_name, self._data.axes[1].units)
            self.xlabel.disabled = True
        else:
            self.xlabel.disabled = False

    def __update_ylabel(self, *_):
        if self.if_reset_ylabel.value:
            self.ylabel.value = osh5vis.axis_format(self._data.axes[0].long_name, self._data.axes[0].units)
            self.ylabel.disabled = True
        else:
            self.ylabel.disabled = False

    def __update_cbar(self, *_):
        if self.if_reset_cbar.value:
            self.cbar.value = self._data.units.tex()
            self.cbar.disabled = True
        else:
            self.cbar.disabled = False

    def __update_y_max(self, change):
        self.y_max_wgt.value = change['new'] if self.y_min_wgt.value < change['new'] < self._data.axes[0].max \
            else self._data.axes[0].max

    def __update_x_max(self, change):
        self.x_max_wgt.value = change['new'] if self.x_min_wgt.value < change['new'] < self._data.axes[1].max \
            else self._data.axes[1].max

    def __update_y_min(self, change):
        self.y_min_wgt.value = change['new'] if self._data.axes[0].min < change['new'] < self.y_max_wgt.value \
            else self._data.axes[0].min

    def __update_x_min(self, change):
        self.x_min_wgt.value = change['new'] if self._data.axes[1].min < change['new'] < self.x_max_wgt.value \
            else self._data.axes[1].min

    def __update_delta_y(self, change):
        if not (0 < round(change['new'] / self._data.axes[0].increment) <= self._data[self._slcs].shape[0]):
            self.y_step_wgt.value = self._data.axes[0].increment

    def __update_delta_x(self, change):
        if not (0 < round(change['new'] / self._data.axes[1].increment) <= self._data[self._slcs].shape[1]):
            self.x_step_wgt.value = self._data.axes[1].increment


class Slicer(Generic2DPlotCtrl):
    def __init__(self, data, d=0, **extra_kwargs):
        self.x, self.comp, self.data = data.shape[d] // 2, d, data
        self.slcs = self.__get_slice(d)
        self.axis_pos = widgets.FloatText(value=data.axes[self.comp][self.x],
                                          description=self.__axis_descr_format(), continuous_update=False)
        self.index_slider = widgets.IntSlider(min=0, max=self.data.shape[self.comp] - 1, step=1, description='index:',
                                              value=self.data.shape[self.comp] // 2, continuous_update=False)

        self.axis_selector = widgets.Dropdown(options=list(range(data.ndim)), value=self.comp, description='axis:')
        self.axis_selector.observe(self.switch_slice_direction, 'value')
        self.index_slider.observe(self.update_slice, 'value')
        self.axis_pos.observe(self.__update_index_slider, 'value')

        super(Slicer, self).__init__(data[self.slcs], slcs=[i for i in self.slcs if not isinstance(i, int)],
                                     time_in_title=not data.has_axis('t'), **extra_kwargs)

    @property
    def widgets_list(self):
        return self.tab, self.axis_pos, self.index_slider, self.axis_selector, self.out_main, self.out

    @property
    def widget(self):
        return widgets.VBox([widgets.HBox([self.axis_pos, self.index_slider, self.axis_selector]),
                             self.out_main, self.out])

    def __update_index_slider(self, _change):
        self.index_slider.value = round((self.axis_pos.value - self.data.axes[self.comp].min)
                                        / self.data.axes[self.comp].increment)

    def __axis_descr_format(self):
        return osh5vis.axis_format(self.data.axes[self.comp].long_name, self.data.axes[self.comp].units)

    def __get_slice(self, c):
        slcs = [slice(None)] * self.data.ndim
        slcs[c] = self.data.shape[c] // 2
        return slcs

    def switch_slice_direction(self, change):
        self.slcs, self.comp, self.x = \
            self.__get_slice(change['new']), change['new'], self.data.shape[change['new']] // 2
        self.reset_slider_index()
        self.__update_axis_descr()
        self.update_data(self.data[self.slcs], slcs=[i for i in self.slcs if not isinstance(i, int)])
        self.reset_plot_area()
        self.set_norm()
        # self.ax.cla()
        self.cb.remove()
        self.im.remove()
        self.im, self.cb = self.plot_data(im=self.im)

    def reset_slider_index(self):
        # stop the observe while updating values
        self.index_slider.unobserve(self.update_slice, 'value')
        self.index_slider.max = self.data.shape[self.comp] - 1
        self.__update_axis_value()
        self.index_slider.observe(self.update_slice, 'value')

    def __update_axis_value(self, *_):
        self.axis_pos.value = str(self.data.axes[self.comp][self.x])

    def __update_axis_descr(self, *_):
        self.axis_pos.description = self.__axis_descr_format()

    def update_slice(self, index):
        self.x = index['new']
        self.__update_axis_value()
        self.slcs[self.comp] = self.x
        self.redraw(self.data[self.slcs])


class DirSlicer(Generic2DPlotCtrl):
    def __init__(self, filefilter, processing=Generic2DPlotCtrl._idle, **extra_kwargs):
        fp = filefilter + '/*.h5' if os.path.isdir(filefilter) else filefilter
        self.filter, self.flist, self.processing = fp, sorted(glob.glob(fp)), processing
        try:
            self.data = processing(osh5io.read_h5(self.flist[0]))
        except IndexError:
            raise IOError('No file found matching ' + fp)

        items_layout = Layout(flex='1 1 auto', width='auto')
        self.file_slider = widgets.SelectionSlider(options=self.flist, description='filename:', value=self.flist[0],
                                                   continuous_update=False, layout=items_layout)
        self.time_label = widgets.Label(value=osh5vis.time_format(self.data.run_attrs['TIME'][0],
                                                                  self.data.run_attrs['TIME UNITS']),
                                        layout=items_layout)
        self.file_slider.observe(self.update_slice, 'value')

        super(DirSlicer, self).__init__(self.data, time_in_title=False, **extra_kwargs)


    @property
    def widgets_list(self):
        return self.tab, self.file_slider, self.time_label, self.out_main, self.out

    @property
    def widget(self):
        return widgets.VBox([widgets.HBox[self.file_slider, self.time_label], self.out_main, self.out])

    def update_slice(self, change):
        self.data = self.processing(osh5io.read_h5(change['new']))
        self.time_label.value = osh5vis.time_format(self.data.run_attrs['TIME'][0], self.data.run_attrs['TIME UNITS'])
        self.redraw(self.data)


class Animation(Slicer):
    def __init__(self, data, interval=10, step=1, **kwargs):
        super(Animation, self).__init__(data, **kwargs)
        self.play = widgets.Play(interval=interval, value=self.x, min=0, max=len(self.data.axes[self.comp]),
                                 step=step, description="Press play", disabled=False)
        self.interval_wgt = widgets.IntText(value=interval, description='Interval:', disabled=False)
        self.step_wgt = widgets.IntText(value=step, description='Step:', disabled=False)

        # link everything together
        widgets.jslink((self.play, 'value'), (self.index_slider, 'value'))
        self.interval_wgt.observe(self.update_interval, 'value')
        self.step_wgt.observe(self.update_step, 'value')

    @property
    def widgets_list(self):
        return (self.tab, self.axis_pos, self.index_slider, self.axis_selector,
                self.play, self.interval_wgt, self.step_wgt, self.out_main, self.out)

    def switch_slice_direction(self, change):
        super(Animation, self).switch_slice_direction(change)
        self.play.max = len(self.data.axes[self.comp])

    def update_interval(self, change):
        self.play.interval = change['new']

    def update_step(self, change):
        self.play.step = change['new']
