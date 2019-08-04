import autodisc as ad
from PIL import Image, ImageTk # pip3 install pillow
import numpy as np
try:
    import tkinter as tk
except:
    import Tkinter as tk
from tkinter import ttk
import uuid


class BaseFrame(tk.Frame):

    @staticmethod
    def default_gui_config():
        gui_config = ad.Config()

        gui_config.dialog = ad.Config()
        gui_config.dialog.title = 'GUIElement'
        gui_config.dialog.is_transient = False
        gui_config.dialog.geometry = None

        gui_config['frame'] = ad.Config()

        return gui_config


    def __init__(self, master=None, is_dialog=False, gui_config=None, **kwargs):

        if is_dialog:
            assert master is not None
            self.is_dialog = is_dialog
            self.master = tk.Toplevel(master=master)
        else:

            if master is None:
                self.master = tk.Tk()
                self.is_dialog = True

            else:
                self.master = master
                self.is_dialog = False

        # call the static get_default_gui_config method of the class that inherites from GUIElement
        self.gui_config = ad.config.set_default_config(kwargs, gui_config, self.__class__.default_gui_config())

        # overwrite the guiconfig parameters with the given frame parameters
        frame_parameters = ad.helper.data.set_dict_default_values(self.gui_config['frame'], kwargs)

        super().__init__(master=self.master, **frame_parameters)

        if self.is_dialog:
            # if the GUI is a dialog by itself, then resize the frame with it
            # otherwise, use the properties of the master
            self.master.rowconfigure(0, weight=1)
            self.master.columnconfigure(0, weight=1)
            self.grid(column=0, row=0, sticky=tk.NSEW)

            # set the title
            self.master.title(self.gui_config.dialog.title)
            if self.gui_config.dialog.is_transient:
                self.master.transient(master)

            if self.gui_config.dialog.geometry is not None:
                self.master.geometry(self.gui_config.dialog.geometry)



class DictTableGUI(BaseFrame):
    '''Displays a dictionary as a table'''

    @staticmethod
    def default_gui_config():

        default_gui_config = BaseFrame.default_gui_config()
        default_gui_config['value_column_heading'] = {'text': 'Values'}
        default_gui_config['value_column'] = {'width': 100, 'anchor': 'center'}

        return default_gui_config

    def __init__(self, master=None, is_dialog=False, gui_config=None, **kwargs):

        super().__init__(master=master, is_dialog=is_dialog, gui_config=gui_config, **kwargs)
        self.create_gui()


    def create_gui(self):

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.tree = ttk.Treeview(self, columns=('values'))
        self.tree.column('values', **self.gui_config['value_column'])
        self.tree.heading('values', **self.gui_config['value_column_heading'])
        self.tree.grid(column=0, row=0, sticky=tk.NSEW)

        self.scrollbar_y = tk.Scrollbar(self, orient=tk.VERTICAL, command=self.tree.yview)
        self.scrollbar_y.grid(column=1, row=0, sticky=tk.NS)

        self.scrollbar_x = tk.Scrollbar(self, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.scrollbar_x.grid(column=0, row=1, sticky=tk.EW)

        self.tree.config(yscrollcommand=self.scrollbar_y.set)
        self.tree.config(xscrollcommand=self.scrollbar_x.set)

        # Limit windows minimum dimensions
        self.master.update_idletasks()
        #self.master.minsize(self.master.winfo_reqwidth(), self.master.winfo_reqheight())


    def display_data(self, data):
        self.fill_tree('', data)


    def fill_tree(self, parent, dic):

        self.tree.delete(*self.tree.get_children())

        for key in dic.keys():
            uid = uuid.uuid4()
            if isinstance(dic[key], dict):
                self.tree.insert(parent, 'end', uid, text=key)
                self.fill_tree(uid, dic[key])
            # elif isinstance(dic[key], tuple):
            #     self.tree.insert(parent, 'end', uid, text=str(key) + '()')
            #     self.fill_tree(uid,
            #                    dict([(i, x) for i, x in enumerate(dic[key])]))
            # elif isinstance(dic[key], list):
            #     self.tree.insert(parent, 'end', uid, text=str(key) + '[]')
            #     self.fill_tree(uid,
            #                    dict([(i, x) for i, x in enumerate(dic[key])]))
            else:
                value = dic[key]
                if isinstance(value, str):
                    value = value.replace(' ', '_')
                self.tree.insert(parent, 'end', uid, text=key, value=value)



class ImageViewerGUI(BaseFrame):
    '''Displays regular image files.'''

    @staticmethod
    def default_gui_config():

        default_gui_config = BaseFrame.default_gui_config()
        default_gui_config['value_column'] = {'width': 100, 'anchor': 'center'}
        default_gui_config['image_labels'] = {'anchor': tk.NW}
        default_gui_config['images'] = {'anchor': tk.NW}
        default_gui_config['padding'] = 5
        default_gui_config['is_resize_to_imagesize'] = True
        return default_gui_config


    def __init__(self, master=None, is_dialog=False, gui_config=None, **kwargs):

        super().__init__(master=master, is_dialog=is_dialog, gui_config=gui_config, **kwargs)

        self.photo_images = []  # needs to keep a reference to the photo_images, otherwise they might be deleted
        self.create_gui()


    def create_gui(self):

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # canvas where the 2d observations are drawn
        self.canvas = tk.Canvas(self)
        self.canvas.grid(row=0, column=0, sticky=tk.NSEW)

        self.scrollbar_y = AutoScrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollbar_y.grid(column=1, row=0, sticky=tk.NS)

        self.scrollbar_x = AutoScrollbar(self, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.scrollbar_x.grid(column=0, row=1, sticky=tk.EW)

        self.canvas.config(yscrollcommand=self.scrollbar_y.set)
        self.canvas.config(xscrollcommand=self.scrollbar_x.set)


    def bind(self, sequence=None, func=None, add=None):
        # add binding to the canvas that the frame has
        self.canvas.bind(sequence, func, add)


    def display_data(self, image_data, image_names=None):
        self.set_image_data(image_data)
        self.set_image_names(image_names)

        # clear all existing images and labels
        self.canvas.delete("all")
        self.photo_images = []

        if self.image_data:

            row_y_pos = 0
            x_pos = 0
            max_column_y_pos = 0

            for row_idx in range(len(self.image_data)):

                row_images = self.image_data[row_idx]

                row_image_names = None
                if self.image_names:
                    row_image_names = self.image_names[row_idx]

                # check if the column has several rows
                # if not then encapsulate the single image of the column in a list for convenience
                if not isinstance(row_images, list):
                    row_images = [row_images]

                    if row_image_names:
                        row_image_names = [row_image_names]

                # draw each image in the current column
                for col_idx in range(len(row_images)):

                    y_pos = row_y_pos

                    # check if a label exists for it
                    if row_image_names and (len(row_image_names) > col_idx):

                        image_name = row_image_names[col_idx]

                        # draw the name:
                        textitem_id = self.canvas.create_text(x_pos, y_pos, text=image_name, **self.gui_config['image_labels'])

                        bounds = self.canvas.bbox(textitem_id) #(x1,y1, x2, y2)
                        y_pos = bounds[3] + 1  + self.gui_config['padding']

                    # draw image

                    img = self.get_image_from_data(row_images[col_idx])
                    photo_image = ImageTk.PhotoImage(image=img)
                    canvas_image_item_id = self.canvas.create_image(x_pos, y_pos, image=photo_image, **self.gui_config['images'])
                    self.photo_images.append(photo_image)

                    bounds = self.canvas.bbox(canvas_image_item_id)  # get bounds: (x1,y1, x2, y2)
                    x_pos = bounds[2] + 1  + self.gui_config['padding']

                    # get width of the image and check if it defines the maximum width of the row
                    max_column_y_pos = max(max_column_y_pos, bounds[3])

                row_y_pos = max_column_y_pos + 1 + self.gui_config['padding']
                x_pos = 0
                max_column_y_pos = 0

        elif image_names:
            # no image, but names --> plot them
            y_pos = 0
            for image_name in self.image_names:
                textitem_id = self.canvas.create_text(0, y_pos, text=image_name, **self.gui_config['image_labels'])
                bounds = self.canvas.bbox(textitem_id)  # (x1,y1, x2, y2)
                y_pos = bounds[3] + 1 + self.gui_config['padding']

        images_bound = self.canvas.bbox("all") # (x1,y1, x2, y2)
        self.canvas.configure(scrollregion=images_bound)

        if self.gui_config['is_resize_to_imagesize'] and images_bound is not None:
            images_height = images_bound[3] - images_bound[1]
            images_width = images_bound[2] - images_bound[0]
            self.canvas.configure(width=images_width, height=images_height)


    def get_image_from_data(self, data):
        '''By overriding this function the subclasses can convert the data to a format that the ImageTk.PhotoImage class accepts as input.'''
        return data


    def set_image_data(self, image_data):

        # identify if the image_data is a list of images or a single image
        # if it is a single image, encapsulate it in a list

        if isinstance(image_data, list) or image_data is None:
            self.image_data = image_data
        else:
            self.image_data = [image_data]


    def set_image_names(self, image_names):
        if isinstance(image_names, list) or image_names is None:
            self.image_names = image_names
        else:
            self.image_names = [image_names]



class ArrayToImageGUI(ImageViewerGUI):
    '''Displays images that are defined as arrays.'''

    # TODO: use the ImageViewerGUI as basis for this gui, because they have the same functionality

    @staticmethod
    def default_gui_config():

        default_gui_config = ImageViewerGUI.default_gui_config()
        default_gui_config.colormap = [[0, 0, 0], [3, 3, 3], [4, 4, 4], [5, 5, 5], [8, 8, 8]] # BW
        default_gui_config.pixel_size = 1
        return default_gui_config


    def __init__(self, master=None, is_dialog=False, gui_config=None, **kwargs):
        super().__init__(master=master, is_dialog=is_dialog, gui_config=gui_config, **kwargs)
        self.colormap = self.create_colormap(np.array(self.gui_config['colormap']))


    def get_image_from_data(self, data):

        if data is not None:
            buffer = np.uint8(np.clip(data, 0, 1) * 252)
            buffer = np.repeat(np.repeat(buffer, self.gui_config['pixel_size'], axis=0),
                               self.gui_config['pixel_size'], axis=1)
            image = Image.frombuffer('P', (buffer.shape[1], buffer.shape[0]), buffer, 'raw', 'P', 0, 1)
            image.putpalette(self.colormap)
        else:
            size = (3, 3)
            image = Image.frombuffer('L', size, np.zeros(size), 'raw', 'L', 0, 1)

        return image


    def create_colormap(self, colors):
        nval = 256 - 3
        ncol = colors.shape[0]
        colors = np.vstack((colors, np.array([[0, 0, 0]])))
        v = np.repeat(range(nval), 3)  # [0 0 0 1 1 1 ... 252 252 252]
        i = np.array(list(range(3)) * nval)  # [0 1 2 0 1 2 ... 0 1 2]
        k = v / (nval - 1) * (ncol - 1)  # interpolate between 0 .. ncol-1
        k1 = k.astype(int)
        c1, c2 = colors[k1, i], colors[k1 + 1, i]
        c = (k - k1) * (c2 - c1) + c1  # interpolate between c1 .. c2
        return np.rint(c / 8 * 255).astype(int).tolist() + [95, 95, 95, 127, 127, 127, 255, 255, 255]



class AutoScrollbar(tk.Scrollbar):
    '''
    A scrollbar that hides itself if it's not needed.

    Only works if you use the grid geometry manager.
    '''

    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            # grid_remove is currently missing from Tkinter!
            self.tk.call("grid", "remove", self)
        else:
            self.grid()
        tk.Scrollbar.set(self, lo, hi)

    def pack(self, **kw):
        raise tk.TclError("cannot use pack with this widget")

    def place(self, **kw):
        raise tk.TclError("cannot use place with this widget")



def print_progress_bar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '#'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    if iteration == total:
        print()
