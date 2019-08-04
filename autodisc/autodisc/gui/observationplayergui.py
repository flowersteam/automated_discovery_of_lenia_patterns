from autodisc.gui.gui import BaseFrame
import numpy as np
import PIL.ImageDraw
from PIL import Image, ImageTk # pip3 install pillow
try:
    import tkinter as tk
except:
    import Tkinter as tk
import math
import time


class ObservationPlayerGUI(BaseFrame):

    @staticmethod
    def default_gui_config():

        def_config = BaseFrame.default_gui_config()

        def_config.dialog.title = 'Observation Player'
        def_config.pixel_size = 8
        def_config.frame_rate = 10
        def_config.colormap_id = 6

        return def_config


    def __init__(self, master=None, gui_config=None, **kwargs):
        super().__init__(master=master, gui_config=gui_config, **kwargs)

        self.is_run = True
        self.is_closing = False
        self.last_frame_in_millisec = 0
        self.step_idx = None

        self.statistics = None


        ''' http://hslpicker.com/ '''
        self.colormaps = [
            self.create_colormap(np.array([[0, 0, 4], [0, 0, 8], [0, 4, 8], [0, 8, 8], [4, 8, 4], [8, 8, 0], [8, 4, 0], [8, 0, 0], [4, 0, 0]])), # BCYR
            self.create_colormap(np.array([[0, 2, 0], [0, 4, 0], [4, 6, 0], [8, 8, 0], [8, 4, 4], [8, 0, 8], [4, 0, 8], [0, 0, 8], [0, 0, 4]])), # GYPB
            self.create_colormap(np.array([[4, 0, 2], [8, 0, 4], [8, 0, 6], [8, 0, 8], [4, 4, 4], [0, 8, 0], [0, 6, 0], [0, 4, 0], [0, 2, 0]])), # PPGG
            self.create_colormap(np.array([[4, 4, 6], [2, 2, 4], [2, 4, 2], [4, 6, 4], [6, 6, 4], [4, 2, 2]])),  # BGYR
            self.create_colormap(np.array([[4, 6, 4], [2, 4, 2], [4, 4, 2], [6, 6, 4], [6, 4, 6], [2, 2, 4]])),  # GYPB
            self.create_colormap(np.array([[6, 6, 4], [4, 4, 2], [4, 2, 4], [6, 4, 6], [4, 6, 6], [2, 4, 2]])),  # YPCG
            self.create_colormap(np.array([[0, 0, 0], [3, 3, 3], [4, 4, 4], [5, 5, 5], [8, 8, 8]])), # B/W
            # white --> blue --> pink
            self.create_colormap(np.array([[255, 255, 255], [119, 255, 255], [23, 223, 252], [0, 190, 250], [0, 158, 249], [0, 142, 249], [81, 125, 248], [150, 109, 248], [192, 77, 247], [232, 47, 247], [255, 9, 247], [200, 0, 84]]) / 255 * 8),
            ]
        #self.colormap_id = 6

        self.image_size_y = 0
        self.image_size_x = 0

        self.create_gui()

        self.update_win()

        self.master.protocol('WM_DELETE_WINDOW', self.close)
        self.master.after(100, self.run)


    def set_observations(self, observations):
        self.observations = observations

        #if self.observations is None or not self.observations:
        if self.observations is None:
            self.image_size_x = 0
            self.image_size_y = 0
        else:
            self.image_size_x = self.observations[0].shape[0] * self.gui_config.pixel_size
            self.image_size_y = self.observations[0].shape[1] * self.gui_config.pixel_size

        self.step_idx = 0


    def create_gui(self):

        # make the canvas in the frame resizable
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # canvas where the 2d observations are drawn
        self.canvas = tk.Canvas(self)
        self.canvas.grid(row=0, column=0, sticky=tk.NSEW)

        self.canvas_image = self.canvas.create_image(0,0,anchor=tk.NW)

        self.info_label = tk.Label(self.master)
        self.info_label.grid(row=1, column=0, sticky=tk.NSEW)


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


    def display_exploration_data(self, exploration_data, run_id):
        self.display_data(observations = exploration_data.runs[run_id].observations.states,
                          statistics = exploration_data.runs[run_id].statistics)


    def display_data(self, observations, statistics):
        self.set_observations(observations)
        self.statistics = statistics

        # update the size of the canvas depending on image size:
        self.canvas.configure(width=self.image_size_x, height=self.image_size_y)


    def update_win(self):

        if self.step_idx:
            self.draw_observation(self.observations[self.step_idx], 0, 1)

            # draw additional elements if specified
            # TODO: allow to turn on and of the drawing of elements via a menu
            if 'elements' in self.gui_config:
                for element_config in self.gui_config['elements']:
                    if 'is_visible' not in element_config or element_config['is_visible']:
                        self.draw_element(element_config)

            self.img.putpalette(self.colormaps[self.gui_config.colormap_id])

        else:
            self.draw_black()

        self.photo = ImageTk.PhotoImage(image=self.img)
        self.canvas.itemconfig(self.canvas_image, image=self.photo)
        self.canvas.update()


    def draw_observation(self, A, vmin=0, vmax=1):
        buffer = np.uint8(np.clip((A - vmin) / (vmax - vmin), 0, 1) * 252)  # .copy(order='C')
        buffer = np.repeat(np.repeat(buffer, self.gui_config['pixel_size'], axis=0), self.gui_config['pixel_size'], axis=1)
        self.img = Image.frombuffer('P', (self.image_size_x, self.image_size_y),
                                    buffer, 'raw', 'P', 0, 1)


    def draw_element(self, element_config):

        draw = PIL.ImageDraw.Draw(self.img)

        if element_config['type'] is 'arrow_angle':
            # TODO: draw the arrow head
            # TODO: allow selection of line width and point radius
            # TODO: allow selection of color

            # get the data needed to draw the arrow
            position = self.get_element_source(element_config['position'])
            length = self.get_element_source(element_config['length'])
            angle = self.get_element_source(element_config['angle'])

            start_y = position[0] * self.gui_config['pixel_size']
            start_x = position[1] * self.gui_config['pixel_size']

            # draw start point of arrow
            # defined by a 4-tuple, (x0, y0, x1, y1) where (x0, y0) is the top left (northwest) corner of the rectangle, and (x1, y1) is the bottom right (southeast) corner
            radius = 3
            radius = radius * self.gui_config['pixel_size']
            circle_box = (start_x - radius,
                          start_y - radius,
                          start_x + radius + 1,
                          start_y + radius + 1)
            draw.ellipse(circle_box, fill=255)

            # draw arrow if an angle and length are defined
            if not np.isnan(angle) or not np.isnan(length):
                end_y = (position[0] - length * math.sin(math.radians(angle))) * self.gui_config['pixel_size']
                end_x = (position[1] + length * math.cos(math.radians(angle))) * self.gui_config['pixel_size']

                draw.line([(start_x, start_y), (end_x, end_y)], fill=255, width=3)

        else:
            raise ValueError('Unknown element type {!r}!'.format(element_config['type']))

        del draw


    def get_element_source(self, source_config):

        val = None

        if not isinstance(source_config, dict):
            val = source_config
        else:
            if source_config['source'] is 'statistics' or source_config['source'] is 'stat':

                if isinstance(self.statistics[source_config['name']], np.ndarray):
                    if self.statistics[source_config['name']].shape[0] == 1:
                        val = self.statistics[source_config['name']][0]
                    else:
                        val = self.statistics[source_config['name']][self.step_idx]

                elif isinstance(self.statistics[source_config['name']], list):
                    if len(self.statistics[source_config['name']]) == 1:
                        val = self.statistics[source_config['name']][0]
                    else:
                        val = self.statistics[source_config['name']][self.step_idx]
                else:
                    val = self.statistics[source_config['name']]

            else:
                raise ValueError('Unknown source type {!r} for an element configuaration!'.format(source_config['source']))

            if 'factor' in source_config:
                val = val * source_config['factor']

        return val


    def draw_black(self):
        size = (self.image_size_x * self.gui_config['pixel_size'], self.image_size_y * self.gui_config['pixel_size'])
        self.img = Image.frombuffer('L', size, np.zeros(size), 'raw', 'L', 0, 1)


    def update_info(self):
        info_st = 'Step: ' + str(self.step_idx)
        self.info_label.config(text=info_st)


    def close(self):
        self.is_closing = True
        self.master.destroy()


    def run(self):

        if self.is_closing:
            return

        if self.is_run:

            update_every_millisec = 1000 / self.gui_config['frame_rate']

            cur_time_in_millisec = int(round(time.time() * 1000))
            if cur_time_in_millisec - self.last_frame_in_millisec >= update_every_millisec:

                if self.step_idx is None or self.step_idx >= len(self.observations)-1:
                    self.step_idx = 0
                else:
                    self.step_idx = self.step_idx + 1

                self.last_frame_in_millisec = cur_time_in_millisec

            self.update_info()
            self.update_win()

        self.master.after(int(update_every_millisec/2), self.run)