import autodisc as ad
import ipywidgets
import collections
import zipfile
import numpy as np
import warnings
import os
import PIL
import io


class PlotExperimentImagesWidget(ipywidgets.Box):

    @staticmethod
    def get_default_gui_config():
        default_config = ad.Config()

        default_config.box_layout = ad.Config()
        default_config.box_layout.flex_flow = 'column'
        default_config.box_layout.display = 'flex'
        default_config.box_layout.align_items = 'stretch'

        default_config.experiment_repetition_loader = ad.gui.jupyter.ExperimentRepetitionLoaderWidget.get_default_gui_config()

        default_config.filter_selection = ad.Config()
        default_config.filter_selection.description = 'Filter:'
        default_config.filter_selection.layout = ad.Config()
        default_config.filter_selection.layout.flex = '1 1 0%'
        default_config.filter_selection.layout.width = 'auto'

        default_config.color_map = np.array([[255,255,255], [119,255,255],[23,223,252],[0,190,250],[0,158,249],[0,142,249],[81,125,248],[150,109,248],[192,77,247],[232,47,247],[255,9,247],[200,0,84]])/255*8

        return default_config

    def __init__(self, experiment_definitions, repetition_ids, data_source, experiment_statistics, filters=None, config=None, **kwargs):

        self.config = ad.config.set_default_config(kwargs, config, PlotExperimentImagesWidget.get_default_gui_config())

        self.data = []
        self.experiment_definitions = experiment_definitions
        self.repetition_ids = repetition_ids
        self.data_source = data_source
        self.experiment_statistics = experiment_statistics
        self.filters = filters

        gui_elements = []

        self.exp_rep_selection_widget = ad.gui.jupyter.ExperimentRepetitionLoaderWidget(
            self.experiment_definitions,
            self.repetition_ids,
            self.load_images,
            config=self.config.experiment_repetition_loader
        )
        gui_elements.append(self.exp_rep_selection_widget)

        self.exp_rep_selection_widget.load_btn.on_click(self.on_load_button_click)

        # create Filter selection
        if self.filters is not None and self.filters:
            self.filter_selection_widget = ipywidgets.Dropdown(
                options=list(self.filters.keys()),
                **self.config.filter_selection
            )
            self.filter_selection_widget.observe(self.on_filter_change)

            gui_elements.append(self.filter_selection_widget)

        self.image_list_widget = ad.gui.jupyter.ImageListWidget()
        gui_elements.append(self.image_list_widget)

        super().__init__(
            gui_elements,
            layout=self.config.box_layout
        )

        self.experiment_ids = [exp_def['id'] for exp_def in self.experiment_definitions]

        self.images = dict()

    def on_filter_change(self, change):
        if change['type'] == 'change' and change['name'] == 'value':
            self.plot_images()

    def on_load_button_click(self, button):
        self.plot_images()

    def load_images(self, experiment_id, repetition_id):

        exp_idx = self.experiment_ids.index(experiment_id)

        if repetition_id is None:
            path = self.experiment_definitions[exp_idx]['directory']
        else:
            path = os.path.join(
                self.experiment_definitions[exp_idx]['directory'],
                'repetition_{:06d}'.format(int(repetition_id))
            )

        path = os.path.join(path, 'statistics')
        for sub_data_source in self.data_source.split('.'):
            path = os.path.join(path, sub_data_source)

        self.images = collections.OrderedDict()

        # check if the given datasource is a directory with images or a zip file
        if os.path.isdir(path):

            dir_content = os.listdir(path)
            for dir_item in dir_content:

                file_path = os.path.join(path, dir_item)

                if os.path.isfile(file_path) and '.png' in dir_item:
                    item_id = dir_item.split('.')[0]
                    item_id = int(item_id)

                    file = open(file_path, "rb")

                    self.images[item_id] = file.read()

        elif zipfile.is_zipfile(path + '.zip'):

            zf = zipfile.ZipFile(path + '.zip', 'r')

            for image_file_name in zf.namelist():
                item_id = image_file_name.split('.')[0]
                item_id = int(item_id)

                self.images[item_id] = zf.read(image_file_name)
        else:
            warnings.warn('No Images found under {!r}'.format(path))

        # change color of images if defined
        if self.config.color_map is not None:
            color_map = ad.gui.jupyter.misc.create_colormap(self.config.color_map)
            for image_id, image in self.images.items():
                img_bytes = io.BytesIO(image)
                img_pil = PIL.Image.open(img_bytes)
                self.images[image_id] = ad.gui.jupyter.misc.transform_image_PIL_to_bytes(ad.gui.jupyter.misc.transform_image_from_colormap(img_pil, color_map))


    def plot_images(self):

        if len(self.images) > 0:

            exp_id = self.exp_rep_selection_widget.exp_selection_widget.value
            rep_id = self.exp_rep_selection_widget.rep_selection_widget.value

            cur_filter = self.filters[self.filter_selection_widget.value]

            # filter
            if cur_filter is None or not cur_filter:
                img_ids = range(len(self.images))
            else:
                run_ids = ad.gui.jupyter.misc.filter_single_experiment_data(self.experiment_statistics[exp_id], cur_filter, int(rep_id))
                img_ids = np.where(run_ids)[0]

            cur_images = [self.images[img_id] for img_id in img_ids]
            captions = ['{}:'.format(img_id) for img_id in img_ids]

            self.image_list_widget.update(cur_images, captions)
        else:
            self.image_list_widget.update([])