import autodisc as ad
import ipywidgets
import numpy as np
import IPython.display

class ImageListWidget(ipywidgets.VBox):

    @staticmethod
    def get_default_gui_config():
        default_config = ad.Config()

        default_config.elements_per_page = 100

        default_config.output_layout = ad.Config()
        # default_config.output_layout.border='3px solid black'

        default_config.box_layout = ad.Config()
        default_config.box_layout.overflow_y = 'scroll'
        default_config.box_layout.width = '100%'
        default_config.box_layout.height = '500px'
        default_config.box_layout.flex_flow = 'row wrap'
        default_config.box_layout.display = 'flex'

        default_config.content_ouput = ad.Config()
        default_config.page_label = ad.Config()

        default_config.page_selection = ad.Config()
        default_config.page_selection.description = 'Page: '

        default_config.previous_page_button = ad.Config()
        default_config.previous_page_button.description = '<'
        default_config.previous_page_button.layout = ad.Config()
        default_config.previous_page_button.layout.width = '20px'

        default_config.next_page_button = ad.Config()
        default_config.next_page_button.description = '>'
        default_config.next_page_button.layout = ad.Config()
        default_config.next_page_button.layout.width = '20px'

        default_config.button_box = ad.Config()
        default_config.button_box.layout = ad.Config()
        default_config.button_box.layout.flex_flow = 'row'
        default_config.button_box.layout.display = 'flex'
        default_config.button_box.layout.align_items = 'center'
        default_config.button_box.layout['justify-content'] = 'flex-end'
        default_config.button_box.layout.width = '100%'

        default_config.image_items = ad.Config()
        default_config.image_items.layout = ad.Config()
        default_config.image_items.layout.height = '200px'
        default_config.image_items.layout.width = '200px'
        default_config.image_items.layout.border = '2px solid white'

        default_config.image_captions = ad.Config()

        return default_config

    def __init__(self, images=None, config=None, **kwargs):

        self.config = ad.config.set_default_config(kwargs, config, ImageListWidget.get_default_gui_config())

        self.images = None
        self.main_box = None

        self.content_ouput_widget = ipywidgets.Output(**self.config.content_ouput)

        self.page_label_widget = ipywidgets.Label(**self.config.page_label, value='of 0')

        self.previous_page_button_widget = ipywidgets.Button(**self.config.previous_page_button)
        self.previous_page_button_widget.on_click(self.on_prev_page_button_click)

        self.page_selection_widget = ipywidgets.Dropdown(**self.config.page_selection)
        self.page_selection_widget.observe(self.on_page_selection_change)

        self.next_page_button_widget = ipywidgets.Button(**self.config.next_page_button)
        self.next_page_button_widget.on_click(self.on_next_page_button_click)
        self.page_selection_widget_ignore_next_value_change = False

        self.button_box_widget = ipywidgets.Box(
            [self.page_selection_widget,
             self.page_label_widget,
             self.previous_page_button_widget,
             self.next_page_button_widget],
            **self.config.button_box
        )

        super().__init__([self.content_ouput_widget, self.button_box_widget], layout=self.config.output_layout)

        self.cur_page_idx = 0

        if images is not None:
            self.update(images)

    def update(self, images, captions=None):
        self.images = images
        self.captions = captions

        if self.images is not None and self.images:

            # update page selection widget
            n_pages = int(np.ceil(len(self.images) / self.config.elements_per_page))

            opts = [page_idx + 1 for page_idx in range(n_pages)]
            self.page_selection_widget.options = opts

            # update number of pages
            self.page_label_widget.value = 'of {}'.format(n_pages)

            self.update_page_items(0, force_update=True)
        else:
            self.page_selection_widget.options = []
            self.page_label_widget.value = 'of 0'
            self.content_ouput_widget.clear_output()

    def update_page_items(self, page_idx, force_update=False):

        if self.images is not None and self.images:
            n_pages = int(np.ceil(len(self.images) / self.config.elements_per_page))

            if n_pages == 0:
                self.content_ouput_widget.clear_output()
            elif page_idx >= 0 and page_idx < n_pages and (self.cur_page_idx != page_idx or force_update):

                items = []

                self.cur_page_idx = page_idx

                start_idx = self.config.elements_per_page * self.cur_page_idx
                end_idx = min(self.config.elements_per_page * (self.cur_page_idx + 1), len(self.images))

                for image_idx in range(start_idx, end_idx):

                    image = self.images[image_idx]

                    item_elems = []

                    if self.captions is not None:

                        if image_idx < len(self.captions):
                            caption_text = self.captions[image_idx]
                        else:
                            caption_text = ''

                        caption_widget = ipywidgets.Label(
                            value=caption_text,
                            **self.config.image_captions
                        )

                        item_elems.append(caption_widget)

                    img_widget = ipywidgets.Image(
                        value=image,
                        format='png',
                        **self.config.image_items
                    )
                    item_elems.append(img_widget)

                    items.append(ipywidgets.VBox(item_elems))

                self.main_box = ipywidgets.Box(items, layout=self.config.box_layout)

                self.content_ouput_widget.clear_output(wait=True)
                with self.content_ouput_widget:
                    IPython.display.display(self.main_box)

                self.page_selection_widget.value = page_idx + 1
        else:
            self.content_ouput_widget.clear_output()

    def on_prev_page_button_click(self, button):
        self.update_page_items(self.cur_page_idx - 1)

    def on_next_page_button_click(self, button):
        self.update_page_items(self.cur_page_idx + 1)

    def on_page_selection_change(self, change):
        if change['type'] == 'change' and change['name'] == 'value':
            if self.page_selection_widget.value is not None:
                self.update_page_items(self.page_selection_widget.value - 1)