import autodisc as ad
import ipywidgets
import collections
import zipfile
import numpy as np
import warnings
import os
import plotly
from PIL import Image
from autodisc.gui.jupyter.misc import create_colormap, transform_image_from_colormap, transform_image_PIL_to_bytes


''' ---------------------------------------------------------------------------------
        LATENT SPACE VIEWER GUI:
            - LATENT SPACE VISUALISATION GRAPH (TSNE - PCA - UMAP - 2 by 2 dims)
            - IMAGE LIST DISPLAY
----------------------------------------------------------------------------------'''

class LatentSpaceViewerWidget(ipywidgets.Box):    
    @staticmethod
    def get_default_gui_config():
        default_config = ad.Config()
        
        # general box layout config
        default_config.box_layout = ad.Config()
        default_config.box_layout.flex_flow = 'column'
        default_config.box_layout.display = 'flex'
        default_config.box_layout.align_items = 'stretch'
        
        # Row 0: 2D visualisation type selector
        default_config.visualisation_type_selector = ad.Config()
        default_config.visualisation_type_selector.description = 'visualisation type'
        default_config.visualisation_type_selector.options = ['T-SNE', 'PCA', 'UMAP', '0-1', '0-2', '0-3', '0-4', '0-5', '0-6', '0-7']
        default_config.visualisation_type_selector.value = 'UMAP'
        default_config.visualisation_type_selector.disabled = False
        
        # Row 1: Canvas graph
        default_config.canvas = ad.Config()
        
        ## default print properties for layout
        multiplier = 2
        pixel_cm_ration = 35
        width_full = int(13.95 * pixel_cm_ration) * multiplier
        width_half = int(13.95/2 * pixel_cm_ration) * multiplier
        height_default = int(3.5 * pixel_cm_ration) * multiplier
        top_margin = 0 * multiplier 
        left_margin = 0 * multiplier 
        right_margin = 0 * multiplier 
        bottom_margin = 0 * multiplier 
        font_family = 'Times New Roman'
        font_size = 10*multiplier
        default_config.canvas.layout = dict(
                    font = dict(
                        family=font_family, 
                        size=font_size, 
                        ),
                    updatemenus=[],
                    width=width_half, # in cm
                    height=height_default, # in cm
                    
                    margin = dict(
                        l=left_margin, #left margin in pixel
                        r=right_margin, #right margin in pixel
                        b=bottom_margin, #bottom margin in pixel
                        t=top_margin,  #top margin in pixel
                        ),
                    
                    legend=dict(
                        font = dict(
                        family=font_family, 
                        size=font_size, 
                        ),
                        xanchor='right',
                        yanchor='bottom',
                        y=0.,
                        x=1.0,
                        ),
                    showlegend=False
                    )  
        
        ## scatter default config
        default_config.canvas_data = ad.Config()
        default_config.canvas_data.mode = 'markers'
        default_config.canvas_data.marker = dict(
                            size=4.0,
                            opacity=1.0
                            )
        
        ## colors and symbols for markers
        default_config.marker_colorlist = dict()
        default_config.marker_colorlist['dead'] = 'rgb(0,0,0)'
        default_config.marker_colorlist['animal'] = 'rgb(230,159,0)'
        default_config.marker_colorlist['non-animal'] = 'rgb(0,158,115)'
        
        default_config.marker_symbollist = dict()
        default_config.marker_symbollist['dead'] = 'circle'
        default_config.marker_symbollist['animal'] = 'diamond'
        default_config.marker_symbollist['non-animal'] = 'square'
        
        
        # Row 2: Image list widget for selected points
        default_config.image_list_widget = ad.gui.jupyter.ImageListWidget.get_default_gui_config()
        default_config.image_list_widget.image_items.layout.border = '1px solid gray' # add gray countour around images displayed
        
        ## colormap for images (BW --> colormap)
        default_config.transform_images = True
        default_config.transform_images_colormaps = [
            create_colormap(np.array([[255,255,255], [119,255,255],[23,223,252],[0,190,250],[0,158,249],[0,142,249],[81,125,248],[150,109,248],[192,77,247],[232,47,247],[255,9,247],[200,0,84]])/255*8), #WBPR
			create_colormap(np.array([[0,0,4],[0,0,8],[0,4,8],[0,8,8],[4,8,4],[8,8,0],[8,4,0],[8,0,0],[4,0,0]])), #BCYR
			create_colormap(np.array([[0,2,0],[0,4,0],[4,6,0],[8,8,0],[8,4,4],[8,0,8],[4,0,8],[0,0,8],[0,0,4]])), #GYPB
			create_colormap(np.array([[4,0,2],[8,0,4],[8,0,6],[8,0,8],[4,4,4],[0,8,0],[0,6,0],[0,4,0],[0,2,0]])), #PPGG
			create_colormap(np.array([[4,4,6],[2,2,4],[2,4,2],[4,6,4],[6,6,4],[4,2,2]])), #BGYR
			create_colormap(np.array([[4,6,4],[2,4,2],[4,4,2],[6,6,4],[6,4,6],[2,2,4]])), #GYPB
			create_colormap(np.array([[6,6,4],[4,4,2],[4,2,4],[6,4,6],[4,6,6],[2,4,2]])), #YPCG
			create_colormap(np.array([[8,8,8],[7,7,7],[5,5,5],[3,3,3],[0,0,0]]), is_marker_w=False), #W/B
			create_colormap(np.array([[0,0,0],[3,3,3],[5,5,5],[7,7,7],[8,8,8]]))] #B/W

        
        return default_config


    def __init__(self, database=None, config=None, **kwargs):
        self.config = ad.config.set_default_config(kwargs, config, LatentSpaceViewerWidget.get_default_gui_config())
        
        # displayed list of images and labels 
        self.displayed_data = dict()
        self.displayed_data['images'] = []
        self.displayed_data['captions'] = []
        # current selected box by the user
        self.selection = dict()
        self.selection['xrange'] = [None] * 2
        self.selection['yrange'] = [None] * 2
        
        # GUI LAYOUT
        gui_elements = []
        ## Row 0: 
        self.visualisation_type_selector = ipywidgets.Dropdown(**self.config.visualisation_type_selector)
        self.visualisation_type_selector.observe(self.on_visualisation_type_selector_change)
        gui_elements.append(self.visualisation_type_selector)
        
        ## Row 1: 
        self.canvas = plotly.graph_objs.FigureWidget(**self.config.canvas)
        gui_elements.append(self.canvas)
        
        ## Row 2: 
        self.image_list_widget = ad.gui.jupyter.ImageListWidget(config=self.config.image_list_widget)

        gui_elements.append(self.image_list_widget)
        
        super().__init__(gui_elements, layout=self.config.box_layout)
        
        return
    
    def reset_displayed_content(self):
        # reset the displayed data:
        self.displayed_data['images'] = []
        self.displayed_data['captions'] = []
        self.image_list_widget.update(self.displayed_data['images'], self.displayed_data['captions'])
        ## current selected box by the user
        self.selection = dict()
        self.selection['xrange'] = [None] * 2
        self.selection['yrange'] = [None] * 2
        
        if len(self.canvas.data) > 0:
            self.canvas.data = []
            
    
    def update(self, database):
        # Update database
        self.database = database
        
        # if transform apply it to images of database
        if self.config.transform_images:
            for item_id, item in self.database.items():
                item['image'] = transform_image_PIL_to_bytes(transform_image_from_colormap(item['image'], self.config.transform_images_colormaps[0]))
                
        # Data filtered structure
        self.database_filtered_ids = collections.OrderedDict()
        for item_id, item in self.database.items():
            if item['label'] not in self.database_filtered_ids:
                self.database_filtered_ids[item['label']] = []
            self.database_filtered_ids[item['label']].append(item_id)   

        # Update plot (default: tsne) with one trace per "filter"
        self.reset_displayed_content()
        self.plot_latent_space_graph()
            
    
    def plot_latent_space_graph(self):        
                
        if self.visualisation_type_selector.value == 'T-SNE':
            for filter_idx, filter_id in enumerate(self.database_filtered_ids):
                xdata = [self.database[point_id]['tsne_point'][0] for point_id in self.database_filtered_ids[filter_id]]
                ydata = [self.database[point_id]['tsne_point'][1] for point_id in self.database_filtered_ids[filter_id]]
                self.canvas.add_scatter(x = xdata,
                                y = ydata,
                                **self.config.canvas_data)
                self.canvas.data[filter_idx].marker.color = self.config.marker_colorlist[filter_id]
                self.canvas.data[filter_idx].marker.symbol = self.config.marker_symbollist[filter_id]
                self.canvas.data[filter_idx].name = filter_id
                
                self.canvas.data[filter_idx].on_selection(self.selection_fn)
                
        elif self.visualisation_type_selector.value == 'PCA':
            for filter_idx, filter_id in enumerate(self.database_filtered_ids):
                xdata = [self.database[point_id]['pca_point'][0] for point_id in self.database_filtered_ids[filter_id]]
                ydata = [self.database[point_id]['pca_point'][1] for point_id in self.database_filtered_ids[filter_id]]
                self.canvas.add_scatter(x = xdata,
                                y = ydata,
                                **self.config.canvas_data)
                self.canvas.data[filter_idx].marker.color = self.config.marker_colorlist[filter_id]
                self.canvas.data[filter_idx].marker.symbol = self.config.marker_symbollist[filter_id]
                self.canvas.data[filter_idx].name = filter_id
                
                self.canvas.data[filter_idx].on_selection(self.selection_fn)
        
        elif self.visualisation_type_selector.value == 'UMAP':
            for filter_idx, filter_id in enumerate(self.database_filtered_ids):
                xdata = [self.database[point_id]['umap_point'][0] for point_id in self.database_filtered_ids[filter_id]]
                ydata = [self.database[point_id]['umap_point'][1] for point_id in self.database_filtered_ids[filter_id]]
                self.canvas.add_scatter(x = xdata,
                                y = ydata,
                                **self.config.canvas_data)
                self.canvas.data[filter_idx].marker.color = self.config.marker_colorlist[filter_id]
                self.canvas.data[filter_idx].marker.symbol = self.config.marker_symbollist[filter_id]
                self.canvas.data[filter_idx].name = filter_id
                
                self.canvas.data[filter_idx].on_selection(self.selection_fn)
                
        else:
            dim0 = int(self.visualisation_type_selector.value.split('-')[0])
            dim1 = int(self.visualisation_type_selector.value.split('-')[1])
            for filter_idx, filter_id in enumerate(self.database_filtered_ids):
                xdata = [self.database[point_id]['point'][dim0] for point_id in self.database_filtered_ids[filter_id]]
                ydata = [self.database[point_id]['point'][dim1] for point_id in self.database_filtered_ids[filter_id]]
                self.canvas.add_scatter(x = xdata,
                                y = ydata,
                                **self.config.canvas_data)
                self.canvas.data[filter_idx].marker.color = self.config.marker_colorlist[filter_id]
                self.canvas.data[filter_idx].marker.symbol = self.config.marker_symbollist[filter_id]
                self.canvas.data[filter_idx].name = filter_id
                
                self.canvas.data[filter_idx].on_selection(self.selection_fn)
        
        return
    
    def on_visualisation_type_selector_change(self, change):
        if change['type'] == 'change' and change['name'] == 'value':
            self.reset_displayed_content()
            self.plot_latent_space_graph()
            
        
        
        
    def selection_fn(self, trace, points, selector):
        # if a new box is selected by the user, reset displayed content and set the current box ranges
        if (selector.xrange != self.selection['xrange']) and (selector.yrange != self.selection['yrange']):
            self.displayed_data['images'] = []
            self.displayed_data['captions'] = []
            self.selection['xrange'] = selector.xrange
            self.selection['yrange'] = selector.yrange
        # this function is called for each trace in self.canvas, but we add only selected points for visible traces to the displayed content
        if trace.visible:  
            self.displayed_data['images'] += [self.database[self.database_filtered_ids[trace.name][relative_point_id]]['image'] for relative_point_id in points.point_inds]
            self.displayed_data['captions'] += ['{}, id: {}'.format(self.database[self.database_filtered_ids[trace.name][relative_point_id]]['label'], self.database_filtered_ids[trace.name][relative_point_id]) for relative_point_id in points.point_inds]
                
        # finally once the last trace is called display the content
        if points.trace_index == len(self.database_filtered_ids)-1:
            self.image_list_widget.update(self.displayed_data['images'], self.displayed_data['captions'])
            
                    
        return
    

''' ---------------------------------------------------------------------------------
        MAIN GUI:
            - EXPERIMENT / REPETITION SELECTION WIDGET
            - LATENT SPACE VIEWER WIDGET
----------------------------------------------------------------------------------'''

class PlotRepresentationSpaceWidget(ipywidgets.Box):
    
    @staticmethod
    def get_default_gui_config():
        default_config = ad.Config()
        
        # general box layout config
        default_config.box_layout = ad.Config()
        default_config.box_layout.flex_flow = 'column'
        default_config.box_layout.display = 'flex'
        default_config.box_layout.align_items = 'stretch'
        
        # Row 0: Experiment/Repetition selection widget
        default_config.exp_rep_selection_widget = ad.gui.jupyter.ExperimentRepetitionLoaderWidget.get_default_gui_config()
        
        # Row 1: Latent Space viewer widget
        default_config.latent_space_viewer_widget = ad.gui.jupyter.plotrepresentationspacewidget.LatentSpaceViewerWidget.get_default_gui_config()

        return default_config


    def __init__(self, experiment_definitions, repetition_ids, pca_data, tsne_data, umap_data, experiment_statistics=None, filters=None, config=None, **kwargs):
        # set config with priority for args in kwards -> config -> default_config
        self.config = ad.config.set_default_config(kwargs, config, PlotRepresentationSpaceWidget.get_default_gui_config())

        self.experiment_definitions = experiment_definitions
        self.experiment_ids = [exp_def['id'] for exp_def in self.experiment_definitions]
        self.repetition_ids = repetition_ids
        self.pca_data = pca_data
        self.tsne_data = tsne_data
        self.umap_data = umap_data
        self.experiment_statistics = experiment_statistics
        self.filters = filters
        
        # GUI LAYOUT
        gui_elements = []

        ## Row 0
        self.exp_rep_selection_widget = ad.gui.jupyter.ExperimentRepetitionLoaderWidget(
            self.experiment_definitions,
            self.repetition_ids,
            self.load_database,
            config= self.config.exp_rep_selection_widget
        )
        gui_elements.append(self.exp_rep_selection_widget)

        ## Row 1
        self.latent_space_viewer_widget = LatentSpaceViewerWidget(config=self.config.latent_space_viewer_widget)
        gui_elements.append(self.latent_space_viewer_widget)
        
        super().__init__(
            gui_elements,
            layout=self.config.box_layout
        )

        # Initialise database
        self.load_database(self.experiment_ids[0], self.repetition_ids[0])


    def load_database(self, experiment_id, repetition_id):
        """ Loads the database

        Database is reset for each new (experiment,repetition) and contains one item per experimental run with:
            - point: coordinates of the outcome in the goal space
            - tsne_point: coordinates of the outcome in the tsne space of the goal space
            - pca_point: coordinates of the outcome in the pca space of the goal space
            - umap_point: coordinates of the outcome in the umap space of the goal space
            - image: the final observation (PIL image format)
            - label: label according to given filters (filters must define a partition of the set of results)
        """
        
        exp_idx = self.experiment_ids.index(experiment_id)
        if repetition_id is None:
            path = self.experiment_definitions[exp_idx]['directory']
        else:
            path = os.path.join(
                self.experiment_definitions[exp_idx]['directory'],
                'repetition_{:06d}'.format(int(repetition_id))
            )

        path = os.path.join(path, 'statistics')
        final_observation_path = os.path.join(path, 'final_observation')
        representation_path = os.path.join(path, 'representations')

        self.database = collections.OrderedDict()

        # Recover representations points (from npz compression)
        representations = dict(np.load(representation_path + '.npz'))
        for subrepresentation_name, subrepresentation_val in representations.items():
            if len(subrepresentation_val.shape) == 0:
                representations[subrepresentation_name] = subrepresentation_val.dtype.type(subrepresentation_val)

        for item_id in range(len(representations['gs_00'])):
            self.database[item_id] = dict()
            self.database[item_id]['point'] = representations['gs_00'][item_id]
            self.database[item_id]['tsne_point'] = self.tsne_data[exp_idx][repetition_id][item_id]
            self.database[item_id]['pca_point'] = self.pca_data[exp_idx][repetition_id][item_id]
            self.database[item_id]['umap_point'] = self.umap_data[exp_idx][repetition_id][item_id]
            
        # Recover images (from png or zip)
        if os.path.isdir(final_observation_path):
            dir_content = os.listdir(final_observation_path)
            for image_file_name in dir_content:
                file_path = os.path.join(path, image_file_name)
                if os.path.isfile(file_path) and '.png' in image_file_name:
                    item_id = image_file_name.split('.')[0]
                    item_id = int(item_id)
                    file = open(file_path, "rb")
                    image_PIL = Image.open(file)
                    with BytesIO() as output:
                        image_PIL.save(output, 'png')
                        self.database[item_id]['image'] = output.getvalue()

        elif zipfile.is_zipfile(final_observation_path + '.zip'):
            zf = zipfile.ZipFile(final_observation_path + '.zip', 'r')
            dir_content = zf.namelist()

            for image_file_name in dir_content:
                item_id = image_file_name.split('.')[0]
                item_id = int(item_id)
                with zf.open(image_file_name) as file:
                    image_PIL = Image.open(file)
                    self.database[item_id]['image'] = image_PIL
                    '''
                    image_array = np.array(image_PIL)
                    image_array = np.uint8(image_array.astype(float)/255.0 * 252.0)
                    image_PIL = Image.fromarray(image_array)
                    image_PIL.putpalette(colormaps[0])
                    with BytesIO() as output:
                        image_PIL.save(output, 'png')
                        self.database[item_id]['image'] = output.getvalue()
                    '''
       
        else:
            dir_content = []
            warnings.warn('No Images found under {!r}'.format(final_observation_path))
        
        if not dir_content.__len__() == self.database.__len__():
                warnings.warn('The number of final observation runs from {!r} differs from the number of representation runs from {!r} '.format(final_observation_path, representation_path))
            
        # Recover labels
        for curr_filter_k, curr_filter_val in self.filters.items():
            # filter
            if curr_filter_val is None or not curr_filter_val:
                img_ids = range(len(self.database))
            else:
                run_ids = ad.gui.jupyter.misc.filter_single_experiment_data(self.experiment_statistics[experiment_id], curr_filter_val, int(repetition_id))
                img_ids = np.where(run_ids)[0]
                
            # add label information to database
            for img_id in img_ids:
                img_id = int(img_id)
                if img_id in self.database:
                    self.database[img_id]['label'] = curr_filter_k
        
        # Update Latent Space Viewer with the database
        self.latent_space_viewer_widget.update(self.database)
        
    
