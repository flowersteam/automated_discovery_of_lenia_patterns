import autodisc as ad
import plotly
import zipfile
import os
import numpy as np
from PIL import Image
import plotly.graph_objs as go
import warnings
import random
from autodisc.gui.jupyter.misc import create_colormap, transform_image_from_colormap

#def plot_discoveries_treemap(experiment_definitions=None, repetition_ids=None, experiment_statistics=None, data_filters=None, config=None, **kwargs):
def plot_discoveries_treemap(experiment_definition, repetition_id=0, experiment_statistics=None, data_filters=None, config=None, **kwargs):
    
    default_config = dict(
            random_seed = 0,
            
            squarify = dict(
                        x = 0.0,
                        y = 0.0,
                        width = 100.0,
                        height = 100.0,
                    
                    ),

            # global style config
            global_layout = dict(
                    height=700, 
                    width=700,
                    margin = dict(
                            l = 0,
                            r = 0,
                            b = 0,
                            t = 0
                            ),
                    xaxis=dict(
                        autorange=True,
                        showgrid=False,
                        zeroline=False,
                        showline=False,
                        ticks='',
                        showticklabels=False
                    ),
                    yaxis=dict(
                        autorange=True,
                        showgrid=False,
                        zeroline=False,
                        showline=False,
                        ticks='',
                        showticklabels=False
                    ),
                    title = dict(
                            text = '',
                            font = dict(
                                  color = "black",
                                  size = 22,
                                  family='Times New Roman'
                                )
                            ),
                    hovermode='closest',
                    showlegend =  True,
                ),
            
            # Shapes style config
            shapes = dict(
                    line = dict(
                            width = 2
                            ),
                    layer = "below"
                    ),
            shapes_background_colors = ['rgb(0,0,0)', 'rgb(204,121,167)', 'rgb(0,114,178)'],
            shapes_lines_colors = [ 'rgb(0,0,0)','rgb(154,71,117)', 'rgb(0,64,128)'],
            
            # Images style config
            margin_x = 1,
            margin_y = 1, 
            images = dict (
                        xref= "x",
                        yref= "y",
                        sizex= 10,
                        sizey= 10,
                        opacity = 1,
                        xanchor= "center",
                        yanchor= "middle",
                        layer = "above"
                    ),
            images_transform = True,
            images_transform_colormaps = [
                            create_colormap(np.array([[255,255,255], [119,255,255],[23,223,252],[0,190,250],[0,158,249],[0,142,249],[81,125,248],[150,109,248],[192,77,247],[232,47,247],[255,9,247],[200,0,84]])/255*8), #WBPR
                			create_colormap(np.array([[0,0,4],[0,0,8],[0,4,8],[0,8,8],[4,8,4],[8,8,0],[8,4,0],[8,0,0],[4,0,0]])), #BCYR
                			create_colormap(np.array([[0,2,0],[0,4,0],[4,6,0],[8,8,0],[8,4,4],[8,0,8],[4,0,8],[0,0,8],[0,0,4]])), #GYPB
                			create_colormap(np.array([[4,0,2],[8,0,4],[8,0,6],[8,0,8],[4,4,4],[0,8,0],[0,6,0],[0,4,0],[0,2,0]])), #PPGG
                			create_colormap(np.array([[4,4,6],[2,2,4],[2,4,2],[4,6,4],[6,6,4],[4,2,2]])), #BGYR
                			create_colormap(np.array([[4,6,4],[2,4,2],[4,4,2],[6,6,4],[6,4,6],[2,2,4]])), #GYPB
                			create_colormap(np.array([[6,6,4],[4,4,2],[4,2,4],[6,4,6],[4,6,6],[2,4,2]])), #YPCG
                			create_colormap(np.array([[8,8,8],[7,7,7],[5,5,5],[3,3,3],[0,0,0]]), is_marker_w=False), #W/B
                			create_colormap(np.array([[0,0,0],[3,3,3],[5,5,5],[7,7,7],[8,8,8]]))], #B/W
            
            # Annotations style config
            annotations = dict(
                    font = dict(
                      color = "#140054",
                      size = 18,
                      family='Times New Roman'
                    ),
                    showarrow = False,
                    opacity=1.0,
                    bgcolor='rgb(255,255,255)'
                    )
            
            
            )
    
    config = ad.config.set_default_config(kwargs, config, default_config)
    
    random.seed(config.random_seed)
    
    # experiment_ids=[exp_def['id'] for exp_def in experiment_definitions]
    # for experiment_id in experiment_ids:
    #     experiment_idx = experiment_ids.index(experiment_id)
    #     for repetition_idx in repetition_ids:
    #         experiment_statistics[experiment_id][repetition_idx] = ad.gui.jupyter.misc.load_statistics(os.path.join(experiment_definitions[experiment_idx]['directory'], 'repetition_{:06d}'.format(repetition_idx)))

    experiment_id = experiment_definition['id']
    #exp_idx = experiment_ids.index(experiment_id)
    if repetition_id is None:
        path = experiment_definition['directory']
    else:
        path = os.path.join(
            experiment_definition['directory'],
            'repetition_{:06d}'.format(int(repetition_id))
        )
    
    path = os.path.join(path, 'statistics')
    final_observation_path = os.path.join(path, 'final_observation')

    # Recover images (from png or zip)
    all_images = []
    if os.path.isdir(final_observation_path):
        dir_content = os.listdir(final_observation_path)
        for image_file_name in dir_content:
            file_path = os.path.join(path, image_file_name)
            if os.path.isfile(file_path) and '.png' in image_file_name:
                item_id = image_file_name.split('.')[0]
                item_id = int(item_id)
                file = open(file_path, "rb")
                image_PIL = Image.open(file)
                if config.images_transform:
                    image_PIL = transform_image_from_colormap(image_PIL, config.images_transform_colormaps[0])
                all_images.append(image_PIL)
    
    elif zipfile.is_zipfile(final_observation_path + '.zip'):
        zf = zipfile.ZipFile(final_observation_path + '.zip', 'r')
        dir_content = zf.namelist()
        for image_file_name in dir_content:
            item_id = image_file_name.split('.')[0]
            item_id = int(item_id)
            with zf.open(image_file_name) as file:
                image_PIL = Image.open(file)
                if config.images_transform:
                    image_PIL = transform_image_from_colormap(image_PIL, config.images_transform_colormaps[0])
                all_images.append(image_PIL)

    else:
        warnings.warn('No Images found under {!r}'.format(final_observation_path))
        
    # Recover labels
    filters_ids = list(data_filters.keys())
    database_filtered_ids = dict()
    for curr_filter_k, curr_filter_val in data_filters.items():
        # filter
        if curr_filter_val is None or not curr_filter_val:
            img_ids = range(len(all_images))
        else:
            run_ids = ad.gui.jupyter.misc.filter_single_experiment_data(experiment_statistics[experiment_id], curr_filter_val, int(repetition_id))
            img_ids = np.where(run_ids)[0]
            database_filtered_ids[curr_filter_k] = list(img_ids)  

    values = []
    for filter_id in filters_ids:
        values.append(len(database_filtered_ids[filter_id])/len(all_images)*100)
    sorted_indexes = np.argsort(-np.array(values))
    
    values = np.array(values)[sorted_indexes]
    filters_ids = np.array(filters_ids)[sorted_indexes]
    config.shapes_background_colors = np.array(config.shapes_background_colors)[sorted_indexes]
    config.shapes_lines_colors = np.array(config.shapes_lines_colors)[sorted_indexes]
    
    normed = ad.gui.jupyter.squarify.normalize_sizes(values, config.squarify.width, config.squarify.height)
    rects = ad.gui.jupyter.squarify.squarify(normed, config.squarify.x, config.squarify.y, config.squarify.width, config.squarify.height)
    
    annotations = []
    shapes = []
    images = []
    counter = 0
    
    for r in rects:
        # annotations layout
        annotation_config = ad.config.set_default_config(
                    dict(
                        x = r['x']+(r['dx']/2),
                        y = r['y']+(r['dy']/2),
                        text = "<b>{}:<br>{:.1f}%<b>".format(filters_ids[counter],values[counter]),
                        ),
                    config.annotations
                    )
        annotations.append(annotation_config)
        
        # shapes layout
        shape_config = ad.config.set_default_config(
                    dict(
                        type = 'rect', 
                        x0 = r['x'], 
                        y0 = r['y'], 
                        x1 = r['x']+r['dx'], 
                        y1 = r['y']+r['dy'],
                        fillcolor = config.shapes_background_colors[counter],
                        line = dict(color = config.shapes_lines_colors[counter])
                        ),
                    config.shapes  
                    )
        shapes.append(shape_config)
        
        # images layout
        x0 = r['x']
        y0 = r['y']
        w = r['dx']
        h = r['dy']
        
        n_cols = int((w - 2 * config.margin_x) // config.images.sizex)
        space_x = (w - 2 * config.margin_x) / n_cols
        centers_x = []
        for j in range(n_cols):
            centers_x.append(j * space_x + space_x / 2.0)
        
        n_rows = int((h - 2 * config.margin_y) // config.images.sizey)
        space_y = (h - 2 * config.margin_y) / n_rows
        centers_y = []
        for i in range(n_rows):
            centers_y.append(i * space_y + space_y / 2.0)
        
        list_of_random_items = random.sample(database_filtered_ids[filters_ids[counter]], n_rows*n_cols)
        
        for i in range(n_rows):
            for j in range(n_cols):
                image_config = ad.config.set_default_config(
                        dict(
                            source = all_images[list_of_random_items[np.ravel_multi_index((i, j), dims=(n_rows,n_cols), order='F')]],
                            x = x0 + config.margin_x + centers_x[j],
                            y = y0 + config.margin_y + centers_y[i],
                        ),
                        config.images)
                images.append(image_config)
        
        counter = counter + 1
        if counter >= len(config.shapes_background_colors):
            counter = 0
    

    # append to global layout
    global_layout = ad.config.set_default_config(
                dict(
                    annotations = annotations,
                    shapes=shapes,
                    images=images
                    ),
                config.global_layout
                )
                        
    
    figure = dict(data=[go.Scatter()],layout=global_layout)
    
    
    plotly.offline.iplot(figure)
    
    return figure