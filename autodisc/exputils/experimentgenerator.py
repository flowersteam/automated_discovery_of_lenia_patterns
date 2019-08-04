import os
import re
import copy
import shutil
from exputils.odsreader import ODSReader
from collections import OrderedDict

def generate_experiment_files(ods_filepath, directory=None, extra_files=None, extra_experiment_files=None, verbose=False):
    '''
    Generates experiment files and configurations based on entries in a ODS file (LibreOffice Spreadsheet).

    The ODS has to be in a specific form.
    Sheets define group of experiment for which an extra subfolder in the output directory will be generated.

    For an example of the ods format see the pytest files.

    :param ods_filepath: Path to the ODS file.
    :param directory: Directory where the experiments are generated.
    :param extra_files: Files that are mentioned in the ODS file but should be added to each experiment folder.
    '''

    if directory is None:
        directory = os.path.dirname(ods_filepath)

    if directory == '':
        directory = '.'

    if verbose:
        print('Load config from {!r} ...'.format(ods_filepath))

    config_data = load_configuration_data_from_ods(ods_filepath)

    # generate experiment files based on the loaded configurations
    if verbose:
        print('Generate experiments ...'.format(ods_filepath))

    generate_files_from_config(config_data, directory, extra_files=extra_files, extra_experiment_files=extra_experiment_files, verbose=verbose)


def load_configuration_data_from_ods(ods_filepath):
    '''

    :param ods_filepath:
    :return:
    '''

    config_data = []

    doc = ODSReader(ods_filepath, clonespannedcolumns=False)
    for sheet_name, sheet_data in doc.sheets.items():

        experiments_data = dict()

        experiments_data['directory'] = sheet_name
        experiments_data['experiments'] = OrderedDict()

        file_config = []

        parameters_start_idx = None
        parameters_end_idx = None

        file_borders = []

        variable_names = []

        #######################################################
        # first row: search for columns that describe parameters
        row_idx = 0

        for col_idx, col_data in enumerate(sheet_data[row_idx]):
            if col_data is not None:
                if col_data.lower() == 'parameters':
                    if parameters_start_idx is None:
                        parameters_start_idx = col_idx
                else:
                    parameters_end_idx = col_idx - 1
                    break

        #######################################################
        # second row: identify the template files
        row_idx = 1

        col_idx = parameters_start_idx
        stop = False

        previous_template_file_path = ''
        while not stop:

            if sheet_data[row_idx][col_idx] is not None and sheet_data[row_idx][col_idx] != previous_template_file_path:
                file_config.append(dict())
                file_config[-1]['template_file_path'] = sheet_data[row_idx][col_idx]

                previous_template_file_path = sheet_data[row_idx][col_idx]

                if file_borders:
                    file_borders[-1][1] = col_idx-1
                file_borders.append([col_idx, None])

            col_idx += 1
            if col_idx >= len(sheet_data[row_idx]) or (parameters_end_idx is not None and col_idx > parameters_end_idx):
                stop = True

        # in case the first line gives a definitive end, but not the second, then use the definitive end of the first line
        if parameters_end_idx is not None and file_borders[-1][1] is None:
            file_borders[-1][1] = parameters_end_idx

        #######################################################
        # third row: identify the file name templates
        row_idx = 2

        for file_idx in range(len(file_config)):
            file_config[file_idx]['file_name_template'] = sheet_data[row_idx][file_borders[file_idx][0]]

        #######################################################
        # fourth row: identify the variable names and if there are repetitions and a experiment folder
        row_idx = 3

        # find repetitions column
        repetitions_info_col_idx = None
        for col_idx in range(parameters_start_idx):
            if sheet_data[row_idx][col_idx] is not None and sheet_data[row_idx][col_idx].lower() == 'repetitions':
                repetitions_info_col_idx = col_idx
                break

        # find column with sources for repetitions
        repetition_source_file_locations_col_idx = None
        for col_idx in range(parameters_start_idx):
            if sheet_data[row_idx][col_idx] is not None and (sheet_data[row_idx][col_idx].lower() == 'files' or sheet_data[row_idx][col_idx].lower() == 'repetition files'):
                repetition_source_file_locations_col_idx = col_idx
                break

        # find column with sources for experiments
        experiment_source_file_locations_col_idx = None
        for col_idx in range(parameters_start_idx):
            if sheet_data[row_idx][col_idx] is not None and sheet_data[row_idx][col_idx].lower() == 'experiment files':
                experiment_source_file_locations_col_idx = col_idx
                break

        # find variable names
        for file_idx in range(len(file_config)):

            start_idx = file_borders[file_idx][0]
            end_idx = file_borders[file_idx][1]

            cur_variable_names = []

            col_idx = start_idx
            stop = False
            while not stop:
                cur_variable_names.append(sheet_data[row_idx][col_idx])

                col_idx += 1

                if col_idx < len(sheet_data[row_idx]) and sheet_data[row_idx][col_idx] is None:
                    file_borders[file_idx][1] = col_idx-1
                    stop = True

                elif col_idx >= len(sheet_data[row_idx]) or (end_idx is not None and col_idx > end_idx):
                    stop = True

            variable_names.append(cur_variable_names)

        #######################################################
        # remaining rows: experiments

        for row_idx in range(4, len(sheet_data)):

            experiment_id = sheet_data[row_idx][0]

            if experiment_id is not None:
                experiment_id = int(experiment_id)

                experiments_data['experiments'][experiment_id] = dict()

                experiments_data['experiments'][experiment_id]['files'] = copy.deepcopy(file_config)

                # repetitions info if it exists
                if repetitions_info_col_idx is not None and sheet_data[row_idx][repetitions_info_col_idx] is not None:
                    experiments_data['experiments'][experiment_id]['repetitions'] = int(sheet_data[row_idx][repetitions_info_col_idx])
                else:
                    experiments_data['experiments'][experiment_id]['repetitions'] = None

                if repetition_source_file_locations_col_idx is not None and sheet_data[row_idx][repetition_source_file_locations_col_idx] is not None:
                    experiments_data['experiments'][experiment_id]['repetition_source_file_locations'] = [i.strip() for i in sheet_data[row_idx][repetition_source_file_locations_col_idx].split(',')]
                else:
                    experiments_data['experiments'][experiment_id]['repetition_source_file_locations'] = None

                if experiment_source_file_locations_col_idx is not None and sheet_data[row_idx][experiment_source_file_locations_col_idx] is not None:
                    experiments_data['experiments'][experiment_id]['experiment_source_file_locations'] = [i.strip() for i in sheet_data[row_idx][experiment_source_file_locations_col_idx].split(',')]
                else:
                    experiments_data['experiments'][experiment_id]['experiment_source_file_locations'] = None

                for file_idx in range(len(file_config)):

                    col_idx = file_borders[file_idx][0]

                    experiments_data['experiments'][experiment_id]['files'][file_idx]['variables'] = dict()

                    for variable_name in variable_names[file_idx]:

                        experiments_data['experiments'][experiment_id]['files'][file_idx]['variables'][variable_name] = get_cell_data(sheet_data[row_idx][col_idx])

                        col_idx +=1

        config_data.append(experiments_data)

    return config_data


def get_cell_data(data):

    if data is None:
        data = ''

    # replace strange characters that are not used for python strings
    data = data.replace('’', '\'')
    data = data.replace('`', '\'')

    return data


def generate_files_from_config(config_data, directory='.', extra_files=None, extra_experiment_files=None, verbose=False):
    '''

    Format of configuration data:

    config_data['directory']
    config_data['experiments']: dictionary with keys=experiment id, values=description:


    config_data['ids']: Ids of the experiments
    config_data['files']:  List with configuration for each template file
        template_file['template_file_path']: filepath of the template file
        template_file['file_name_template']: template for the filename
        template_file['variables']: Dictionary with key=variable name, value=variable value

    :param config_data:
    :return:
    '''

    if extra_files is None:
        extra_files = []
    elif not isinstance(extra_files, list):
        extra_files = [extra_files]

    if extra_experiment_files is None:
        extra_experiment_files = []
    elif not isinstance(extra_experiment_files, list):
        extra_experiment_files = [extra_experiment_files]

    for experiment_group_config in config_data:

        # only create group folder if more than one sheet or the sheet name is not empty or 'Sheet1'
        if len(config_data) == 1 and experiment_group_config['directory'] in ['Sheet1', '']:
            group_directory = directory
        else:
            group_directory = os.path.join(directory, experiment_group_config['directory'])

        # create the folder if not exists
        if not os.path.isdir(group_directory):
            os.makedirs(group_directory)

        # generate the experiment folders and files
        for experiment_id, experiment_config in experiment_group_config['experiments'].items():

            experiment_directory = os.path.join(group_directory, 'experiment_{:06d}'.format(experiment_id))

            # create folders for the repetitions if necessary:
            if experiment_config['repetitions'] is None:
                num_of_repetitions = 1
            else:
                num_of_repetitions = experiment_config['repetitions']

            for repetition_id in range(num_of_repetitions):

                # only create repetition folders if repetitions are specified
                if experiment_config['repetitions'] is None:
                    experiment_files_directory = experiment_directory
                else:
                    experiment_files_directory = os.path.join(experiment_directory, 'repetition_{:06d}'.format(repetition_id))

                # create folder if not exists
                if not os.path.isdir(experiment_files_directory):
                    os.makedirs(experiment_files_directory)

                # generate the files for the experiment, or the repetition if they are defined
                if experiment_config['repetition_source_file_locations'] is None:
                    source_files = extra_files
                else:
                    source_files = experiment_config['repetition_source_file_locations'] + extra_files

                generate_source_files(source_files, experiment_files_directory, experiment_config, experiment_id, repetition_id)

            # if there are experiment - repetitions defined, then generate the files for the experiment folder
            if experiment_config['experiment_source_file_locations'] is None:
                source_files = extra_experiment_files
            else:
                source_files = experiment_config['experiment_source_file_locations'] + extra_experiment_files

            if source_files:
                generate_source_files(source_files, experiment_directory, experiment_config, experiment_id)


def generate_source_files(source_files, experiment_files_directory, experiment_config, experiment_id, repetition_id=None):

    # create the source files that are given by templates
    for file_config in experiment_config['files']:

        # check if the given file exists, otherwise it might be in one of the given directories under the source-files property of the experiment
        template_file_path = None
        if os.path.isfile(file_config['template_file_path']):
            template_file_path = file_config['template_file_path']
        else:
            for src in source_files:
                if os.path.isdir(src):
                    if os.path.isfile(os.path.join(src, file_config['template_file_path'])):
                        template_file_path = os.path.join(src, file_config['template_file_path'])
                        break

        if template_file_path is not None:

            # Read in the template file
            with open(template_file_path, 'r') as file:
                file_content = file.read()

            # Replace the variables
            file_content = file_content.replace('<experiment_id>', str(experiment_id))

            if repetition_id is not None:
                file_content = file_content.replace('<repetition_id>', str(repetition_id))

            for variable_name, variable_value in file_config['variables'].items():
                file_content = re.sub('<{}>'.format(variable_name),
                                      variable_value,
                                      file_content,
                                      flags=re.IGNORECASE)

            # Write the final output file
            file_path = os.path.join(experiment_files_directory, file_config['file_name_template'].format(experiment_id))
            with open(file_path, 'w') as file:
                file.write(file_content)

    # copy all other sources, but not the templates if they are in one of the source directories
    template_files = [file_config['template_file_path'] for file_config in experiment_config['files']]

    for src in source_files:
        copy_experiment_files(src, experiment_files_directory, template_files)


def copy_experiment_files(src, dst, template_files):

    if os.path.isdir(src):
        # if directory, then copy the content

        for item in os.listdir(src):
            s = os.path.join(src, item)

            # if subdirectory, then delete any existing directory and make a new directory

            if os.path.isdir(s):

                d = os.path.join(dst, item)

                if os.path.isdir(d):
                    shutil.rmtree(d, ignore_errors=True)

                os.mkdir(d)

            else:
                d = dst

            copy_experiment_files(s, d, template_files)

    else:
        # if file, then copy it directly

        # do not copy the template files, because they were already processed
        if os.path.basename(src) not in [os.path.basename(f) for f in template_files]:
            shutil.copy2(src, dst)



