import autodisc as ad
from autodisc.gui.gui import ArrayToImageGUI
try:
    import tkinter as tk
except:
    import Tkinter as tk


class ObservationPreviewGUI(ArrayToImageGUI):

    @staticmethod
    def default_gui_config():

        def_config = ad.gui.ArrayToImageGUI.default_gui_config()

        def_config.dialog.title = 'Observation Preview'
        def_config.steps = [[0, 1/4, 1/2, 3/4, -1]]

        return def_config


    def display_exploration_data(self, exploration, run_id):
        self.display_data(observations=exploration[run_id].observations.states)


    def display_data(self, observations):

        self.obs = observations

        # select the wanted images and display them

        image_data = []
        image_names = []

        for row_def in self.gui_config.steps:

            row_steps = []

            if not isinstance(row_def, list):
                row_def = [row_def]

            for step_def in row_def:

                if isinstance(step_def, int):
                    # if step definition is an integer -> defines step
                    if step_def >= 0:
                        row_steps.append(step_def)
                    else:
                        # if negative, count from the end
                        # the step is redefined to have the true step number as label
                        row_steps.append(len(self.obs) + step_def)

                else:
                    # if step definition is a float between 0 and 1 use it as a percentage to find the true step
                    row_steps.append(int((len(self.obs)-1) * step_def))

            image_data.append( [self.obs[idx] for idx in row_steps] )
            image_names.append( ['step ' + str(step_idx) for step_idx in row_steps] )

        super().display_data(image_data, image_names)