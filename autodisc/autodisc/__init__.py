import autodisc.core
import autodisc.classifier
import autodisc.config
import autodisc.gui
import autodisc.cppn
import autodisc.helper
import autodisc.systems
import autodisc.explorers
import autodisc.representations
from autodisc.explorationdatahandler import ExplorationDataHandler, DataEntry
from autodisc.config import Config

# version meaning: <major-release>.<non-compatible-update>.<compatible-update>
#  - major-release: Major new realeases
#  - non-compatible-update: Changes which do not allow to run previous experiments with the framework.
#  - compatible-update: Changes which allow to run previous experiments with the framework
__version__ = '0.7.2'