import yt
import numpy as np
from yt.visualization.volume_rendering.transfer_function_helper import TransferFunctionHelper
from yt.visualization.volume_rendering.api import Scene, VolumeSource

ds = yt.load("/Users/smajee/Downloads/IsolatedGalaxy/galaxy0030/galaxy0030")
sc = yt.create_scene(ds)

print (sc)

print(sc.get_source(0))

# yt.interactive_render(ds)
