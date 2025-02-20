import os
import numpy as np
from scipy import sparse
from molgri.space.fullgrid import FullGrid as fg

################################################################

def makeGrid(prefix: str, b_grid_name: str, o_grid_name: str, tgrid, factor=2, position_grid_cartesian=False, overwrite=False):
    assert prefix[-1] == '/', "Last char in prefix must be '/'"
    t_grid_name = "linspace(" + str(tgrid[0] /10) + "," + str(tgrid[1] / 10) + str(tgrid[2]) + ")"
    try:
        os.mkdir(prefix)
    except:
        if not overwrite:
            print("ABORTING: Folder '" + prefix +  "' already exists. Use 'overwrite = True' to overwrite data")
            return None
    grid = fg(b_grid_name, o_grid_name, t_grid_name, factor=factor, position_grid_cartesian=position_grid_cartesian)
    #sparse.save_npz(prefix + "adjacency_array", grid.get_full_adjacency())
    sparse.save_npz(prefix + "adjacency_only_position",grid.get_full_adjacency(only_position=True))
    #sparse.save_npz(prefix + "adjacency_only_orientation",grid.get_full_adjacency(only_orientation=True))
    #sparse.save_npz(prefix + "borders_array",grid.get_full_borders())
    sparse.save_npz(prefix + "distances_array",grid.get_full_distances())
    np.save(prefix + "_fullgrid", grid.get_full_grid_as_array())


makeGrid("tmp/60grid/","Cube4D_60","ico_60",[1,5,100])
makeGrid("tmp/noRotGrid/","zero","ico_150",[1,5,50])
makeGrid("data/noRotGrid/","zero","ico_150",[1,5,50])
makeGrid("tmp/noRotGridFine/","zero","ico_500",[0.1,5,100])
makeGrid("tmp/noRotGridAlt/","zero","ico_160",[0.01,5,20])



#fg1 = fg("zero","ico_60","linspace(0.01,0.5,100)")
#fg1.get_full_distances()
#a= fg1.get_full_grid_as_array()
#b = fg1.position_grid.get_position_grid_as_array()