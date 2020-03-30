import yt
import numpy as np

arr = np.random.random(size=(64,64,64))

data = dict(density = (arr, "g/cm**3"))
bbox = np.array([[-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5]])
ds = yt.load_uniform_grid(data, arr.shape, length_unit="Mpc", bbox=bbox, nprocs=64)

slc = yt.SlicePlot(ds, "z", ["density"])
slc.set_cmap("density", "Blues")
slc.annotate_grids(cmap=None)
slc.show()

posx_arr = np.random.uniform(low=-1.5, high=1.5, size=10000)
posy_arr = np.random.uniform(low=-1.5, high=1.5, size=10000)
posz_arr = np.random.uniform(low=-1.5, high=1.5, size=10000)
data = dict(density = (np.random.random(size=(64,64,64)), "Msun/kpc**3"), 
            particle_position_x = (posx_arr, 'code_length'), 
            particle_position_y = (posy_arr, 'code_length'),
            particle_position_z = (posz_arr, 'code_length'))
bbox = np.array([[-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5]])
ds = yt.load_uniform_grid(data, data["density"][0].shape, length_unit=(1.0, "Mpc"), mass_unit=(1.0,"Msun"), 
                          bbox=bbox, nprocs=4)


slc = yt.SlicePlot(ds, "z", ["density"])
slc.set_cmap("density", "Blues")
slc.annotate_particles(0.25, p_size=12.0, col="Red")
slc.show()

