import sys
import ast
import glob
import json
import functools
import subprocess

import numpy
from numpy.ma import masked_array
import matplotlib
from matplotlib import pyplot as plt

from scipy.spatial import KDTree

import ifcopenshell
import ifcopenshell.geom

id, fns = sys.argv[1:]

fns = ast.literal_eval(fns)

settings = ifcopenshell.geom.settings(
    USE_WORLD_COORDS=False
)
create_shape = functools.partial(ifcopenshell.geom.create_shape, settings)

settings_2d = ifcopenshell.geom.settings(
    INCLUDE_CURVES=True,
    EXCLUDE_SOLIDS_AND_SURFACES=True
)
create_shape_2d = functools.partial(ifcopenshell.geom.create_shape, settings_2d)

with open("mtl.mtl", 'w') as f:
    f.write("newmtl red\n")
    f.write("Kd 1.0 0.0 0.0\n\n")
    f.write("newmtl green\n")
    f.write("Kd 0.0 1.0 0.0\n\n")
    f.write("newmtl gray\n")
    f.write("Kd 0.6 0.6 0.6\n\n")

class ifc_element:
    valid = None
    
    def __init__(self, inst, geom):
        self.inst = inst
        self.geom = geom
        
        v = self.geom.geometry.verts
        d = self.geom.transformation.matrix.data
        
        self.M = numpy.concatenate((
            numpy.array(d).reshape(4, 3), 
            numpy.array([[0,0,0,1]]).T
        ), axis=1).T
        
        V = numpy.concatenate((
            numpy.array(v).reshape((-1, 3)),
            numpy.ones((len(v) // 3, 1))
        ), axis=1)
        
        bounds = numpy.vstack((
            numpy.min(V, axis=0),
            numpy.max(V, axis=0)))
            
        c = numpy.average(bounds, axis=0)
        c[2] = bounds[0][2]
            
        self.center = self.M @ c
        self.VV = numpy.array([self.M @ v for v in V])
        
        Mi = numpy.linalg.inv(self.M)
        
        self.pts = [
            self.center - Mi[1] * 0.2,
            self.center + Mi[1] * 0.2
        ]
    
    def draw(self, use_2d=True, color='black'):
        es = self.geom.geometry.edges
        vs = self.VV
        
        try:
            if not use_2d:
                raise RuntimeError("hack")
            plan = create_shape_2d(self.inst)
            es = plan.geometry.edges
            vs = numpy.array(plan.geometry.verts).reshape((-1, 3))
            vs = numpy.concatenate((
                vs,
                numpy.ones((len(vs), 1))
            ), axis=1)
            vs = numpy.array([self.M @ v for v in vs])
        except RuntimeError as e:
            # use 3d shape
            pass
        
        es = numpy.array(es).reshape((-1, 2))
        
        for ab in list(vs[:,0:2][es]):
            plt.plot(ab.T[0], ab.T[1], color=color, lw=0.5)
            
    def draw_arrow(self):
        st, en = self.pts
        en = en - st
        
        clr = {
            None: 'k',
            True: 'g',
            False: 'r'
        }[self.valid]
            
        plt.arrow(
            st[0], st[1], en[0], en[1],
            color=clr, head_width=0.15, head_length=0.2,
            length_includes_head=True
        )
        
    def draw_quiver(self, xya, filename):
        obj = open(filename, "w")
        obj_c = 1
        
        clr = {
            None: 'gray',
            True: 'green',
            False: 'red'
        }[self.valid]
        
        print('mtllib mtl.mtl\n', file=obj)
        print('usemtl %s\n' % clr, file=obj)
    
        z = self.center[2] + 0.5
        dists = numpy.linalg.norm(xya[:,0:2] - self.center[0:2], axis=1)
        samples = xya[dists < 1]
        ss = numpy.sin(samples.T[2])
        cs = numpy.cos(samples.T[2])
        m = numpy.ndarray((2,2))
        coords = numpy.array((
            ( 0.75,  0.5),
            (-0.75,  0.0),
            ( 0.75, -0.5),
            ( 0.25,  0.0)
        ))
        indices = numpy.array(((0,1,3),(1,2,3)))
        coords /= 20
        for xy, s, c in zip(samples.T[0:2].T, ss, cs):
            M = (c, -s), (s, c)
            for v in coords:
                vv = (M @ v) + xy
                print("v", *vv, z, file=obj)
            print("f", *(indices[0] + obj_c), file=obj)
            print("f", *(indices[1] + obj_c), file=obj)
            obj_c += 4
        
    def validate(self, tree, values):    
        
        def read(p):
            d, i = tree.query(p[0:3])
            if d > 0.4:
                raise ValueError("%f > 0.4" % d)
            return values[i]
        
        try:
            vals = list(map(read, self.pts))
            self.valid = vals[0] > vals[1]
        except ValueError as e:
            pass
        
inf = float("inf")

fs = list(map(ifcopenshell.open, fns))
doors = sum(map(lambda f: f.by_type("IfcDoor"), fs), [])
shapes = list(map(create_shape, doors))
objs = list(map(ifc_element, doors, shapes))
storeys = sum(map(lambda f: f.by_type("IfcBuildingStorey"), fs), [])
elevations = list(map(lambda s: getattr(s, 'Elevation', 0.), storeys))
elevations = sorted(set(elevations))
elevations[0] = -inf
elev_pairs = list(zip(elevations, elevations[1:] + [inf]))


walls = sum(map(lambda f: f.by_type("IfcWall"), fs), [])
wall_shapes = list(map(create_shape, walls))
wall_objs = list(map(ifc_element, walls, wall_shapes))

flow = numpy.genfromtxt('flow.csv', delimiter=',')
flow = flow[flow[:,3] != 1.]

def get_slice(data, min_z, max_z):
    data = data[data[:,2] >= min_z]
    data = data[data[:,2] < max_z]
    return data

def get_mean(data, spacing=10):
    ints = numpy.int_(data * spacing)
    mi = ints.min(axis=0)
    ma = ints.max(axis=0)
    sz = (ma - mi)[0:2] + 1
    arr = numpy.zeros(sz, dtype=float)
    counts = numpy.zeros(sz, dtype=int)
    ints2 = ints[:,0:2] - mi[0:2]
    for i,j,v in zip(ints2.T[0], ints2.T[1], data.T[3]):
        arr[i,j] += v
        counts[i,j] += 1
    arr[counts > 0] /= counts[counts > 0]
    
    x = (numpy.arange(sz[0]) + mi[0]) / spacing
    y = (numpy.arange(sz[1]) + mi[1]) / spacing
    
    xs, ys = numpy.meshgrid(y, x)
    
    return \
        masked_array(xs, counts == 0) , \
        masked_array(ys, counts == 0) , \
        masked_array(arr, counts == 0)


tree = KDTree(flow[:, 0:3])

plt.figure(figsize=(8,12))

norm = matplotlib.colors.Normalize(vmin=flow.T[3].min() / 10., vmax=flow.T[3].max() / 10.)

results = []

result_mapping = {}

for i, (mi, ma) in enumerate(elev_pairs):

    plt.clf()
    
    flow_mi_ma = get_slice(flow, mi - 1, ma)
    
    # plt.scatter(flow_mi_ma.T[0], flow_mi_ma.T[1], marker='s', s=1, norm=norm, c=(flow_mi_ma.T[3] -1.) / 10., label='distance from exterior')
    plt.gca().set_aspect('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.colorbar(fraction=0.04)
    
    x_y_angle = None
    
    if flow_mi_ma.size:
        x, y, arr = get_mean(flow_mi_ma)
        
        dx, dy = numpy.gradient(arr, 1. / 10)
        lens = numpy.sqrt(dx.data ** 2 + dy.data ** 2)
        too_long = lens > 120
        dx /= lens
        dy /= lens
        
        # for a_ in x, y, dx, dy:
        #     a_.mask[too_long] = True
            
        angles = numpy.arctan2(dy, dx)
        
        # somehow angles contains more non-zero masks.. nans?
        xf = x.data[~angles.mask]
        yf = y.data[~angles.mask]
        af = angles.data[~angles.mask]
        
        x_y_angle = numpy.column_stack((
            yf,
            xf,
            af
        ))
              
        # plt.quiver(y, x, -dx, -dy, scale=200,
        #     headwidth=3/1.5,
        #     headlength=5/3,
        #     headaxislength=4.5/3)
    
    for ob in objs:
        Z = ob.M[2,3] + 1.
        if Z < mi or Z > ma:
            continue
            
        ob.draw()
        ob.validate(tree, flow[:, 3])
        ob.draw_arrow()
        if ob in result_mapping:
            print("Warning element already emitted")
        else:
            N = len(result_mapping)
            fn = "%s_%d.obj" % (id, N)
            ob.draw_quiver(x_y_angle, fn)
            result_mapping[ob] = N
        
        
    for ob in wall_objs:
        Z = ob.M[2,3] + 1.
        if Z < mi or Z > ma:
            continue       
    
        ob.draw(use_2d=False, color='gray')
    
    plt.savefig("flow-%d.png" % i, dpi=600)


for ob in objs:
    st = {
        None: 'UNKNOWN',
        True: 'NOTICE',
        False: 'ERROR'
    }[ob.valid]
    
    desc = {
        "status": st,
        "guid": ob.inst.GlobalId
    }
    N = result_mapping.get(ob)
    
    if N is not None:
        desc["visualization"] = "/run/%s/result/resource/gltf/%d.glb" % (id, N)
        
    results.append(desc)

with open(id + ".json", "w") as f:
    json.dump({
        "id": id,
        "results": results
    }, f)


for fn in glob.glob("*.obj"):
    subprocess.check_call(["blender", "-b", "-P", "convert.py", "--", fn, fn.replace(".obj", ".dae")])
    subprocess.check_call(["COLLADA2GLTF-bin", "-i", fn.replace(".obj", ".dae"), "-o", fn.replace(".obj", ".glb"), "-b", "1"])
    