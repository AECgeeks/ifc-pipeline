import sys
import ast
import glob
import json
import functools

import numpy
from matplotlib import pyplot as plt

from scipy.spatial import KDTree

import ifcopenshell
import ifcopenshell.geom

id, fns  = sys.argv[1:]

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
        

fs = list(map(ifcopenshell.open, fns))
doors = sum(map(lambda f: f.by_type("IfcDoor"), fs), [])
shapes = list(map(create_shape, doors))
objs = list(map(ifc_element, doors, shapes))

walls = sum(map(lambda f: f.by_type("IfcWall"), fs), [])
wall_shapes = list(map(create_shape, walls))
wall_objs = list(map(ifc_element, walls, wall_shapes))

flow = numpy.genfromtxt('flow.csv', delimiter=',')

tree = KDTree(flow[:, 0:3])

plt.figure(figsize=(8,12))

results = []

for ob in objs:
    ob.draw()
    ob.validate(tree, flow[:, 3])
    ob.draw_arrow()
    
    st = {
        None: 'UNKNOWN',
        True: 'NOTICE',
        False: 'ERROR'
    }[ob.valid]
    
    results.append({
        "status": st,
        "guid": ob.inst.GlobalId
    })
    
for ob in wall_objs:
    ob.draw(use_2d=False, color='gray')

plt.scatter(flow.T[0], flow.T[1], marker='s', s=1, c=(flow.T[3] -1.) / 10., label='distance from exterior')
plt.gca().set_aspect('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(fraction=0.04)

plt.savefig("flow.png", dpi=600)

with open(id + ".json", "w") as f:
    json.dump({
        "bier": "lekker",
        "id": id,
        "results": results
    }, f)
