import os
import sys
import ast
import sknw
import glob
import json
import functools
import itertools
import subprocess

import bisect
import operator
from collections import Counter, defaultdict
from functools import lru_cache

from dataclasses import dataclass
from typing import Any

import numpy
import skimage
from numpy.ma import masked_array

WITH_PLOT = False
if WITH_PLOT:
    import matplotlib
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
else:
    class whatever:
        def __getattr__(self, _):
            return self
        def __call__(self, *args, **kwargs):
            return self
        def __len__(self):
            return 1
        def __getitem__(self, _):
            return self
    plt = whatever()
    matplotlib = whatever()

import pandas as pd

import networkx as nx
from rdp import rdp
from scipy.spatial import KDTree
from scipy import ndimage
from skimage.morphology import skeletonize

from bresenham import bresenham

WITH_MAYAVI = False
if WITH_MAYAVI:
    from mayavi import mlab
    ax_3d = mlab.figure(size=(1600, 1200))

# mlab.options.offscreen = True

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element
import ifcopenshell.util.placement

import OCC.Core.BRep
import OCC.Core.BRepGProp
import OCC.Core.BRepTools
import OCC.Core.TopTools
import OCC.Core.TopoDS
import OCC.Core.TopExp
import OCC.Core.BRepPrimAPI

from OCC.Core.TopAbs import *

from ifc_utils import get_unit

normalize = lambda v: v / numpy.linalg.norm(v)


def yield_subshapes(s, type=None, vertices=False):
    """
    Based on type in `s` and additional keyword arguments
    initialize an OCCT iterator or explorer and yield its
    values.
    """
    if isinstance(s, OCC.Core.TopTools.TopTools_ListOfShape):
        it = OCC.Core.TopTools.TopTools_ListIteratorOfListOfShape(s)
    else:
        if type is None:
            if s.ShapeType() == TopAbs_WIRE:
                it = OCC.Core.BRepTools.BRepTools_WireExplorer(s)
            else:
                it = OCC.Core.TopoDS.TopoDS_Iterator(s)
        else:
            it = OCC.Core.TopExp.TopExp_Explorer(s, type)
    while it.More():
        if isinstance(it, OCC.Core.BRepTools.BRepTools_WireExplorer) and vertices:
            yield it.CurrentVertex()
        elif hasattr(it, 'Value'):
            yield it.Value()
        else:
            yield it.Current()
        it.Next()

        
def to_tuple(xyz):
    """
    Converts gp_Pnt/Vec/Dir(2d) to python tuple
    """
    return (xyz.X(), xyz.Y()) + ((xyz.Z(),) if hasattr(xyz, "Z") else ())


def wrap_try(fn):
    def inner(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except:
            return None
    return inner

id, fns, command, config = sys.argv[1:]

fns = ast.literal_eval(fns)
try:
    config = ast.literal_eval(config)
except:
    config = json.load(open(config))

LENGTH = config.get('length', 0.5)

settings = ifcopenshell.geom.settings(
    USE_WORLD_COORDS=False
)
create_shape = functools.partial(wrap_try(ifcopenshell.geom.create_shape), settings)

settings_2d = ifcopenshell.geom.settings(
    INCLUDE_CURVES=True,
    EXCLUDE_SOLIDS_AND_SURFACES=True
)
create_shape_2d = functools.partial(wrap_try(ifcopenshell.geom.create_shape), settings_2d)

with open("mtl.mtl", 'w') as f:
    f.write("newmtl red\n")
    f.write("Kd 1.0 0.0 0.0\n\n")
    f.write("newmtl green\n")
    f.write("Kd 0.0 1.0 0.0\n\n")
    f.write("newmtl gray\n")
    f.write("Kd 0.6 0.6 0.6\n\n")
    f.write("newmtl green2\n")
    f.write("Kd 0.6 1.0 0.6\n\n")
    f.write("newmtl red2\n")
    f.write("Kd 1.0 0.6 0.6\n\n")

class ifc_element:
    valid = None
    
    def __init__(self, inst, geom):
        self.inst = inst
        self.geom = geom
        
        self.is_external = False
        psets = ifcopenshell.util.element.get_psets(self.inst)
        for pn, pset in psets.items():
            if pn.endswith("Common"):
                self.is_external = pset.get('IsExternal', False) is True
        
        self.width, self.height = None, None
        if hasattr(inst, "OverallWidth") and inst.OverallWidth is not None:
            self.width = inst.OverallWidth * lu
        if hasattr(inst, "OverallHeight") and inst.OverallHeight is not None:
            self.height = inst.OverallHeight * lu
            
        self.M = None
        
        if self.geom is None:
            return
            
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
        
        self.bounds = numpy.vstack((
            numpy.min(V, axis=0),
            numpy.max(V, axis=0)))
            
        c = numpy.average(self.bounds, axis=0)
        c[2] = self.bounds[0][2]
            
        self.center = self.M @ c
        self.VV = numpy.array([self.M @ v for v in V])
        
        self.Mi = numpy.linalg.inv(self.M)
        
        # this is a bit of a hack. we do not really start from external anymore,
        # so sampling around the external doors is just sampling around the minimum
        # which is symmetric for small distances. The only true solution is to do
        # another traversal from the max value (masked by interior volume) to get
        # a monotonic function around the external doors (in the opposite direction).
        self.read_distance = 8 if self.is_external else 0.2
        
        self.pts = numpy.array([
            self.center - self.Mi[1] * self.read_distance,
            self.center + self.Mi[1] * self.read_distance
        ])
    
    def draw(self, ax, use_2d=True, color='black'):
        if self.geom is None:
            return
            
        es = self.geom.geometry.edges
        vs = self.VV
        
        try:
            if not use_2d:
                raise RuntimeError("hack")
            plan = create_shape_2d(self.inst)
            if plan is None:
                raise RuntimeError("hack")
            es_2d = plan.geometry.edges
            if len(es_2d) == 0:
                raise RuntimeError("hack")
            vs = numpy.array(plan.geometry.verts).reshape((-1, 3))
            vs = numpy.concatenate((
                vs,
                numpy.ones((len(vs), 1))
            ), axis=1)
            vs = numpy.array([self.M @ v for v in vs])
            es = es_2d
        except RuntimeError as e:
            # use 3d shape
            pass
        
        es = numpy.array(es).reshape((-1, 2))
        
        for ab in list(vs[:,0:2][es]):
            ax.plot(ab.T[0], ab.T[1], color=color, lw=0.5)
            
        ax.scatter(self.pts.T[0], self.pts.T[1], color='green', s=2)
            
        ax.text(self.center[0], self.center[1], f"#{self.inst.id()}")
            
    def draw_arrow(self, ax):
        if self.geom is None:
            return
            
        st, en = self.pts
        en = en - st
        
        clr = {
            None: 'k',
            True: 'g',
            False: 'r'
        }[self.valid]
            
        ax.arrow(
            st[0], st[1], en[0], en[1],
            color=clr, head_width=0.15, head_length=0.2,
            length_includes_head=True
        )
        
    def draw_quiver(self, flow, xya, filename):
        if self.geom is None:
            return
            
        obj = open(filename, "w")
        obj_c = 1
        
        clr = {
            None: 'gray',
            True: 'green',
            False: 'red'
        }[self.valid]
        
        print('mtllib mtl.mtl\n', file=obj)
        print('usemtl %s\n' % clr, file=obj)
    
        z = self.center[2] + 0.05
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
        coords /= (1. / flow.spacing) * 2
        coords *= 4
        for xy, s, c in zip(samples.T[0:2].T, ss, cs):
            M = (c, -s), (s, c)
            for v in coords:
                vv = (M @ v) + xy
                print("v", *vv, z, file=obj)
            print("f", *(indices[0] + obj_c), file=obj)
            print("f", *(indices[1] + obj_c), file=obj)
            obj_c += 4
            
            
    def outwards_direction(self):
        def read(p):
            return flow.lookup(p[0:3], max_dist=0.6 if self.read_distance < 1. else self.read_distance)
            # d, i = tree.query(p[0:3])
            # if d > 0.4:
            #     raise ValueError("%f > 0.4" % d)
            # return values[i]
        
        vals = list(map(read, self.pts))
        print(self.inst.id(), *vals)
        d = self.Mi[1][0:3].copy()
        if vals[0] < vals[1]:
            d[:] *= -1
        return d

        
    def validate(self, flow):    
        if self.geom is None:
            return
            
        def read(p):
            return flow.lookup(p[0:3], max_dist=0.4 if self.read_distance < 1. else self.read_distance)
            # d, i = tree.query(p[0:3])
            # if d > 0.4:
            #     raise ValueError("%f > 0.4" % d)
            # return values[i]
        
        try:
            vals = list(map(read, self.pts))
            print(self.inst.id(), *vals)
            self.valid = vals[0] > vals[1]
        except ValueError as e:
            pass
        
inf = float("inf")


fs = list(map(ifcopenshell.open, fns))
lu = get_unit(fs[0], "LENGTHUNIT", 1.0)
s = fs[0].by_type('IfcSite')[0]
z_offset = 0.
if s.ObjectPlacement and s.ObjectPlacement.PlacementRelTo:
    z_offset = s.ObjectPlacement.PlacementRelTo.RelativePlacement.Location.Coordinates[2] * lu
    print("z_offset", z_offset)
storeys = sum(map(lambda f: f.by_type("IfcBuildingStorey"), fs), [])
# elevations = list(map(lambda s: (lu * getattr(s, 'Elevation', 0.)) + z_offset, storeys))
def get_elevation_from_placement(s):
    return ifcopenshell.util.placement.get_local_placement(
        s.ObjectPlacement
    )[2,3] * lu
elevations = list(map(get_elevation_from_placement, storeys))
storey_by_elevation = dict(zip(elevations, storeys))
elevations = sorted(set(elevations))
elevations_2 = list(elevations)
elevations_2[0] = -inf
elev_pairs = list(zip(elevations_2, elevations_2[1:] + [inf]))

tree_settings = ifcopenshell.geom.settings(
    DISABLE_TRIANGULATION=True,
    DISABLE_OPENING_SUBTRACTIONS=True
)
tree = ifcopenshell.geom.tree()
# @todo speed up
# it = ifcopenshell.geom.iterator(tree_settings, fs[0], include=["IfcSpace"])
# tree.add_iterator(it)
for f in fs:
    tree.add_file(f, tree_settings)

class flow_field:
    
    def __init__(self, fn):
        self.flow = pd.read_csv(fn, delimiter=',').values
        # self.flow = self.flow[self.flow[:,3] != 1.]

        self.spacing = min(numpy.diff(sorted(set(self.flow.T[0]))))

        self.flow_ints = numpy.int_(self.flow / self.spacing)
        self.global_mi = self.flow_ints.min(axis=0)
        self.global_ma = self.flow_ints.max(axis=0)
        self.global_sz = (self.global_ma - self.global_mi)[0:2] + 1
        
        self.tree = KDTree(self.flow[:, 0:3])
        
        self.norm = None
        if "matplotlib" in globals():
            self.norm = matplotlib.colors.Normalize(
                vmin=self.flow.T[3].min() * self.spacing, 
                vmax=self.flow.T[3].max() * self.spacing)
        

    def get_slice(self, min_z, max_z):
        data = self.flow[self.flow[:,2] >= min_z]
        data =      data[     data[:,2] <= max_z]
        return data


    def get_mean(self, data, heights_as_int=False):
        ints = numpy.int_(data / self.spacing)
        arr = numpy.zeros(self.global_sz, dtype=float)
        counts = numpy.zeros(self.global_sz, dtype=int)
        highest = numpy.zeros(self.global_sz, dtype=int if heights_as_int else float)
        highest[:] = -1e5
        
        ints2 = ints[:,0:3] - self.global_mi[0:3]
        
        if heights_as_int:
            iterable = zip(*ints2.T[0:3], *data.T[3:4])
        else:
            iterable = zip(*ints2.T[0:2], *data.T[2:4])
        
        for i,j,z,v in iterable:
            if v > arr[i,j]:
                arr[i,j] = v
            counts[i,j] += 1
            if z > highest[i,j]:
                highest[i,j] = z
                
        # arr[counts > 0] /= counts[counts > 0]
        
        x = (numpy.arange(self.global_sz[0]) + self.global_mi[0]) * self.spacing
        y = (numpy.arange(self.global_sz[1]) + self.global_mi[1]) * self.spacing
        
        xs, ys = numpy.meshgrid(y, x)
        
        return \
            masked_array(xs, counts == 0)  , \
            masked_array(ys, counts == 0)  , \
            masked_array(arr, counts == 0) , \
            masked_array(highest, counts == 0)
            
    def lookup(self, xyz, max_dist = None):
         d, i = self.tree.query(xyz)
         if max_dist is not None and d > max_dist:
             raise ValueError("%f > %f" % (d, max_dist))
         return self.flow[i,3]

flow = flow_field('flow.csv')

def process_doors():

    results = []

    result_mapping = {}
    door_types = {}

    doors = sum(map(lambda f: f.by_type("IfcDoor"), fs), [])
    shapes = list(map(create_shape, doors))
    objs = list(map(ifc_element, doors, shapes))
    
    def get_decompositions(elem):
        has_any = False
        for rel in elem.IsDecomposedBy:
            for child in rel.RelatedObjects:
                yield from get_decompositions(child)
                has_any = True
        if not has_any:
            yield elem

    def flatmap(func, *iterable):
        return itertools.chain.from_iterable(map(func, *iterable))

    walls = sum(map(lambda f: f.by_type("IfcWall"), fs), [])
    walls = flatmap(get_decompositions, walls)
    wall_shapes = list(map(create_shape, walls))
    wall_objs = list(map(ifc_element, walls, wall_shapes))


    for storey_idx, (mi, ma) in enumerate(elev_pairs):
        
        flow_mi_ma = flow.get_slice(mi - 1, ma)
        
        figures = [plt.figure(figsize=(8,12)) for x_ in range(3)]
        axes = [f.add_subplot(111) for f in figures]
           
        for idx, (fig, ax) in enumerate(zip(figures, axes)):
            if idx != 2:
                # NotImplementedError: Axes3D currently only supports the aspect argument 'auto'. You passed in 'equal'.
                ax.set_aspect('equal')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            if idx == 0:
                sc = ax.scatter(flow_mi_ma.T[0], flow_mi_ma.T[1], marker='s', s=1, norm=flow.norm, c=(flow_mi_ma.T[3] -1.) * flow.spacing, label='distance from exterior')
                # fig.colorbar(sc, fraction=0.04, cax=ax)
            
        x_y_angle = None
        
        if flow_mi_ma.size:
            x, y, arr, heights = flow.get_mean(flow_mi_ma)
            
            dx, dy = numpy.gradient(arr, flow.spacing)
            
            # make sure that height differences are not resulting in connectivity
            # this doesn't work well, because at the stairs it will create create junctions to the corners
            # we need to later break edges at height differences instead
            # 
            # dxz, dyz = numpy.gradient(heights, spacing)
            # dZ = numpy.linalg.norm((dxz, dyz),axis=0)
            # obs = ~arr.mask
            # obs[dZ > 1] = False
            
            obs = ~arr.mask
                   
            ds = ndimage.distance_transform_edt(obs)
            
            lp = ndimage.filters.laplace(ds)
            maa = lp < lp.min() / 20.
            maa2 = skeletonize(maa)
                   
            ds_copy = ds.copy()
            ds_copy[arr.mask] = numpy.nan
            maa2[arr.mask] = numpy.nan
            
            # axes[2].imshow(heights.T, cmap='gray', alpha=0.25, extent=(y.data.min(), y.data.max(), x.data.min(), x.data.max()), origin='lower')
            # axes[2].imshow(ds_copy.T,  cmap='gray', alpha=0.5, extent=(y.data.min(), y.data.max(), x.data.min(), x.data.max()), origin='lower')
            
            axes[2].imshow(maa2.T,  cmap='gray', extent=(y.data.min(), y.data.max(), x.data.min(), x.data.max()), origin='lower')
            
            # img_xs, img_ys = numpy.meshgrid(
            #     numpy.linspace(y.min(), y.max(), heights.data.shape[0]),
            #     numpy.linspace(x.min(), x.max(), heights.data.shape[1]),
            # )
            # img_zs = numpy.ones(img_xs.shape) * mi
            # 
            # norm = matplotlib.colors.Normalize(ds[~arr.mask].min(), ds[~arr.mask].max())
            # s_map = plt.cm.ScalarMappable(norm=norm, cmap='gray')
            # fc = s_map.to_rgba(ds.T)
            # fc[:,:,3] = ~arr.mask.T
            # ax_3d.plot_surface(img_xs, img_ys, img_zs, rstride=1, cstride=1, facecolors=fc, shade=False, zorder=10 * i)
            
            x = x[::4, ::4]
            y = y[::4, ::4]
            arr = arr[::4, ::4]
            dx = dx[::4, ::4]
            dy = dy[::4, ::4]
            
            lens = numpy.sqrt(dx.data ** 2 + dy.data ** 2)
            # too_long = lens > 120
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
                  
            axes[1].quiver(y, x, -dx, -dy, scale=200,
                           headwidth=3/1.5,
                           headlength=5/3,
                           headaxislength=4.5/3)
        
            for ob in objs:
                if ob.M is None:
                    continue
                    
                Z = ob.M[2,3] + 1.
                if Z < mi or Z > ma:
                    continue
                    
                for ax in axes:
                    ob.draw(ax)
                    
                needs_validation = False
                is_valid = False
                doortype = "unknown"
                
                obtype = ifcopenshell.util.element.get_type(ob.inst)
                if obtype:
                    if "DOUBLE_SWING" in obtype.OperationType:
                        needs_validation = False
                        is_valid = True
                        doortype = "bidirectional"
                    elif "SLIDING" in obtype.OperationType:
                        needs_validation = False
                        is_valid = True
                        doortype = "slide"
                    elif "FOLDING" in obtype.OperationType:
                        needs_validation = False
                        is_valid = True
                        doortype = "fold"
                    elif "REVOLVING" == obtype.OperationType:
                        needs_validation = False
                        is_valid = True
                        doortype = "revolve"
                    elif "SWING" in obtype.OperationType:
                        needs_validation = True
                        doortype = "swing"
                    
                if needs_validation:
                    ob.validate(flow)
                    
                    # AttributeError: 'FancyArrow' object has no attribute 'do_3d_projection'
                    for ax in axes:
                        ob.draw_arrow(ax)
                else:
                    ob.valid = is_valid

                if ob in result_mapping:
                    print("Warning element already emitted")
                else:
                    N = len(result_mapping)
                    fn = "%s_%d.obj" % (id, N)
                    ob.draw_quiver(flow, x_y_angle, fn)
                    result_mapping[ob] = N
                    door_types[ob] = doortype
            
            
        for ob in wall_objs:
            Z = ob.M[2,3] + 1.
            if Z < mi or Z > ma:
                continue       
        
            for ax in axes:
                ob.draw(ax, use_2d=False, color='gray')
        
        for idx, fig in enumerate(figures):
            fig.savefig("flow-%d-%d.png" % (idx, storey_idx), dpi=150)
            
    for ob in objs:
        st = {
            None: 'UNKNOWN',
            True: 'NOTICE',
            False: 'ERROR'
        }[ob.valid]
        
        desc = {
            "status": st,
            "guid": ob.inst.GlobalId,
            "doorType": door_types.get(ob)
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

MAX_STEP_LENGTH = 1. / flow.spacing

from aabbtree import AABB, AABBTree
import networkx as nx

@dataclass(eq=True, frozen=True)
class riser:
    xy : Any
    norm: Any
    heights: Any
    width: Any
    
    @staticmethod
    def tupelize(a):
        if isinstance(a, numpy.ndarray):
            a = a.flatten().tolist()
        if isinstance(a, list):
            return tuple(a)
        return a
    
    def points_gen(self):
        nx = self.norm.copy()
        nx[:] = -nx[1], nx[0]
        for dx, dy, dz in itertools.product((-self.width / 2., +self.width / 2.), (0., MAX_STEP_LENGTH), self.heights):
            lxy = dx * nx + self.norm * dy + self.xy
            lxyz = numpy.concatenate((lxy, [dz]))
            yield (lxyz + flow.global_mi[0:3]) * flow.spacing
            
    @property
    def points(self):
        return numpy.array(list(self.points_gen()))
        
    @property
    def bottom_center(self):
        lxyz = numpy.concatenate((self.xy, [self.heights[0]]))
        return (lxyz + flow.global_mi[0:3]) * flow.spacing
        
    @property
    def center(self):
        lxyz = numpy.concatenate((self.xy, [(self.heights[0] + self.heights[1]) / 2.]))
        return (lxyz + flow.global_mi[0:3]) * flow.spacing
            
    def obj_export(self, obj):
        obj.write(self.points[[1,5,4,0]], [(0,1,2,3)])
        
    @property
    def aabb(self):
        pts = self.points
        return AABB(list(zip(pts.min(axis=0), pts.max(axis=0) + 0.1)))
        
    def to_tuple(self):
        return tuple(map(riser.tupelize, (
            self.xy,
            # exclude norm because it is fitted using pca and apparently not fully deterministic?
            # self.norm,
            self.heights,
            self.width
        )))

    def __hash__(self):
        # because ndarray is not hashable by default
        return hash(self.to_tuple())
        
    def __eq__(self, other):
        # because ndarray.__eq__ returns an ndarray
        return self.to_tuple() == other.to_tuple()


class obj_model:
    def __init__(self, fn):
        self.F = open(fn, "w")
        self.G = open(fn[:-4] + ".mtl", "w")
        self.vs = []
        self.fs = []
        self.N = 1
        self.object_count = 1
        print("mtlib", os.path.basename(fn)[:-4] + ".mtl", file=self.G)
        
    def add_material(self, nm, clr):
        print("newmtl", nm, file=self.G)
        print("Kd", *clr, file=self.G)
        
    def use_material(self, nm):
        print("usemtl", nm, file=self.F)
        
    def write(self, vs, idxs, g=None):
        if not idxs:
            return
            
        if g is None:
            g = "object-%003d" % self.object_count
            
        self.object_count += 1
            
        print("g", g, file=self.F)
        
        assert min(min(i) for i in idxs) >= 0
        assert max(max(i) for i in idxs) < len(vs)
        
        for v in vs:
            print("v", *v, file=self.F)
        for f in idxs:
            print("f", *(i + self.N for i in f), file=self.F)
            
        self.N += len(vs)

import scipy
from sklearn import decomposition

def process_risers():
    
    f = plt.figure(figsize=(12,12))
    
    mi = flow.flow_ints[:,2].min()
    ma = flow.flow_ints[:,2].max()
    stp = 10
    
    num_steps = int(numpy.ceil((ma-mi)/stp))
    
    risers = set()
    riser_list = []
    riser_tree = AABBTree()
    
    for ii in range(num_steps):
        i = mi + ii * stp
        
        # print(i)
    
        mask = numpy.logical_and(
            flow.flow_ints[:,2] >= i,
            flow.flow_ints[:,2] < i + stp + stp // 2
        )
        x, y, arr, heights = flow.get_mean(flow.flow[mask], heights_as_int=True)
        
        del x
        del y
        del arr
        
        print(f"[ {i}, {i + stp + stp // 2} )  ->  [ {heights.min()},  {heights.max()} ]")
        
        heights_clean = heights.copy()
        for h in numpy.unique(heights):
            hh = heights == h
            labels, n = ndimage.label(hh)
            for i in range(1, n+1):
                hhl = labels == i
                if numpy.count_nonzero(hhl) < 5:
                    heights_clean.mask[hhl] = True
               

        norm = matplotlib.colors.Normalize(vmin=mi, vmax=ma)
        
        # this does not work, diff is only 1d       
        # edges = numpy.abs(numpy.diff(heights_clean)) > (flow.spacing + 1.e-4)
        
        grad = numpy.gradient(heights_clean, 1.);
                
        # plt.clf()
        # plt.imshow(numpy.row_stack(grad), norm=norm)
        # plt.savefig("gradient-%02d.png" % ii, dpi=450)

        plt.clf()
        plt.imshow(heights + flow.global_mi[2], norm=norm)        
        
        edge_pts = numpy.column_stack(numpy.ma.where(
            numpy.logical_or(
                numpy.abs(grad[0]) > 1.,
                numpy.abs(grad[1]) > 1.
            )
        ))
        
        if edge_pts.size == 0:
            print("no edges...")
            continue       
        
        ept_tree = KDTree(edge_pts)
        pairs = numpy.array(list(ept_tree.query_pairs(numpy.sqrt(2.)+1.e-3)))
        del ept_tree
        
        # we no longer use these corners, but rather the connected component graph periphery
        # idxs, cnts = numpy.unique(pairs.flatten(), return_counts=True)
        # corners = edge_pts[idxs[cnts == 1]]
        # with gradient() we no longer add 0.5
        # plt.scatter(corners.T[1] + 0.5, corners.T[0] + 0.5, s=1)
        # plt.scatter(corners.T[1], corners.T[0], s=1)
        
        G = nx.Graph()
        G.add_edges_from(pairs)
        
        for nds in nx.connected_components(G):
        
            all_pts = numpy.array(list(map(edge_pts.__getitem__, nds)))
            plt.scatter(all_pts.T[1], all_pts.T[0], s=0.3)
            
            if len(nds) < 3:
                continue            
        
            cmp = G.subgraph(nds)
            cnt = nx.barycenter(cmp)
            cntxy = numpy.average(edge_pts[cnt], axis=0)# + 0.5
            
            nodes_from_center = list(set(sum([list(nx.dfs_preorder_nodes(cmp, source=c, depth_limit=4)) for c in cnt], [])))
            points_from_center = numpy.array(list(map(edge_pts.__getitem__, nodes_from_center)))
            pca = decomposition.PCA(n_components=2)
            pca.fit(points_from_center)
            dxy = pca.components_[1]
            dxy /= numpy.linalg.norm(dxy)

            a = numpy.int_(cntxy - dxy * 2)
            b = numpy.int_(cntxy + dxy * 2)
            
            ha = heights[a[0], a[1]]
            hb = heights[b[0], b[1]]
                                  
            if numpy.ma.masked in [ha, hb]:
                continue
                
            if ha > hb:
                dxy *= -1
                ha, hb = hb, ha

            plt.scatter(cntxy[1], cntxy[0], marker="*", s=1)
            
            ln = numpy.row_stack((cntxy, cntxy + dxy * 3))
            
            plt.plot(ln.T[1], ln.T[0], linewidth=0.2)
            
            pr = list(nx.periphery(cmp))
            
            pr_pts = numpy.array(list(map(edge_pts.__getitem__, pr)))
            plt.scatter(pr_pts.T[1], pr_pts.T[0], s=1, facecolors='none', edgecolors='r', linewidth=0.2)
            
            # use distance matrix, so that in case of multiple end points the largest pairwise distance can be picked
            L = numpy.max(scipy.spatial.distance_matrix(pr_pts, pr_pts))
            
            R = riser(cntxy, dxy, [ha, hb], L)
            
            # print(R.aabb)
            
            if R not in risers:
                risers.add(R)
                riser_tree.add(R.aabb, len(riser_list))
                riser_list.append(R)
                print(f"Adding {R}")
            else:
                print(f"{R} already present")
                pass
                
        plt.savefig("range-%02d.png" % ii, dpi=450)
    
    riser_graph = nx.Graph()
    
    for i,r in enumerate(riser_list):
        for j in riser_tree.overlap_values(r.aabb):
            if numpy.arccos(riser_list[i].norm @ riser_list[j].norm) > numpy.pi / 4:
                # incorporate an angle check to get consistent values due to rotated risers            
                continue
            riser_graph.add_edge(i, j)
    
    results = []
    
    for N, nds in enumerate([sg for sg in nx.connected_components(riser_graph) if len(sg) >= 3]):
        obj = obj_model("%s_%d.obj" % (id, N))
        obj.add_material("red",   (1.0, 0.0, 0.0))
        obj.add_material("green", (0.0, 1.0, 0.0))

        # Find relating element by bounding box search, not very robust,
        # awaiting change to IfcOpenShell to use exact distance
        c = Counter()

        for n in nds:
            p = riser_list[n].bottom_center
            pp = tuple(map(float, p))
            for inst in tree.select(pp, extend=0.1):
                if not inst.is_a("IfcSpace"):
                    c.update([inst])

        ifc_elem = c.most_common(1)[0][0] if len(c) else None
        
        st = "NOTICE" if ifc_elem and ifc_elem.is_a("IfcStair") or ifc_elem.is_a("IfcStairFlight") else "ERROR"
        obj.use_material("red" if st == "ERROR" else "green")
        
        for n in nds:
            riser_list[n].obj_export(obj)
            
        desc = {
            "status": st,
            "numRisers": len(nds),
            "visualization": "/run/%s/result/resource/gltf/%d.glb" % (id, N),
            "guid": ifc_elem.GlobalId if ifc_elem else None
        }
        
        results.append(desc)
            
    with open(id + ".json", "w") as f:
        json.dump({
            "id": id,
            "results": results
        }, f)
            
    # nx.draw(riser_graph)
    # plt.show()


class connectivity_graph:

    def __init__(self, G):
        self.G = G
        
        connected_nodes = []
        for comp in nx.connected_components(self.G):
            if len(comp) >= 3:
                connected_nodes.extend(comp)
                
        connected_nodes = set(connected_nodes)
        
        self.G = self.G.subgraph(connected_nodes)
        
    def get_node_z(self, n):
        attrs = self.G.nodes[n]
        LD = attrs['level']
        idx = numpy.int_(attrs['o'])
        return LD.height_map[idx[0], idx[1]]
    
    def get_node_xyz(self, n):
        attrs = self.G.nodes[n]
        xy = attrs['o']
        xyn = tuple(numpy.int16(xy))
        LD = attrs['level']
        dz = LD.height_map[xyn]
        xy2 = xy * flow.spacing + (LD.ymin, LD.xmin)
        xyz = tuple(xy2) + (dz,)
        return tuple(map(float, xyz))
        
    def get_edge_dz(self, se):
        attrs = self.G[se[0]][se[1]]
        ps = attrs['pts']
        LD = attrs['level']
        dz = LD.height_map[tuple(ps.T)]
        return dz.max() - dz.min()


    @lru_cache(maxsize=1024)
    def get_edge_points(self, se):
        attrs = self.G[se[0]][se[1]]
        ps = attrs['pts']

        st_o = self.G.nodes[se[0]]['o']
        en_o = self.G.nodes[se[1]]['o']

        if (numpy.linalg.norm(ps[ 0] - st_o) > numpy.linalg.norm(ps[-1] - st_o)) and \
           (numpy.linalg.norm(ps[-1] - en_o) > numpy.linalg.norm(ps[ 0] - en_o)):
                ps = ps[::-1, :]

        LD = attrs['level']
        dz = LD.height_map[tuple(ps.T)]
        ps2 = ps * flow.spacing + (LD.ymin, LD.xmin)
        ps3 = numpy.column_stack((ps2, dz))
        if dz.max() - dz.min() > 1.e-5:
            ps3 = stair_case(ps3)
        return ps3

    def __getattr__(self, k):
        return getattr(self.G, k)
        
    def draw_nodes(self, node_colouring=None):
            
        nodes = self.G.nodes()
        sorted_nodes = sorted(map(int, nodes))
        
        if node_colouring is not None:
            end_point_mask = numpy.array([x in node_colouring for x in sorted_nodes])
        # else:
        #     end_point_ids = {p.node_id for p in all_end_points}
        #     end_point_mask = numpy.array([x in end_point_ids for x in sorted_nodes])
        
        ps = numpy.array([nodes[i]['o'] for i in sorted_nodes])
        levels = [nodes[i]['level'] for i in sorted_nodes]
        offset = numpy.array([(l.ymin, l.xmin) for l in levels])
        psn = numpy.int16(ps)
        
        dz = numpy.zeros((len(nodes),))
        
        flow_dist = {}
        
        for p in sorted_nodes:
            flow_dist[p] = flow.lookup(self.get_node_xyz(p))

        for L in set(levels):
            mask = numpy.array([L == l for l in levels])
            dz[mask] = L.height_map[tuple(psn[mask].T)]
            
        ps2 = ps * flow.spacing + offset
        
        for N, xy, z, L in zip(sorted_nodes, ps2, dz, levels):
            fd = flow_dist.get(N)
            s = "    %d-%d" % (L.storey_idx, N)
            if fd:
                s += "(%d)" % fd
            mlab.text3d(*xy, z, s, figure=ax_3d, scale=0.05)
            
        if node_colouring is None:
            return mlab.points3d(*ps2.T, dz, color=(1.0,0.0,0.0), figure=ax_3d, scale_factor=0.1)
        
        mlab.points3d(*ps2[~end_point_mask].T, dz[~end_point_mask], color=(0.5,0.5,0.5), figure=ax_3d, scale_factor=0.1)
        mlab.points3d(*ps2[ end_point_mask].T, dz[ end_point_mask], color=(1.0,0.0,0.0), figure=ax_3d, scale_factor=0.1)
        
            
    def draw_edges(self, pts=None):
        if pts is not None:
            # max([numpy.linalg.norm(self.G.nodes[a]['o'] - self.G.nodes[b]['o']) for a,b in self.G.edges])
            
            # import pdb; pdb.set_trace()
            ls = numpy.array(self.G.edges())
            mapping = {b:a for a,b in enumerate(sorted(self.G.nodes()))}
            ls2 = numpy.vectorize(mapping.get)(ls)
            pts.mlab_source.dataset.lines = ls2
            tube = mlab.pipeline.tube(pts, tube_radius=0.02)
            mlab.pipeline.surface(tube, color=(0.8,0.8,0.8))
        else:
            for comp in nx.connected_components(self.G):
                if len(comp) < 3: continue
                for s,e in self.G.subgraph(comp).edges():
                    attrs = self.G[s][e]
                    ps = attrs['pts']
                    LD = attrs['level']
                    dz = LD.height_map[tuple(ps.T)]
                    ps2 = ps * flow.spacing + (LD.ymin, LD.xmin)
                    ps_3d = numpy.column_stack((ps2, dz))
                    ps3 = ps_3d
                    if dz.max() - dz.min() > 1.e-5:
                        ps3 = stair_case(ps3)
                    mlab.plot3d(*ps3.T, color=(1,1,1), tube_radius=None, figure=ax_3d)

                            
def path_to_edges(p):
    return [(p[i], p[i+1]) for i in range(len(p)-1)] 
    
# https://stackoverflow.com/a/3252222
def perp( a ) :
    b = numpy.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

def seg_intersect(a1,a2, b1,b2) :
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = numpy.dot( dap, db)
    num = numpy.dot( dap, dp )
    return (num / denom.astype(float))*db + b1
#####################################

# create nice staircases by splitting dZ of the edges from dXY
def stair_case(points):
    # compute edge vectors
    edges = numpy.roll(points, shift=-1, axis=0) - points
    
    def split_xy_z(x):
      # generator that splits dZ from dXY
      for y in x:
        if abs(y[2]) > 1e-5:
            zz = numpy.zeros_like(y)
            zz[2] = y[2]
            yield zz
            xyxy = y.copy()
            xyxy[2] = 0.
            yield xyxy
        else: yield y
    
    edges2 = numpy.array(list(split_xy_z(edges[:-1])))
    edges2_start = numpy.row_stack((points[0], edges2))
    
    # cumsum to apply deltas to start point
    return numpy.cumsum(edges2_start, axis=0)
    

@dataclass
class level_data:
    storey_idx : Any
    elev : Any
    height_map : Any
    xmin : Any
    ymin : Any
    
    def __hash__(self):
        return hash(self.storey_idx)


    def storey_inst(self):
        return storey_by_elevation.get(self.elev)


@dataclass
class end_point:
    node_id : int
    to_level : int
    from_level : int
    
    
SUPERSAMPLE = 2

def create_connectivity_graph():

    levels = list(elevations)

    for a,b in zip(elevations, elevations[1:]):
        if b - a > 2:
            levels.append((b+a)/2.)
            
    levels = [x + flow.spacing * 2 for x in sorted(levels)]
    ranges = []

    # Create until 2/3, trim at 1/2
    # for i in range(len(levels)):
    #     st, en, st2, en2 = None, None, None, None
    #     if i == 0:
    #         st = levels[i] - 1.5
    #         st2 = st + 0.5
    #     else:
    #         st = (levels[i] + levels[i-1] * 2.) / 3.
    #         st2 = (levels[i] + levels[i-1]) / 2.
    #     if i == len(levels) - 1:
    #         en = levels[i] + 1.5
    #         en2 = en - 0.5
    #     else:
    #         en = (levels[i] + levels[i+1] * 2.) / 3.
    #         en2 = (levels[i] + levels[i+1]) / 2.
    #     ranges.append((st, en, st2, en2))
        
    # Create until 2/3, trim at +0.5
    # Later we join trimmed edges with neighbouring levels
    for i in range(len(levels)):
        st, en, st2, en2 = None, None, None, None
        if i == 0:
            st = levels[i] - 1.5
            st2 = levels[i] - 0.7
        else:
            st = (levels[i] + levels[i-1] * 2.) / 3.
            st2 = levels[i] - 0.7
        if i == len(levels) - 1:
            en = levels[i] + 1.5
            en2 = levels[i] + 0.7
        else:
            en = (levels[i] + levels[i+1] * 2.) / 3.
            en2 = levels[i] + 0.7
        ranges.append((st, en, st2, en2))
        
    print("ranges")
    for r in ranges:
        print(*zip(("st", "en", "st2", "en2"), r))
        
    complete_graph = nx.Graph()


    def apply_end_point(mapping):
        return lambda ep: end_point(mapping[ep.node_id], ep.to_level, ep.from_level)

       
    def get_node_xyz(graph, LD0=None):
        def inner(ep):
            nid = ep if isinstance(ep, int) else ep.node_id
            xy = graph.nodes[nid]['o']
            xyn = tuple(numpy.int16(xy))
            LD = LD0 or graph.nodes[nid]['level']
            dz = LD.height_map[xyn]
            xy2 = xy * flow.spacing + (LD.ymin, LD.xmin)
            xyz = tuple(xy2) + (dz,)
            return xyz
        return inner


    all_end_points = []
    
    f = plt.figure()
    f.gca().hist(flow.flow.T[2], bins=64)
    f.gca().set_yscale('log')
    
    clrs = plt.get_cmap("Dark2").colors
    for storey_idx, (elev, (zmin, zmax, zmin_2, zmax_2)) in enumerate(zip(levels, ranges)):
        f.gca().axvline(zmin_2, ls='--', color=clrs[storey_idx % len(clrs)])
        f.gca().axvline(zmax_2, ls='--', color=clrs[storey_idx % len(clrs)])
        if numpy.any((numpy.array(elevations) - (elev - flow.spacing)) < 0.01):
            f.gca().axvline(elev, color=clrs[storey_idx % len(clrs)])
            f.gca().text(elev, f.gca().get_ylim()[0], "%.3f" % elev)
            
    
    f.savefig("z_histogram.png", dpi=300)
        
    for storey_idx, (elev, (zmin, zmax, zmin_2, zmax_2)) in enumerate(zip(levels, ranges)):
        
        end_points = []
        
        
        # We extend beyond zmin and zmax, but trim the graph edges:
        
        # assymetric more extension downwards, because it cannot obscure route
        # but also not too much, because it can create topological connections also
        flow_mi_ma = flow.get_slice(zmin, zmax)
        
        if flow_mi_ma.size:
                        
            x, y, arr, heights = flow.get_mean(flow_mi_ma)
            
            fn = "section-%05d-%05d-%05d.png" % (elev * 1000, zmin * 1000, zmax * 1000)
            fig = plt.figure(figsize=((y.data.max() - y.data.min()), (x.data.max() - x.data.min())))
            axx = fig.add_subplot(111)
            
            # print("x", x.min(), x.data.min())
            # print("y", y.min(), y.data.min())
           
            LD = level_data(storey_idx, elev, heights.data, x.data.min(), y.data.min())        
            
            dxz, dyz = numpy.gradient(heights, flow.spacing)
            dZ = numpy.linalg.norm((dxz, dyz), axis=0)
            obs = ~arr.mask
                                         
            # after distance?
            # no, after results in weird artefacts near the walls...
            # it really works better without, and then solve wrongly connected edges later.
            # obs[dZ > 3] = False
            
            if SUPERSAMPLE > 1:
                obs = skimage.transform.rescale(obs.astype(float)*100., SUPERSAMPLE, anti_aliasing=None, preserve_range=True) > 50.
            
            ds = ndimage.distance_transform_edt(obs)
            ds = ndimage.gaussian_filter(ds, sigma=3.)
            
            # import pdb; pdb.set_trace()
            
            # @todo would it help to blur a bit?                                                                                                                                     
            
            # results in weird artefacts near the walls...
            # ds[numpy.logical_and(dZ > 3, ~arr.mask)] = -1.

            ds_copy = ds.copy()
            ds_copy[~obs] = numpy.nan       
            
            # if storey_idx == 3:
            #     import pdb; pdb.set_trace()
            
            lp = ndimage.filters.laplace(ds)
                       
            maa = lp < lp.min() / 10.
            maa2 = skeletonize(maa)
            
            image_extents = (y.data.min(), y.data.max(), x.data.min(), x.data.max())
            image_extents_3d = image_extents + (elev, elev)
            
            # axx.imshow(ds_copy.T,  cmap='gray', alpha=0.5, extent=image_extents, origin='lower')
            axx.imshow(maa2.T,  cmap='gray', extent=image_extents, origin='lower')
            
            for bnd_idx, (clr, ctr) in enumerate(zip([(0,1,0,0.5), (1,0,0,0.5)], [zmin_2, zmax_2])):
                tmp = numpy.zeros(shape=tuple(arr.mask.shape) + (4,))
                if bnd_idx == 0:
                    tmp[numpy.logical_and(heights < ctr, ~arr.mask)] = clr
                else:
                    tmp[numpy.logical_and(heights > ctr, ~arr.mask)] = clr
                axx.imshow(tmp.transpose((1,0,2)), extent=image_extents, origin='lower')
            
        
            if WITH_MAYAVI:
                img = mlab.imshow(ds_copy[::-1, ::-1], colormap='gray', extent=image_extents_3d, figure=ax_3d, transparent=False, interpolate=False)
                img.module_manager.scalar_lut_manager.lut.nan_color = 0, 0, 0, 0
                img.update_pipeline()
                    
            
            graph = sknw.build_sknw(maa2)
            for n in graph.nodes:
                graph.nodes[n]['pts'] = graph.nodes[n]['pts'].astype(float) / SUPERSAMPLE
                graph.nodes[n]['o'] = graph.nodes[n]['o'].astype(float) / SUPERSAMPLE
            for e in graph.edges:
                graph.edges[e]['pts'] = graph.edges[e]['pts'].astype(float) / SUPERSAMPLE
            
            # import pdb; pdb.set_trace()
            
            # nodes_to_add = []
            # edges_to_add = []
            # edges_to_remove = []
            # 
            # for s,e in graph.edges():
            #     ps = graph[s][e]['pts']
            #     dz = heights.data[tuple(ps.T)]
            #     ddz = numpy.abs(numpy.diff(dz) / spacing)
            #     if numpy.where(ddz > 5)[0].size:
            #         
            #         i = numpy.argmax(ddz)
            #         n = len(graph.nodes())
            #         
            #         edges_to_remove.append((s, e))
            #         nodes_to_add.append((n, {"o":ps[i]}))
            #         edges_to_add.append(((s, n), {"pts":ps[:i-2]}))
            #         edges_to_add.append(((n, e), {"pts":ps[i+1:]}))
            #         
            # graph.remove_edges_from(edges_to_remove)
            # for n, kwargs in nodes_to_add:
            #     graph.add_node(n, **kwargs)
            # for ab, kwargs in edges_to_add:
            #     graph.add_edge(*ab, **kwargs)
            
            # retry again trimming to zmin, zmax
            
            # remove small components
            # for component in list(nx.connected_components(graph)):
            #     if len(component) < 3:
            #         for node in component:
            #             graph.remove_node(node)
            
            nodes_to_add = []
            edges_to_add = []
            edges_to_remove = []
            
            # new node index
            N = lambda: len(graph.nodes()) + len(nodes_to_add) # + 1
            
            # fig.savefig(fn[:-4] + "_0.png")
            
            for s,e in graph.edges():
                ps = graph[s][e]['pts']
                dz = heights.data[tuple(numpy.int_(ps.T))]
                # dzmin, dzmax = dz.min(), dz.max()
                within = numpy.logical_and(dz >= zmin_2, dz <= zmax_2)
                
                # under = dz < zmin_2 + flow.spacing * 2 + 1.e-3
                # above = dz > zmax_2 - flow.spacing * 2 - 1.e-3
                
                if not numpy.any(within):
                    # all outside of the [zmin, zmax] range
                    edges_to_remove.append((s, e))
                    
                elif not numpy.all(within):
                
                    # if storey_idx == 8:
                    #     import pdb; pdb.set_trace()
                    
                    # some outside of the [zmin, zmax] range
                    # import pdb; pdb.set_trace()
                    
                    # if {s,e} == {47,48}:
                    #     import pdb; pdb.set_trace()
                    
                    edges_to_remove.append((s, e))
                    
                    # get indices of transitions
                    breaks = numpy.concatenate((
                        # true -> false 
                        # ----
                        numpy.where(numpy.diff(numpy.int8(within)) == -1)[0],
                        # false -> true
                        #          ----
                        numpy.where(numpy.diff(numpy.int8(within)) == +1)[0] + 1,
                    ))
                    
                    bb = sorted(breaks)
                    
                    # add start if starts within range
                    if within[0]:
                        bb.insert(0, 0)
                        
                    # add end if odd number
                    if len(bb) % 2 == 1:
                        bb.append(ps.shape[0] - 1)
                        
                    # create pairs / intervals
                    bbb = [(bb[i], bb[i+1]) for i in range(0, len(bb), 2)]
                    for st_, en_ in bbb:
                    
                        if st_ == en_: continue
                    
                        # import pdb; pdb.set_trace()
                    
                        if st_ != 0:
                            STN = N()
                            st_ += 1
                            nodes_to_add.append((STN, {"o": ps[st_]}))
                            
                            # import pdb; pdb.set_trace()
                            
                            # if under[st_] or above[st_]:
                            #     tlvl = storey_idx - 1 if under[st_] else storey_idx + 1
                            #     end_points.append(
                            #         end_point(STN, tlvl, storey_idx)
                            #     )
                        else:
                            STN = s

                        if en_ != ps.shape[0] - 1:
                            ENN = N()
                            # @todo should this still be applied?
                            # en_ += 1
                            nodes_to_add.append((ENN, {"o": ps[en_]}))
                            
                            # import pdb; pdb.set_trace()
                            
                            # if under[en_] or above[en_] or under[en_+1] or above[en_+1]:
                            #     tlvl = storey_idx - 1 if (under[en_] or under[en_+1]) else storey_idx + 1
                            #     end_points.append(
                            #         end_point(ENN, tlvl, storey_idx)
                            #     )
                        else:
                            ENN = e
                            
                        # if len(ps[st_:en_+1]) == 0:
                        #     import pdb; pdb.set_trace()
                            
                        edges_to_add.append(((STN,ENN), {"pts": ps[st_:en_+1]}))
                    
                    # if breaks.size == 1:
                    #     st_ = breaks[0]
                    #     en_ = breaks[0] + 1
                    #     if within[0]:
                    #         edges_to_add.append(((s,N), {"pts": ps[:st_ + 1]}))
                    #     else:
                    #         edges_to_add.append(((N,e), {"pts": ps[en_:]}))
                    #     nodes_to_add.append((N, {"o": ps[st_]}))
                    # else:
                    #     import pdb; pdb.set_trace()
                    #     edges_to_remove.append((s, e))
                    
            print("removing", len(edges_to_remove), "edges, adding", len(edges_to_add))
            # print("found", len(end_points), "end points")
                    
            graph.remove_edges_from(edges_to_remove)
            for n, kwargs in nodes_to_add:
                graph.add_node(n, **kwargs)
            for ab, kwargs in edges_to_add:
                st_pos = graph.nodes[ab[0]]['o']
                en_pos = graph.nodes[ab[1]]['o']
                # if not numpy.all(st_pos == en_pos):
                #     import pdb; pdb.set_trace()
                    
                graph.add_edge(*ab, **kwargs)
            graph.remove_nodes_from(list(nx.isolates(graph)))
            
            for s,e in graph.edges():
            
                # if storey_idx == 2 and {s,e} == {18,19}:
                #     import pdb; pdb.set_trace()
            
                # i,j
                ps = graph[s][e]['pts']
                
                st_o = graph.nodes()[s]['o']
                en_o = graph.nodes()[e]['o']
                
                if (numpy.linalg.norm(ps[ 0] - st_o) > numpy.linalg.norm(ps[-1] - st_o)) and \
                   (numpy.linalg.norm(ps[-1] - en_o) > numpy.linalg.norm(ps[ 0] - en_o)):
                        ps = ps[::-1, :]
                            
                ps = numpy.int_(numpy.row_stack((
                    st_o,
                    ps,
                    en_o
                )))
                
                graph[s][e]['pts'] = ps
                
                edges = numpy.roll(ps, shift=-1, axis=0) - ps
                max_el = numpy.linalg.norm(edges[:-1], axis=1).max()

                # lookup z
                dz = heights.data[tuple(ps.T)]
                # i,j -> x,y
                ps2 = ps * flow.spacing + (y.data.min(), x.data.min())
                # x,y,z
                ps_3d = numpy.column_stack((ps2, dz))
                # simplify
                # ps3 = rdp(ps_3d, epsilon=flow.spacing/2.)
                ps3 = ps_3d
                
                if dz.max() - dz.min() > 1.e-5:
                    ps3 = stair_case(ps3)
                
                # plot
                axx.plot(*ps3.T[0:2], 'green')
                
                if WITH_MAYAVI:
                    # mlab.plot3d(*ps3.T, color=(1,1,1), figure=ax_3d)
                    pass
                    
            # take nodes from the 10 largest components to consider for end points
            nodes_to_consider = []
            comps = list(nx.connected_components(graph))
            
            if len(comps) == 0:
                continue
            
            max_comp_size = max(map(len, comps))
            for comp in comps:
                if len(comp) >= max_comp_size // 4:
                    nodes_to_consider.extend(comp)
                
            node_xyzs = list(zip(nodes_to_consider, map(get_node_xyz(graph, LD), nodes_to_consider)))
            # we have to account for threads, when we cut of above a certain threshold, the next z value below that can be significantly lower
            end_points_above = [end_point(a, storey_idx + 1, storey_idx) for a,(bx,by,bz) in node_xyzs if bz >= zmax_2 - 0.3]
            end_points_below = [end_point(a, storey_idx - 1, storey_idx) for a,(bx,by,bz) in node_xyzs if bz <= zmin_2 + 0.3]
            end_points = end_points_below + end_points_above
            
            # import pdb; pdb.set_trace()
                    
            relabeled = nx.relabel.convert_node_labels_to_integers(
                graph,
                first_label=len(complete_graph.nodes()),
                label_attribute='old_label'
            )
            
            node_mapping = {relabeled.nodes[n]['old_label']: n for n in relabeled.nodes}
            
            # import pdb; pdb.set_trace()
            
            print(len(end_points), "end points")
            
            all_end_points.extend(map(apply_end_point(node_mapping), end_points))
            
            nx.classes.function.set_edge_attributes(relabeled, LD, "level")
            nx.classes.function.set_node_attributes(relabeled, LD, "level")
            
            complete_graph = nx.union(complete_graph, relabeled)

            nodes = graph.nodes()
            if nodes:
                ps = numpy.array([nodes[i]['o'] for i in nodes])
                psn = numpy.int16(ps)
                dz = heights.data[tuple(psn.T)]
                # @todo use level_data instead of y.data.min() 
                ps2 = ps * flow.spacing + (y.data.min(), x.data.min())
                
                axx.plot(*ps2.T, 'r.', markersize=2)
                
                if WITH_MAYAVI:
                    pass
                    # mlab.points3d(*ps2.T, dz, color=(1,0,0), figure=ax_3d, scale_factor=0.05)
                    # for N, xy, z in zip(nodes, ps2, dz):
                    #     mlab.text3d(*xy, z, "%d-%d" % (storey_idx, N), figure=ax_3d, scale=0.05)

            fig.savefig(fn)

    # mlab.savefig("flow-3.png") #, dpi=150)

            
    def get_node_intermediate_path(Na, Nb):
        a_edges = complete_graph[a.node_id]
        b_edges = complete_graph[b.node_id]

        a_pt = complete_graph.nodes[a.node_id]['o']
        b_pt = complete_graph.nodes[b.node_id]['o']
        
        if len(a_edges) == 1 and len(b_edges) == 1:
            ae = next(iter(a_edges.values()))
            be = next(iter(b_edges.values()))

            a_pts = ae['pts']
            b_pts = be['pts']

                
            if numpy.linalg.norm(a_pts[0] - a_pt) > 1:
                a_pts = a_pts[::-1]


            if numpy.linalg.norm(b_pts[0] - b_pt) > 1:
                b_pts = b_pts[::-1]


            if len(a_pts) >= 6 and len(b_pts) >= 6:
                da = a_pts[0] - a_pts[5]
                db = b_pts[0] - b_pts[5]

                a0 = a_pts[0]
                a1 = a0 + da

                b0 = b_pts[0]
                b1 = b0 + db

                if abs(normalize(da).dot(normalize(db))) < 0.5:
                    
                    xx = seg_intersect(a0, a1, b0, b1)
                    
                    return numpy.concatenate((
                        numpy.array(list(bresenham(*numpy.int_(a_pt), *numpy.int_(xx)))),
                        numpy.array(list(bresenham(*numpy.int_(xx), *numpy.int_(b_pt))))
                    ))

        return numpy.array(list(bresenham(*numpy.int_(a_pt), *numpy.int_(b_pt))))
    
    """    
    for p in all_end_points:
        print(p, get_node_xyz(complete_graph)(p))
    """
            
    for a, b in itertools.combinations(all_end_points, 2):        
    
        # if {a.node_id, b.node_id} == {1648, 1652}:
        #     import pdb; pdb.set_trace()

        a_to_b = a.to_level == complete_graph.nodes[b.node_id]['level'].storey_idx
        b_to_a = b.to_level == complete_graph.nodes[a.node_id]['level'].storey_idx
        
        if a_to_b and b_to_a:
            # fa = flow_dist[a.node_id]
            # fb = flow_dist[b.node_id]
            
            xyza = get_node_xyz(complete_graph)(a)
            xyzb = get_node_xyz(complete_graph)(b)
            
            euclid_dist = numpy.sqrt(numpy.sum((numpy.array(xyza) - numpy.array(xyzb)) ** 2))
            
            fa = flow.lookup(xyza)
            fb = flow.lookup(xyzb)            
            
            # print("considering ", a.node_id, "->", b.node_id, ":", *xyza, "->", *xyzb, "values", fa, fb)
            # print("euclid dist", euclid_dist)
            
            if euclid_dist > 10.:
                # print("too far")
                continue            
            
            # manhattan_dist = sum(numpy.abs(numpy.array(xyza) - numpy.array(xyzb)) / flow.spacing)
            # avg_dist  = (manhattan_dist + euclid_dist) / 2.
            
            flow_diff = abs(fa - fb) * flow.spacing / 10. # voxec stores floats as int(v * 10)
            
            # print("flow ratio", abs(flow_diff - euclid_dist) / euclid_dist)
            
            if abs(flow_diff - euclid_dist) / euclid_dist < 0.32:
                # print("flow difference ok")
                na = complete_graph.nodes[a.node_id]
                nb = complete_graph.nodes[b.node_id]
                
                pa = na['o']
                pb = nb['o']
                
                # pts = numpy.array(list(bresenham(*pa, *pb)))
                pts = get_node_intermediate_path(a.node_id, b.node_id)

                ezmin, ezmax = sorted((xyza[2], xyzb[2]))
                flow_mi_ma = flow.get_slice(ezmin - 1., ezmax + 1.)
                x, y, arr, heights = flow.get_mean(flow_mi_ma)
                LD = level_data(-1, (xyza[2] + xyzb[2]) / 2., heights.data, x.data.min(), y.data.min())
                                
                # when we cross masked values we know we're connecting wrong points
                if not numpy.any(heights.mask[tuple(pts.T)]):
                    # print("height mask ok")
                    complete_graph.add_edge(a.node_id, b.node_id, pts=pts, level=LD)
                    
                    
            # check again with 2d intersection of edges
            # @todo always do this when |dot| < 0.5
            """
            
                xxxy = xx * flow.spacing + (LD.ymin, LD.xmin)
                xxz = heights.data[tuple(numpy.int_(xx.T))]
                xxxyz = numpy.concatenate((xxxy, numpy.atleast_1d(xxz)))
                
                euclid_dist = numpy.sqrt(numpy.sum((numpy.array(xyza) - numpy.array(xxxyz)) ** 2)) + \
                              numpy.sqrt(numpy.sum((numpy.array(xxxyz) - numpy.array(xyzb)) ** 2))
            """
    
    G = complete_graph
    
    return connectivity_graph(G)

    
    """
    path = dict(nx.all_pairs_shortest_path(G))
        
    import bisect
    def get_storey(elev):
        i = bisect.bisect(elevations, elev)
        return storeys[i]
        
    
    for a, bs in path.items():
        for b, nodes in path.items():
            node_zs = list(map(get_node_z, nodes))
            bisect.bisect(
    
    import pdb; pdb.set_trace()
    """
    
    # nodes_per_storey
    # 
    # n_Si  path  n_Si+1
    
                                      # todo make sure there are quite a few of them:
    # Nsh: nodes at storey height:   nz >= elevation | !E nnz: nnz >= elevation && nnz < nz
    # Edz: edges with                ||  { pz | p e edge }  || > 1
    # Ndz: nodes connected to Edz    n | n e Edz
    # 
    # N  : storey stair entrance     Nsh & Ndz
    # 
    # 
    
    
def process_landings():
    
    G = create_connectivity_graph()
    
    nzs = list(map(G.get_node_z, G.nodes))

    node_to_ifc_storey = {}

    elevations_covered = numpy.zeros((len(elevations),), dtype=bool)

    # Assign the most frequent Z coords to storeys
    # A storey is assigned at most 1 Z coord (elevations_covered)
    for zz, cnt in Counter(nzs).most_common():
        i = bisect.bisect(elevations, zz) - 1
        if not elevations_covered[i]:
            elevations_covered[i] = True
            for nidx, nz in zip(G.nodes, nzs):
                if nz == zz:
                    node_to_ifc_storey[nidx] = i
                    
    # node_to_ifc_storey = dict(filter(
    #     lambda t: t[1] != -1, 
    #     enumerate(ifc_storeys)
    # ))
                    
    edge_dzs = list(map(G.get_edge_dz, G.edges))
    # Nodes (by summing edges) that are part of a edge with dZ != 0
    ndz = sum(map(
        operator.itemgetter(1), 
        filter(lambda t: t[0] != 0., zip(edge_dzs, G.edges))
    ), ())
    
    stair_points = node_to_ifc_storey.keys() & ndz
    
    if WITH_MAYAVI:
        print("drawing points")
        pipe = G.draw_nodes()
        print("drawing edges")
        G.draw_edges(pipe)
      
    storeys_with_nodes = sorted(set(list(map(node_to_ifc_storey.__getitem__, stair_points))))

    storey_to_nodes = dict(
        (s, [t[0] for t in node_to_ifc_storey.items() if t[1] == s and t[0] in stair_points]) for s in storeys_with_nodes
    )
    
    """
    print("storey nodes")
    for kv in storey_to_nodes.items():
        print(*kv)
    """
        
    def yield_stair_paths():
        
        for i in range(len(storeys_with_nodes) - 1):
            # print("storey", i)
            sa, sb = storeys_with_nodes[i], storeys_with_nodes[i+1]
            if sa + 1 == sb:
                for na, nb in itertools.product(storey_to_nodes[sa], storey_to_nodes[sb]):
                    try:
                        asp = [nx.shortest_path(G.G, na, nb)]
                    except: continue
                    
                    # nx.all_simple_paths(G.G, na, nb)                
                    for path in asp:
                        if stair_points & set(path[1:-1]):
                            # contains another stair point intermediate in path, skip
                            pass
                        else:
                            # print("path", path[0], "->", path[-1])
                            yield path
        
    results = []
                            
    N = -1
    
    stair_paths_by_begin_end = defaultdict(list)
    
    for path in yield_stair_paths():
    
        begin_end = path[0], path[-1]
        stair_paths_by_begin_end[begin_end].append(path)
        
    # get the lowest elevation of a node in the graph for use later on
    def get_height_map_min(lv):
        hm = lv.height_map
        return hm[hm != hm.min()].min()

    levels = set(G.G.nodes[n]['level'] for n in G.G.nodes)
    global_min_z = min(map(get_height_map_min, levels))
    ##################################################################
    
    for _, paths in stair_paths_by_begin_end.items():
    
        shortest = None
        shortest_dist = 1e9
        
        for path in paths:
    
            points = numpy.concatenate(list(map(G.get_edge_points, path_to_edges(path))))
            edges = (numpy.roll(points, shift=-1, axis=0) - points)[:-1]
            path_length = sum(numpy.linalg.norm(e) for e in edges)
            
            if path_length < shortest_dist:
                shortest_dist = path_length
                shortest = points, edges
                
        points, edges = shortest
            
        edges_not_at_start = numpy.array(
            [e for e,p in zip(edges, points[:-1]) if p[2] != global_min_z]
        )
        max_incl = numpy.array(edges_not_at_start)[:, 2].max()
        
        if abs(max_incl - flow.spacing) < 1.e-9:
            # this is a ramp. no inclination beyond spacing
            # we have to discard the global elevation min value
            # though because we often see a jump there
            continue
        
        N += 1
        
        fn = "%s_%d.obj" % (id, N)
        obj_c = 1
        with open(fn, "w") as obj:
        
            incls = numpy.where(edges[:, 2] != 0.)[0]
            stair = points[max(incls.min() - 1, 0):incls.max() + 3]
            stair = rdp(stair, epsilon=flow.spacing/2.)
            sedges = numpy.roll(stair, shift=-1, axis=0) - stair
            
            # Find relating element by bounding box search, not very robust,
            # awaiting change to IfcOpenShell to use exact distance
            c = Counter()
            for p in points:
                lower = p - (0.1,0.1,0.5)
                upper = p + (0.1,0.1,0.1)
                box = (tuple(map(float, lower)), tuple(map(float, upper)))
                for inst in tree.select_box(box):
                    if not inst.is_a("IfcSpace"):
                        if inst.Decomposes:
                            inst = inst.Decomposes[0].RelatingObject
                        c.update([inst])
            ifc_elem = c.most_common(1)[0][0]
                    
            # tree.select((1.,1.,1.), extend=0.5)        

            li = []
            upw = None   # tribool starts unknown ( not false, not true )      

            for se in sedges[:-1]:
                if all(se == 0.):
                    continue
                    
                if upw != bool(se[2]):
                    li.append([])
                    upw = bool(se[2])
                li[-1].append(se)

            upw = [bool(x[0][2]) for x in li]
            lens = [ sum(numpy.linalg.norm(e) for e in es) for es in li ]
            
            num_landings = 0
            
            for is_upw, ll in zip(upw, lens):
                if is_upw is False and ll > LENGTH:
                    num_landings += 1
                    
            clr = 'red' if num_landings == 0 else 'green'
            st = 'ERROR' if num_landings == 0 else 'NOTICE'
                    
            desc = {
                "status": st,
                "numLandings": num_landings,
                "visualization": "/run/%s/result/resource/gltf/%d.glb" % (id, N),
                "guid": ifc_elem.GlobalId
            }            
            results.append(desc)
            
            print('mtllib mtl.mtl\n', file=obj)
            
            # import pdb; pdb.set_trace()
            
            horz = sum((y for x,y in zip(upw, li) if not x), [])
            cross_z = lambda h: normalize(numpy.cross(normalize(h), (0.,0.,1.)))
            crss = list(map(cross_z, horz))
            crss.append(crss[-1])
            # there can be nans as averaging opposite vectors gives a
            # zero vector, normalize gives nan. It's not clear why
            # opposite vectors are present in the edges.
            avgs = numpy.nan_to_num(
                [crss[0]] + [normalize(a+b) for a, b in zip(crss[1:], crss[:-1])]
            )
                   
            pt = stair[0].copy()
            avg_i = 0
            for ii, (is_upw, es, ll) in enumerate(zip(upw, li, lens)):
                d = ii + 1 if ii == 0 else ii - 1
                ref = li[d][0]
                for e in es:                
                    args = [ref, e]
                    
                    if is_upw:
                        args = args[::-1]
                        
                        
                    # dx = normalize(numpy.cross(*map(normalize, args))) * 0.5
                    dx = avgs[avg_i]
                    if is_upw:
                        dx2 = dx
                    else:
                        avg_i += 1
                        dx2 = avgs[avg_i]
                    
                    face_pts = numpy.array([
                        pt + dx  / 5.   ,
                        pt + dx2 / 5. + e,
                        pt - dx2 / 5. + e,
                        pt - dx  / 5.   
                    ])
                    
                    print('usemtl %s%s\n' % (clr, "" if ll > LENGTH else "2"), file=obj)
                    
                    for p in face_pts:
                        print("v", *p, file=obj)
                        
                    print("f", *range(obj_c, obj_c + 4), file=obj)
                    
                    obj_c += 4
                           
                    if WITH_MAYAVI:
                        mlab.plot3d(*face_pts.T, color=(0,1,0), figure=ax_3d, tube_radius=0.03)                
                    
                    pt += e
           
            if WITH_MAYAVI:
                mlab.plot3d(*stair.T, color=(0,1,0), figure=ax_3d, tube_radius=0.06)
    
    with open(id + ".json", "w") as f:
        json.dump({
            "id": id,
            "results": results
        }, f)
    
    if WITH_MAYAVI:
        mlab.show()
        
def process_routes():

    results = []
    
    evacuation_doors = config.get('objects')
    
    evac_zone_guids = set()
    evac_zone_space_guids = set()
    space_to_zone_type = {}
    
    if evacuation_doors is None:
        print("Using internal door selection")
        advance_length_doors = set()
        exit_route_doors = None
        evac_zone_doors = set()
        LENGTHS = [LENGTH, LENGTH]
    else:
        def counts_as_evac_door(d):
            if set(map(operator.itemgetter("type"), d["zones"])) == {"evacuationZone", "exitZone"}:
                return True
            if set(map(operator.itemgetter("type"), d["zones"])) == {"evacuationZone"} and d["evacDoor"]:
                return True
            return False
            
        for o in evacuation_doors:
            for z in o["zones"]:
                if z["type"] == "evacuationZone":
                    evac_zone_guids.add(z["guid"])
            
        evac_zone_doors = [o["guid"] for o in evacuation_doors if "evacuationZone" in set(map(operator.itemgetter("type"), o["zones"]))]
        advance_length_doors = [o["guid"] for o in evacuation_doors if o["evacDoor"]]
        exit_route_doors = [o["guid"] for o in evacuation_doors if o["firesafetyExit"]]
        evacuation_doors = [o["guid"] for o in evacuation_doors if counts_as_evac_door(o)]
        LENGTHS = config['lengths']
        print("Using %d supplied door guids" % len(evacuation_doors))
        
    for f in fs:
        for zn in f.by_type("IfcZone"):
            if zn.IsGroupedBy:
                for sp in zn.IsGroupedBy[0].RelatedObjects:
                    if zn.GlobalId in evac_zone_guids:
                        evac_zone_space_guids.add(sp.GlobalId)
                    space_to_zone_type[sp.GlobalId] = "evacuationZone" if zn.GlobalId in evac_zone_guids else "exitZone"
                    
    G = create_connectivity_graph()
    
    if WITH_MAYAVI:
        print("drawing points")
        pipe = G.draw_nodes()
        print("drawing edges")
        G.draw_edges(pipe)

    nodes_by_space = defaultdict(list)
    exterior_nodes = []
    xyzs = numpy.array(list(map(G.get_node_xyz, G.nodes)))
    node_to_zone_type = {}
    
    for n, xyz in zip(G.nodes, xyzs):
        for inst in tree.select(tuple(map(float, xyz))):
            if inst.is_a("IfcSpace"):
                nodes_by_space[inst].append(n)
                node_to_zone_type[n] = space_to_zone_type.get(inst.GlobalId)
                
    doors = sum(map(lambda f: f.by_type("IfcDoor"), fs), [])
    shapes = list(map(create_shape, doors))
    objs = list(map(ifc_element, doors, shapes))
                   
    for inst, dobj in zip(doors, objs):
        if dobj.is_external and (exit_route_doors is None or dobj.inst.GlobalId in exit_route_doors):
            array_idx = numpy.argmin(numpy.linalg.norm(xyzs - dobj.center[0:3], axis=1))
            node_idx = list(G.nodes)[array_idx]
            exterior_nodes.append(node_idx)
    
    def yield_routes():
        for sp, space_nodes in nodes_by_space.items():
            
            max_len = 0
            longest_path = None
            longest_path_edges = None
            
            # for na, nb in itertools.product(nodes, exterior_nodes):
            
            for na in space_nodes:
                
                min_len_space = 1e9
                shortest_path_space = None
                shortest_path_space_edges = None
                
                for nb in exterior_nodes:
                
                    graph_in_use = G.G
                
                    while True:
                
                        try:
                            path = nx.shortest_path(graph_in_use, na, nb)
                        except:
                            path = None
                            break
                        
                        zone_types = [node_to_zone_type.get(n) for n in path]
                        
                        has_seen_evac_zone = False
                        
                        # keep removing vertices until we arrive at a path
                        # that does not transition from evac to exit
                        valid = True
                        for n, zt in zip(path, zone_types):
                            if zt == "exitZone" and has_seen_evac_zone:
                                graph_in_use = graph_in_use.copy()
                                graph_in_use.remove_node(n)
                                valid = False
                                break
                            if zt == "evacuationZone":
                                has_seen_evac_zone = True
                                
                        if not valid:
                            continue
                        
                        break
                        
                    if path is None or len(path_to_edges(path)) == 0:
                        continue
                    
                    points = numpy.concatenate(list(map(G.get_edge_points, path_to_edges(path))))
                    edges = (numpy.roll(points, shift=-1, axis=0) - points)[:-1]
                    plen = numpy.sum(numpy.linalg.norm(edges, axis=1))
                    
                    if plen < min_len_space:
                        min_len_space = plen
                        shortest_path_space = points
                        shortest_path_space_edges = edges
                    
                if min_len_space > max_len:
                    max_len = min_len_space
                    longest_path = shortest_path_space
                    longest_path_edges = shortest_path_space_edges
            
            if longest_path is not None and longest_path_edges is not None:
                yield sp, space_nodes, None, longest_path, longest_path_edges
                
    def aabb_from_points(pts):
        return AABB(list(zip(pts.min(axis=0), pts.max(axis=0))))
                
    def door_to_aabb(door):
        pts = numpy.array([
            dobj.center[0:2] - dobj.M.T[0][0:2] * dobj.width / 2., 
            dobj.center[0:2] + dobj.M.T[0][0:2] * dobj.width / 2.
        ])
        pts = numpy.concatenate((pts, [[dobj.center[2]], [dobj.center[2] + dobj.height]]), axis=1)
        return aabb_from_points(pts)
                
    door_tree = AABBTree()
    for didx, dobj in enumerate(objs):
        if dobj.geom is not None:
            door_tree.add(door_to_aabb(dobj), didx)

    def break_at_doors(tup):
        sp, nodes, path, points, _ = tup
        
        # we don't do rdp() here but rather rely on a point index step to create small visualization gaps
        # and do the simplification when yielding the paths
        # points = rdp(points, epsilon=flow.spacing/2.)
        
        edges = (numpy.roll(points, shift=-1, axis=0) - points)[:-1]
        
        last_break = 0
        
        doors_used = set()
        
        length_index = 0
        
        in_evac_zone = sp.GlobalId in evac_zone_space_guids
        
        for pidx, (p0, ed) in enumerate(zip(points, edges)):
        
            for didx in door_tree.overlap_values(aabb_from_points(numpy.array([p0, p0+ed]))):
            
                dobj = objs[didx]
            
                if dobj in doors_used: continue

                if evacuation_doors is None:
                    is_fire_door = dobj.height is not None and dobj.width is not None and dobj.width > 1.5
                else:
                    is_fire_door = dobj.inst.GlobalId in evacuation_doors
                    
                advance_length = dobj.inst.GlobalId in advance_length_doors
                
                # External doors are never used as break points because they are already the end points
                # of the graph traversal
                is_external = dobj.is_external
                
                if is_fire_door and not is_external:
                
                    # we discard zero-length edges, but also vertical edges
                    if all(ed[0:2] == 0.):
                        continue
                    
                    if p0[2] >= dobj.center[2] and p0[2] <= dobj.center[2] + dobj.height:
                    
                        
                        dp0 = dobj.center[0:2] - dobj.M.T[0][0:2] * dobj.width / 2. 
                        dp1 = dobj.center[0:2] + dobj.M.T[0][0:2] * dobj.width / 2. 
                    
                        xx = seg_intersect(p0[0:2], (p0 + ed)[0:2], dp0, dp1)
                        
                        if numpy.linalg.norm((xx - dobj.center[0:2])) <= dobj.width / 2.:
                            
                            ponc = xx[0:2] - p0[0:2]
                            u = numpy.dot(ponc, normalize(ed[0:2])) / numpy.linalg.norm(ed[0:2])
                            
                            if u > 1.: continue
                            if u < 0.: continue
                            
                            doors_used.add(dobj)
                            
                            # previously we had code here to precisely split the segments but by
                            # deferring rdp() to after the segments have been broken up we know
                            # that the edge length is minimal at this point in time.
                            
                            yield length_index, in_evac_zone, points[last_break:pidx + 1]
                            last_break = pidx + 2
                            
                            if advance_length:
                                length_index += 1
                                
                            if dobj.inst.GlobalId in evac_zone_doors:
                                in_evac_zone = True
        
        yield length_index, in_evac_zone, points[last_break:]
        
    routes = yield_routes()
    for N, rt in enumerate(routes):
        
        fn = "%s_%d.obj" % (id, N)
        obj_c = 1
        with open(fn, "w") as obj:
            
            print('mtllib mtl.mtl\n', file=obj)
            
            segs = list(break_at_doors(rt))
            segments = [rdp(seg * (1., 1., 10.), epsilon=flow.spacing * 3) / (1., 1., 10.) for lidx, in_ez, seg in segs]
            segment_edges = [(numpy.roll(ps, shift=-1, axis=0) - ps)[:-1] for ps in segments]
            lens = [sum(map(numpy.linalg.norm, e)) for e in segment_edges]
            length_indices = [None if in_ez else int(lidx > 0) for lidx, in_ez, seg in segs]
            
            max_length = max(lens)
            allowed_lens = [None if lidx is None else LENGTHS[lidx] for lidx in length_indices]
            is_error = any(False if b is None else a > b for a, b in zip(lens, allowed_lens))
            
            st = 'ERROR' if is_error else 'NOTICE'
                    
            desc = {
                "status": st,
                "maxLength": max_length,
                "allowedSegmentLengths": allowed_lens,
                "segmentLengths": lens,
                "visualization": "/run/%s/result/resource/gltf/%d.glb" % (id, N),
                "guid": rt[0].GlobalId
            }            
            results.append(desc)
            
            for spoints, sedges, slen, alen in zip(segments, segment_edges, lens, allowed_lens):
            
                clr = 'red' if alen is not None and slen > alen else 'green'
                
                li = []
                upw = None   # tribool starts unknown ( not false, not true )      

                for se in sedges: # [:-1]:
                    if all(se == 0.):
                        continue
                        
                    if upw != bool(se[2]):
                        li.append([])
                        upw = bool(se[2])
                    li[-1].append(se)

                upw = [bool(x[0][2]) for x in li]
                 
                horz = sum((y for x,y in zip(upw, li) if not x), [])
                
                if len(horz) == 0:
                    continue
                
                cross_z = lambda h: normalize(numpy.cross(normalize(h), (0.,0.,1.)))
                crss = list(map(cross_z, horz))
                crss.append(crss[-1])
                # there can be nans as averaging opposite vectors gives a
                # zero vector, normalize gives nan. It's not clear why
                # opposite vectors are present in the edges.
                avgs = numpy.nan_to_num(
                    [crss[0]] + [normalize(a+b) for a, b in zip(crss[1:], crss[:-1])]
                )
                       
                pt = spoints[0].copy()
                avg_i = 0
                           
                for ii, (is_upw, es) in enumerate(zip(upw, li)):
                    for e in es:                
                        dx = avgs[avg_i]
                        
                        if is_upw:
                            dx2 = dx
                        else:
                            avg_i += 1
                            dx2 = avgs[avg_i]
                        
                        face_pts = numpy.array([
                            pt + dx  / 5.   ,
                            pt + dx2 / 5. + e,
                            pt - dx2 / 5. + e,
                            pt - dx  / 5.   
                        ])
                        
                        print('usemtl %s\n' % clr, file=obj)
                        
                        for p in face_pts:
                            print("v", *p, file=obj)
                            
                        print("f", *range(obj_c, obj_c + 4), file=obj)
                        
                        obj_c += 4
                               
                        if WITH_MAYAVI:
                            mlab.plot3d(*face_pts.T, color=(0,1,0), figure=ax_3d, tube_radius=0.03)                
                        
                        pt += e
                
    with open(id + ".json", "w") as f:
        json.dump({
            "id": id,
            "results": results
        }, f)
        
    if WITH_MAYAVI:
        mlab.show()


def process_entrance():

    with open("mtl.mtl", 'w') as f:
        f.write("newmtl red\n")
        f.write("Kd 1.0 0.0 0.0\n\n")
        f.write("newmtl orange\n")
        f.write("Kd 1.0 0.5 0.0\n\n")
        f.write("newmtl green\n")
        f.write("Kd 0.0 1.0 0.0 \n\n")

    results = []
    
    width = config.get('width', 1.5)
    depth = config.get('depth', 1.5)
    
    doors = sum(map(lambda f: f.by_type("IfcDoor"), fs), [])
    # @todo not necessary to create shape for internal doors
    shapes = list(map(create_shape, doors))
    
    # incl internal
    all_objs = list(map(ifc_element, doors, shapes))
    # external only
    objs = list(filter(operator.attrgetter('is_external'), all_objs))
    
    print(objs)

    for storey_idx, (mi, ma) in enumerate(elev_pairs):    
    
        objects_on_storey = []
        
        for ob in objs:
            if ob.M is None:
                continue
                
            Z = ob.M[2,3] + 1.
            if Z < mi or Z > ma:
                continue
                
            objects_on_storey.append(ob)
            
        if not objects_on_storey:
            print("no objects on storey")
            continue    
    
        plt.clf()
        plt.figure(figsize=flow.spacing * flow.global_sz[0:2])
        plt.gca().set_aspect('equal')
        plt.gca().set_xlabel('x')
        plt.gca().set_ylabel('y')
    
        storey = storeys[storey_idx]
        flow_mi_ma = flow.get_slice(mi - 1, ma)
        
        plt.scatter(flow_mi_ma.T[0], flow_mi_ma.T[1], marker='s', s=1, c=(flow_mi_ma.T[3] -1.) * flow.spacing, label='distance from exterior')     # norm=flow.norm,
        plt.colorbar()
        
        settings = ifcopenshell.geom.settings(
            USE_PYTHON_OPENCASCADE=False
        )
        
        # depend on elevation rather than containment relationship
        # as is done in other checks.
        # 
        # elems = None
        # if storey.ContainsElements:
        #     elems = list(set(storey.ContainsElements[0].RelatedElements) & set(external_doors))
        #     
        # if not elems:
        #     continue
                   
        flow_mi_ma = flow.get_slice(mi - 1, ma)
        
        if not flow_mi_ma.size:
            continue
        
        for ob in objects_on_storey:

            ob.draw(plt.gca())
        
            try:
                fdir = ob.outwards_direction()
            except ValueError as e:
                import traceback
                traceback.print_exc()
                continue
                
            plt.arrow(ob.center[0], ob.center[1], fdir[0] * 2., fdir[1] * 2., color='gray')
            
            dims = numpy.array([width, depth, (ob.bounds[1][2] - ob.bounds[0][2]) - 0.1, 0])
            # don't move Y inwards
            dims2 = dims.copy()
            dims2[1] = 0
            
            c = numpy.average(ob.bounds, axis=0)
            # Y outwards
            if fdir @ ob.M.T[1][0:3] > 0.:
                c[1] = ob.bounds[1][1]
                c[1] += 0.1
            else:
                c[1] = ob.bounds[0][1]
                c[1] -= 0.1
                # in this case local X is reversed
                dims2[0] *= -1.                
            
            lower_corner = ob.M @ (c - dims2 / 2.)
            
            ax = OCC.Core.gp.gp_Ax2(
                OCC.Core.gp.gp_Pnt(*tuple(map(float, lower_corner[0:3]))),
                OCC.Core.gp.gp_Dir(*tuple(map(float, ob.M.T[2][0:3]))),
                OCC.Core.gp.gp_Dir(*tuple(map(float, numpy.cross(fdir, ob.M.T[2][0:3])))) # y × z = x
            )
            
            plt.arrow(ob.center[0], ob.center[1], ax.XDirection().X(), ax.XDirection().Y(), color='red')
            plt.arrow(ob.center[0], ob.center[1], ax.YDirection().X(), ax.YDirection().Y(), color='green')
            
            dxdydz = tuple(map(float, dims[0:3]))
            
            bounds_diff = ob.bounds[1] - ob.bounds[0]
            if numpy.count_nonzero(bounds_diff >= 0.5) < 2 or bounds_diff[2] <= 0.5:
                # quick check to remove small doors, or misclassified doors
                continue
            
            try:
                box = OCC.Core.BRepPrimAPI.BRepPrimAPI_MakeBox(
                    ax, *dxdydz).Solid()
            except:
                continue
                
            intersections = [x for x in tree.select(box) if x.is_a() not in {'IfcOpeningElement', 'IfcSpace'}]
            valid = len(intersections) == 0
            
            if not valid:
                st = "ERROR"
                clr = "red"
            else:
                st = "NOTICE"
                clr = "green"
                
            with open("%s_%d.obj" % (id, len(results)), "w") as obj:
                obj_c = 1
                
                print('mtllib mtl.mtl\n', file=obj)
                print('usemtl %s\n' % clr, file=obj)
                
                for face in yield_subshapes(box, TopAbs_FACE):
                    wire = OCC.Core.BRepTools.breptools_OuterWire(face)
                    vertices = list(yield_subshapes(wire, vertices=True))
                    points = list(map(OCC.Core.BRep.BRep_Tool.Pnt, vertices))
                    xyzs = list(map(to_tuple, points))
                    xyzs_np = numpy.array(xyzs)
                    
                    if numpy.cross(xyzs_np[0], xyzs_np[1])[2] > 0.99:
                        cycle = numpy.concatenate((xyzs_np, [xyzs_np[0]]), axis=0)
                        plt.plot(cycle.T[0], cycle.T[1], c=clr)
                    
                    for xyz in xyzs:
                        print("v", *xyz, file=obj)
                        
                    print("f", *range(obj_c, len(xyzs)+obj_c), file=obj)
                    
                    obj_c += len(xyzs)
            
            desc = {
                "status": st,
                "guid": ob.inst.GlobalId,
                "intersections": [x.GlobalId for x in intersections],
                "visualization": "/run/%s/result/resource/gltf/%d.glb" % (id, len(results)),
            }
            
            results.append(desc)
        
        plt.savefig("flow-%d.png" % (storey_idx), dpi=150)
        
    with open(id + ".json", "w") as f:
        json.dump({
            "id": id,
            "results": results
        }, f)


if command == "doors":
    process_doors()
elif command == "landings":
	process_landings()
elif command == "routes":
	process_routes()
elif command == "risers":
	process_risers()
elif command == "entrance":
	process_entrance()
    
try:
    for fn in glob.glob("*.obj"):
        subprocess.check_call(["blender", "-b", "-P", os.path.join(os.path.dirname(__file__), "convert.py"), "--", fn, fn.replace(".obj", ".dae")])
        subprocess.check_call(["COLLADA2GLTF-bin", "-i", fn.replace(".obj", ".dae"), "-o", fn.replace(".obj", ".glb"), "-b", "1"])
except:
    import traceback
    traceback.print_exc(file=sys.stdout)


  
