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

from dataclasses import dataclass
from typing import Any

import numpy
from numpy.ma import masked_array
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
# mlab.options.offscreen = True

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element

from ifc_utils import prefix_mapping

normalize = lambda v: v / numpy.linalg.norm(v)

def wrap_try(fn):
    def inner(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except:
            return None
    return inner

id, fns, command, config = sys.argv[1:]

fns = ast.literal_eval(fns)
config = ast.literal_eval(config)

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
        
        self.width, self.height = None, None
        if hasattr(inst, "OverallWidth"):
            self.width = inst.OverallWidth * lu
        if hasattr(inst, "OverallHeight"):
            self.height = inst.OverallHeight * lu
        
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
        
    def validate(self, flow):    
        if self.geom is None:
            return
            
        def read(p):
            return flow.lookup(p[0:3], max_dist=0.4)
            # d, i = tree.query(p[0:3])
            # if d > 0.4:
            #     raise ValueError("%f > 0.4" % d)
            # return values[i]
        
        try:
            vals = list(map(read, self.pts))
            self.valid = vals[0] > vals[1]
        except ValueError as e:
            pass
        
inf = float("inf")


fs = list(map(ifcopenshell.open, fns))
lu = get_unit(fs[0], "LENGTHUNIT", 1.0)
storeys = sum(map(lambda f: f.by_type("IfcBuildingStorey"), fs), [])
elevations = list(map(lambda s: lu * getattr(s, 'Elevation', 0.), storeys))
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
for f in fs:
    tree.add_file(f, tree_settings)

class flow_field:
    
    def __init__(self, fn):
        self.flow = pd.read_csv(fn, delimiter=',').values
        self.flow = self.flow[self.flow[:,3] != 1.]

        self.spacing = min(numpy.diff(sorted(set(self.flow.T[0]))))

        self.flow_ints = numpy.int_(self.flow / self.spacing)
        self.global_mi = self.flow_ints.min(axis=0)
        self.global_ma = self.flow_ints.max(axis=0)
        self.global_sz = (self.global_ma - self.global_mi)[0:2] + 1
        
        self.tree = KDTree(self.flow[:, 0:3])
        
        self.norm = matplotlib.colors.Normalize(
            vmin=self.flow.T[3].min() * self.spacing, 
            vmax=self.flow.T[3].max() * self.spacing)
        

    def get_slice(self, min_z, max_z):
        data = self.flow[self.flow[:,2] >= min_z]
        data =      data[     data[:,2] <= max_z]
        return data


    def get_mean(self, data):
        ints = numpy.int_(data / self.spacing)
        arr = numpy.zeros(self.global_sz, dtype=float)
        counts = numpy.zeros(self.global_sz, dtype=int)
        highest = numpy.zeros(self.global_sz, dtype=float)
        highest[:] = -1e9
        
        ints2 = ints[:,0:2] - self.global_mi[0:2]
        for i,j,z,v in zip(*ints2.T[0:2], *data.T[2:4]):
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
            Z = ob.M[2,3] + 1.
            if Z < mi or Z > ma:
                continue
                
            for ax in axes:
                ob.draw(ax)
            ob.validate(flow)
            
            # AttributeError: 'FancyArrow' object has no attribute 'do_3d_projection'
            for ax in axes:
                ob.draw_arrow(ax)

            if ob in result_mapping:
                print("Warning element already emitted")
            else:
                N = len(result_mapping)
                fn = "%s_%d.obj" % (id, N)
                ob.draw_quiver(flow, x_y_angle, fn)
                result_mapping[ob] = N
            
            
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


class connectivity_graph:

    def __init__(self, G):
        self.G = G
        
        
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
        end_point_ids = {p.node_id for p in all_end_points}
        
        nodes = self.G.nodes()
        sorted_nodes = sorted(map(int, nodes))
        
        if node_colouring is None:
            end_point_mask = numpy.array([x in end_point_ids for x in sorted_nodes])
        else:
            end_point_mask = numpy.array([x in node_colouring for x in sorted_nodes])
        
        ps = numpy.array([nodes[i]['o'] for i in sorted_nodes])
        levels = [nodes[i]['level'] for i in sorted_nodes]
        offset = numpy.array([(l.ymin, l.xmin) for l in levels])
        psn = numpy.int16(ps)
        
        dz = numpy.zeros((len(nodes),))
        
        flow_dist = numpy.zeros((len(nodes),))
        for p in all_end_points:
            flow_dist[p.node_id] = flow.lookup(get_node_xyz(self.G)(p))

        for L in set(levels):
            mask = numpy.array([L == l for l in levels])
            dz[mask] = L.height_map[tuple(psn[mask].T)]
            
        ps2 = ps * flow.spacing + offset
        
        mlab.points3d(*ps2[~end_point_mask].T, dz[~end_point_mask], color=(0.5,0.5,0.5), figure=ax_3d, scale_factor=0.1)
        mlab.points3d(*ps2[ end_point_mask].T, dz[ end_point_mask], color=(1.0,0.0,0.0), figure=ax_3d, scale_factor=0.1)
        
        for N, xy, z, L, fd in zip(sorted_nodes, ps2, dz, levels, flow_dist):
            s = "    %d-%d" % (L.storey_idx, N)
            if fd:
                s += "(%d)" % fd
            mlab.text3d(*xy, z, s, figure=ax_3d, scale=0.05)
            
    def draw_edges(self):
        for s,e in self.G.edges():
            attrs = self.G[s][e]
            ps = attrs['pts']
            LD = attrs['level']
            dz = LD.height_map[tuple(ps.T)]
            ps2 = ps * flow.spacing + (LD.ymin, LD.xmin)
            ps_3d = numpy.column_stack((ps2, dz))
            ps3 = ps_3d
            if dz.max() - dz.min() > 1.e-5:
                ps3 = stair_case(ps3)
            mlab.plot3d(*ps3.T, color=(1,1,1), figure=ax_3d)

                            
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
    

def create_connectivity_graph():

    if WITH_MAYAVI:
        ax_3d = mlab.figure(size=(1600, 1200))

    levels = list(elevations)

    for a,b in zip(elevations, elevations[1:]):
        if b - a > 2:
            levels.append((b+a)/2.)
            
    levels = [x + flow.spacing for x in sorted(levels)]
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
            st2 = levels[i] - 0.5
        else:
            st = (levels[i] + levels[i-1] * 2.) / 3.
            st2 = levels[i] - 0.5
        if i == len(levels) - 1:
            en = levels[i] + 1.5
            en2 = levels[i] + 0.5
        else:
            en = (levels[i] + levels[i+1] * 2.) / 3.
            en2 = levels[i] + 0.5
        ranges.append((st, en, st2, en2))
        
    complete_graph = nx.Graph()
       
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


    def apply_end_point(mapping):
        return lambda ep: end_point(mapping[ep.node_id], ep.to_level)

       
    def get_node_xyz(graph):
        def inner(ep):              
            xy = graph.nodes[ep.node_id]['o']
            xyn = tuple(numpy.int16(xy))
            LD = graph.nodes[ep.node_id]['level']
            dz = LD.height_map[xyn]
            xy2 = xy * flow.spacing + (LD.ymin, LD.xmin)
            xyz = tuple(xy2) + (dz,)
            return xyz
        return inner


    all_end_points = []
        
    for storey_idx, (elev, (zmin, zmax, zmin_2, zmax_2)) in enumerate(zip(levels, ranges)):
        
        end_points = []
        
        
        # We extend beyond zmin and zmax, but trim the graph edges:
        
        # assymetric more extension downwards, because it cannot obscure route
        # but also not too much, because it can create topological connections also
        flow_mi_ma = flow.get_slice(zmin, zmax)
        
        if flow_mi_ma.size:
                        
            x, y, arr, heights = flow.get_mean(flow_mi_ma)
            
            fn = "section-%05d-%05d-%05d.png" % (elev * 1000, zmin * 1000, zmax * 1000)
            fig = plt.figure(figsize=((x.max() - x.min()), (y.max() - y.min())))
            axx = fig.add_subplot(111)
            
            # print("x", x.min(), x.data.min())
            # print("y", y.min(), y.data.min())

            obs = ~arr.mask
            
            LD = level_data(storey_idx, elev, heights.data, x.data.min(), y.data.min())        
            
            dxz, dyz = numpy.gradient(heights, flow.spacing)
            dZ = numpy.linalg.norm((dxz, dyz), axis=0)
            obs = ~arr.mask
                                
            # after distance?
            # no, after results in weird artefacts near the walls...
            # it really works better without, and then solve wrongly connected edges later.
            # obs[dZ > 3] = False
            
            ds = ndimage.distance_transform_edt(obs)
            
            # @todo would it help to blur a bit?                                                                                                                                     
            
            # results in weird artefacts near the walls...
            # ds[numpy.logical_and(dZ > 3, ~arr.mask)] = -1.

            ds_copy = ds.copy()
            ds_copy[arr.mask] = numpy.nan       
            
            # if storey_idx == 3:
            #     import pdb; pdb.set_trace()
            
            lp = ndimage.filters.laplace(ds)
            maa = lp < lp.min() / 20.
            maa2 = skeletonize(maa)
            
            # axx.imshow(ds_copy.T,  cmap='gray', alpha=0.5, extent=(y.data.min(), y.data.max(), x.data.min(), x.data.max()), origin='lower')
            axx.imshow(maa2.T,  cmap='gray', extent=(y.data.min(), y.data.max(), x.data.min(), x.data.max()), origin='lower')
            
            for bnd_idx, (clr, ctr) in enumerate(zip([(0,1,0,0.5), (1,0,0,0.5)], [zmin_2, zmax_2])):
                tmp = numpy.zeros(shape=tuple(ds_copy.shape) + (4,))
                if bnd_idx == 0:
                    tmp[numpy.logical_and(heights < ctr, ~arr.mask)] = clr
                else:
                    tmp[numpy.logical_and(heights > ctr, ~arr.mask)] = clr
                axx.imshow(tmp.transpose((1,0,2)), extent=(y.data.min(), y.data.max(), x.data.min(), x.data.max()), origin='lower')
            
        
            if WITH_MAYAVI:
                img = mlab.imshow(ds_copy[::-1, ::-1], colormap='gray', extent=(y.data.min(), y.data.max(), x.data.min(), x.data.max(), elev, elev), figure=ax_3d, transparent=False, interpolate=False)
                img.module_manager.scalar_lut_manager.lut.nan_color = 0, 0, 0, 0
                img.update_pipeline()
                    
            
            graph = sknw.build_sknw(maa2)
            
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
            
            nodes_to_add = []
            edges_to_add = []
            edges_to_remove = []
            
            # new node index
            N = lambda: len(graph.nodes()) + len(nodes_to_add) # + 1
            
            # fig.savefig(fn[:-4] + "_0.png")
            
            for s,e in graph.edges():
                ps = graph[s][e]['pts']
                dz = heights.data[tuple(ps.T)]
                dzmin, dzmax = dz.min(), dz.max()
                within = numpy.logical_and(dz >= zmin_2, dz <= zmax_2)
                
                under = dz < zmin_2 + flow.spacing * 2 + 1.e-3
                above = dz > zmax_2 - flow.spacing * 2 - 1.e-3
                
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
                    
                        # import pdb; pdb.set_trace()
                    
                        if st_ != 0:
                            STN = N()
                            st_ += 1
                            nodes_to_add.append((STN, {"o": ps[st_]}))
                            
                            # import pdb; pdb.set_trace()
                            
                            if under[st_] or above[st_]:
                                tlvl = storey_idx - 1 if under[st_] else storey_idx + 1
                                end_points.append(
                                    end_point(STN, tlvl)
                                )
                        else:
                            STN = s

                        if en_ != ps.shape[0] - 1:
                            ENN = N()
                            # @todo should this still be applied?
                            # en_ += 1
                            nodes_to_add.append((ENN, {"o": ps[en_]}))
                            
                            # import pdb; pdb.set_trace()
                            
                            if under[en_] or above[en_] or under[en_+1] or above[en_+1]:
                                tlvl = storey_idx - 1 if (under[en_] or under[en_+1]) else storey_idx + 1
                                end_points.append(
                                    end_point(ENN, tlvl)
                                )
                        else:
                            ENN = e
                            
                        if len(ps[st_:en_+1]) == 0:
                            import pdb; pdb.set_trace()
                            
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
                    
            graph.remove_edges_from(edges_to_remove)
            for n, kwargs in nodes_to_add:
                graph.add_node(n, **kwargs)
            for ab, kwargs in edges_to_add:
                st_pos = graph.nodes[ab[0]]['o']
                en_pos = graph.nodes[ab[1]]['o']
                if numpy.all(st_pos == en_pos):
                    import pdb; pdb.set_trace()
                    
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
                # ps3 = rdp(ps_3d, epsilon=spacing/2.)
                ps3 = ps_3d
                
                if dz.max() - dz.min() > 1.e-5:
                    ps3 = stair_case(ps3)
                
                # plot
                axx.plot(*ps3.T[0:2], 'green')
                
                if WITH_MAYAVI:
                    # mlab.plot3d(*ps3.T, color=(1,1,1), figure=ax_3d)
                    pass
                    
            relabeled = nx.relabel.convert_node_labels_to_integers(
                graph,
                first_label=len(complete_graph.nodes()),
                label_attribute='old_label'
            )
            
            node_mapping = {relabeled.nodes[n]['old_label']: n for n in relabeled.nodes}
            
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


            da = a_pts[0] - a_pts[5]
            db = b_pts[0] - b_pts[5]

            a0 = a_pts[0]
            a1 = a0 + da

            b0 = b_pts[0]
            b1 = b0 + db

            if abs(normalize(da).dot(normalize(db))) < 0.5:
                
                xx = numpy.int_(seg_intersect(a0, a1, b0, b1))
                
                return numpy.concatenate((
                    numpy.array(list(bresenham(*a_pt, *xx))),
                    numpy.array(list(bresenham(*xx, *b_pt)))
                ))

        return numpy.array(list(bresenham(*a_pt, *b_pt)))
        
    for p in all_end_points:
        print(p)
            
    for a, b in itertools.combinations(all_end_points, 2):        
    
        # if {a.node_id, b.node_id} == {1648, 1652}:
        #     import pdb; pdb.set_trace()

        a_to_b = a.to_level == complete_graph.nodes[b.node_id]['level'].storey_idx
        b_to_a = b.to_level == complete_graph.nodes[a.node_id]['level'].storey_idx
        
        if a_to_b and b_to_a:
            # fa = flow_dist[a.node_id]
            # fb = flow_dist[b.node_id]
            
            fa = flow.lookup(get_node_xyz(complete_graph)(a))
            fb = flow.lookup(get_node_xyz(complete_graph)(b))
            
            xyza = get_node_xyz(complete_graph)(a)
            xyzb = get_node_xyz(complete_graph)(b)
            
            ezmin, ezmax = sorted((xyza[2], xyzb[2]))
            flow_mi_ma = flow.get_slice(ezmin - 1., ezmax + 1.)
            x, y, arr, heights = flow.get_mean(flow_mi_ma)
            LD = level_data(-1, (xyza[2] + xyzb[2]) / 2., heights.data, x.data.min(), y.data.min())
            
            # manhattan_dist = sum(numpy.abs(numpy.array(xyza) - numpy.array(xyzb)) / flow.spacing)
            euclid_dist = numpy.sqrt(numpy.sum((numpy.array(xyza) - numpy.array(xyzb)) ** 2))
            # avg_dist  = (manhattan_dist + euclid_dist) / 2.
            
            flow_diff = abs(fa - fb) * flow.spacing / 10. # voxec stores floats as int(v * 10)
            
            if abs(flow_diff - euclid_dist) / euclid_dist < 0.15:
                na = complete_graph.nodes[a.node_id]
                nb = complete_graph.nodes[b.node_id]
                
                pa = na['o']
                pb = nb['o']
                
                # pts = numpy.array(list(bresenham(*pa, *pb)))
                pts = get_node_intermediate_path(a.node_id, b.node_id)
                                
                # when we cross masked values we know we're connecting wrong points
                if not numpy.any(heights.mask[tuple(pts.T)]):
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

    ifc_storeys = numpy.zeros((len(G.nodes),), dtype=int) - 1

    elevations_covered = numpy.zeros((len(elevations),), dtype=bool)

    for zz, cnt in Counter(nzs).most_common():
        i = bisect.bisect(elevations, zz) - 1
        if not elevations_covered[i]:
            elevations_covered[i] = True
            for nidx, nz in zip(G.nodes, nzs):
                if nz == zz:
                    ifc_storeys[nidx] = i
                    
    node_to_ifc_storey = dict(filter(
        lambda t: t[1] != -1, 
        enumerate(ifc_storeys)
    ))
                    
    edge_dzs = list(map(G.get_edge_dz, G.edges))
    ndz = sum(map(
        operator.itemgetter(1), 
        filter(lambda t: t[0] != 0., zip(edge_dzs, G.edges))
    ), ())
    
    stair_points = node_to_ifc_storey.keys() & ndz
    
    if WITH_MAYAVI:
        G.draw_edges()
        G.draw_nodes(stair_points)
      
    storeys_with_nodes = sorted(set(list(map(node_to_ifc_storey.__getitem__, stair_points))))

    storey_to_nodes = dict(
        (s, [t[0] for t in enumerate(ifc_storeys) if t[1] == s and t[0] in stair_points]) for s in storeys_with_nodes
    )
        
    def yield_stair_paths():
        for i in range(len(storeys_with_nodes) - 1):
            sa, sb = storeys_with_nodes[i], storeys_with_nodes[i+1]
            if sa + 1 == sb:
                for na, nb in itertools.product(storey_to_nodes[sa], storey_to_nodes[sb]):
                    for path in nx.all_simple_paths(G.G, na, nb):
                        if stair_points & set(path[1:-1]):
                            # contains another stair point intermediate in path, skip
                            pass
                        else:
                            yield path
        
    results = []
                            
    for N, path in enumerate(yield_stair_paths()):
    
        # import pdb; pdb.set_trace()
    
        fn = "%s_%d.obj" % (id, N)
        obj = open(fn, "w")
        obj_c = 1
        
        points = numpy.concatenate(list(map(G.get_edge_points, path_to_edges(path))))
        edges = numpy.roll(points, shift=-1, axis=0) - points
        incls = numpy.where(edges[:-1, 2] != 0.)[0]
        stair = points[max(incls.min() - 1, 0):incls.max() + 3]
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
        avgs = [crss[0]] + [normalize(a+b) for a, b in zip(crss[1:], crss[:-1])]
               
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

    G = create_connectivity_graph()

    nodes_by_space = defaultdict(list)
    exterior_nodes = []
    xyzs = numpy.array(list(map(G.get_node_xyz, G.nodes)))
    
    for n, xyz in zip(G.nodes, xyzs):
        for inst in tree.select(tuple(map(float, xyz))):
            if inst.is_a("IfcSpace"):
                nodes_by_space[inst].append(n)
                
    doors = sum(map(lambda f: f.by_type("IfcDoor"), fs), [])
    shapes = list(map(create_shape, doors))
    objs = list(map(ifc_element, doors, shapes))
    
    is_external_mapping = {}
                
    for inst, dobj in zip(doors, objs):
        is_external = ifcopenshell.util.element.get_psets(inst).get('Pset_DoorCommon', {}).get('IsExternal', False) is True
        is_external_mapping[dobj] = is_external
        if is_external:
            node_idx = numpy.argmin(numpy.linalg.norm(xyzs - dobj.center[0:3], axis=1))
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
                    for path in nx.all_simple_paths(G.G, na, nb):
                        points = numpy.concatenate(list(map(G.get_edge_points, path_to_edges(path))))
                        edges = (numpy.roll(points, shift=-1, axis=0) - points)[:-1]
                        plen = sum(map(numpy.linalg.norm, edges))
                        
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


    def break_at_doors(tup):
        sp, nodes, path, points, edges = tup
        
        last_break = 0
        previous_break_point = ()
        
        for inst, dobj in zip(doors, objs):
            # if dobj.inst.GlobalId == "2OBrcmyk58NupXoVOHUuXp" and sp.GlobalId == "0BTBFw6f90Nfh9rP1dl_3A":
            #     import pdb; pdb.set_trace()

            is_fire_door = dobj.height is not None and dobj.width is not None and dobj.width > 1.5
            is_external = is_external_mapping.get(dobj)
            
            if is_fire_door and not is_external:
            
                
                for pidx, (p0, ed) in enumerate(zip(points, edges)):
                
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
                            
                            if abs(u) < 0.01:
                                
                                yield numpy.concatenate(previous_break_point + (points[last_break:pidx + 1],))
                                last_break = pidx + 1
                                previous_break_point = ()
                                
                            elif abs(1. - u) < 0.01:
                            
                                yield numpy.concatenate(previous_break_point + (points[last_break:pidx + 2],))
                                last_break = pidx + 2
                                previous_break_point = ()
                                
                            else:
                                
                                pinter = p0 + ed * u
                                
                                yield numpy.concatenate(previous_break_point + (points[last_break:pidx + 1], [pinter]))
                                last_break = pidx + 1
                                previous_break_point = ([pinter],)
                                
        yield numpy.concatenate(previous_break_point + (points[last_break:],))
        
    for N, rt in enumerate(yield_routes()):
        
        fn = "%s_%d.obj" % (id, N)
        obj = open(fn, "w")
        obj_c = 1
        print('mtllib mtl.mtl\n', file=obj)
        
        segments = list(break_at_doors(rt))
        segment_edges = [(numpy.roll(ps, shift=-1, axis=0) - ps)[:-1] for ps in segments]
        lens = [sum(map(numpy.linalg.norm, e)) for e in segment_edges]
        
        max_length = max(lens)
        is_error = lambda l: l > LENGTH
        
        st = 'ERROR' if is_error(max_length) else 'NOTICE'
                
        desc = {
            "status": st,
            "maxLength": max_length,
            "visualization": "/run/%s/result/resource/gltf/%d.glb" % (id, N),
            "guid": rt[0].GlobalId
        }            
        results.append(desc)
        
        for spoints, sedges, slen in zip(segments, segment_edges, lens):
        
            clr = 'red' if is_error(slen) else 'green'
            
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
             
            horz = sum((y for x,y in zip(upw, li) if not x), [])
            
            if len(horz) == 0:
                continue
            
            cross_z = lambda h: normalize(numpy.cross(normalize(h), (0.,0.,1.)))
            crss = list(map(cross_z, horz))
            crss.append(crss[-1])
            avgs = [crss[0]] + [normalize(a+b) for a, b in zip(crss[1:], crss[:-1])]
                   
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


if command == "doors":
    process_doors()
elif command == "landings":
	process_landings()
elif command == "routes":
	process_routes()

try:
    for fn in glob.glob("*.obj"):
        subprocess.check_call(["blender", "-b", "-P", os.path.join(os.path.dirname(__file__), "convert.py"), "--", fn, fn.replace(".obj", ".dae")])
        subprocess.check_call(["COLLADA2GLTF-bin", "-i", fn.replace(".obj", ".dae"), "-o", fn.replace(".obj", ".glb"), "-b", "1"])
except:
    import traceback
    traceback.print_exc(file=sys.stdout)


  