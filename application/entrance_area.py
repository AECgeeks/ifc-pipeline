import os
import sys
import math
import json
import glob
import subprocess

import numpy

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element

import OCC.Core.BRep
import OCC.Core.BRepGProp
import OCC.Core.BRepTools
import OCC.Core.TopTools
import OCC.Core.TopoDS
import OCC.Core.TopExp
import OCC.Core.BRepPrimAPI

from OCC.Core.TopAbs import *

with open("mtl.mtl", 'w') as f:
    f.write("newmtl red\n")
    f.write("Kd 1.0 0.0 0.0\n\n")
    f.write("newmtl orange\n")
    f.write("Kd 1.0 0.5 0.0\n\n")
    f.write("newmtl green\n")
    f.write("Kd 0.0 1.0 0.0 \n\n")


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


print(*enumerate(sys.argv))
id = sys.argv[1]
width = float(sys.argv[2])
depth = float(sys.argv[3])
fs = list(map(ifcopenshell.open,sys.argv[4:]))
doors = sum(map(lambda f: f.by_type("IfcDoor"), fs), [])
is_external = lambda inst: ifcopenshell.util.element.get_psets(inst).get('Pset_DoorCommon', {}).get('IsExternal', False) is True
external_doors = list(filter(is_external, doors))

settings = ifcopenshell.geom.settings(
    USE_PYTHON_OPENCASCADE=False
)

tree_settings = ifcopenshell.geom.settings(
    DISABLE_TRIANGULATION=True,
    DISABLE_OPENING_SUBTRACTIONS=True
)

tree = ifcopenshell.geom.tree()
for f in fs:
    tree.add_file(f, tree_settings)

results = []

for N, door in enumerate(external_doors):
    
    try:
        geom = ifcopenshell.geom.create_shape(settings, door)
    except: continue
    
    v = geom.geometry.verts
    d = geom.transformation.matrix.data
    
    M = numpy.concatenate((
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
    # Y outwards
    c[1] = bounds[1][1]
    c[1] += 0.1
    
    dims = numpy.array([width, depth, (bounds[1][2] - bounds[0][2]) - 0.1, 0])
    # don't move Y inwards
    dims2 = dims.copy()
    dims2[1] = 0
      
    lower_corner = M @ (c - dims2 / 2.)
    
    ax = OCC.Core.gp.gp_Ax2(
        OCC.Core.gp.gp_Pnt(*tuple(map(float, lower_corner[0:3]))),
        OCC.Core.gp.gp_Dir(*tuple(map(float, M.T[2][0:3]))),
        OCC.Core.gp.gp_Dir(*tuple(map(float, M.T[0][0:3])))
    )
    
    dxdydz = tuple(map(float, dims[0:3]))
    
    box = OCC.Core.BRepPrimAPI.BRepPrimAPI_MakeBox(
        ax, *dxdydz).Solid()
        
    valid = len(set(i.is_a() for i in tree.select(box)) - {'IfcOpeningElement', 'IfcSpace'}) == 0
                
    if not valid:
        st = "ERROR"
        clr = "red"
    else:
        st = "NOTICE"
        clr = "green"
    
    with open("%s_%d.obj" % (id, N), "w") as obj:
        obj_c = 1
        
        print('mtllib mtl.mtl\n', file=obj)
        print('usemtl %s\n' % clr, file=obj)
        
        for face in yield_subshapes(box, TopAbs_FACE):
            wire = OCC.Core.BRepTools.breptools_OuterWire(face)
            vertices = list(yield_subshapes(wire, vertices=True))
            points = list(map(OCC.Core.BRep.BRep_Tool.Pnt, vertices))
            xyzs = list(map(to_tuple, points))
            
            for xyz in xyzs:
                print("v", *xyz, file=obj)
                
            print("f", *range(obj_c, len(xyzs)+obj_c), file=obj)
            
            obj_c += len(xyzs)
        
    desc = {
        "status": st,
        "visualization": "/run/%s/result/resource/gltf/%d.glb" % (id, N),
        "guid": door.GlobalId
    }
    
    results.append(desc)
    

with open(id + ".json", "w") as f:
    json.dump({
        "id": id,
        "results": results
    }, f)

try:
    for fn in glob.glob("*.obj"):
        subprocess.check_call(["blender", "-b", "-P", os.path.join(os.path.dirname(__file__), "convert.py"), "--", fn, fn.replace(".obj", ".dae")])
        subprocess.check_call(["COLLADA2GLTF-bin", "-i", fn.replace(".obj", ".dae"), "-o", fn.replace(".obj", ".glb"), "-b", "1"])
except:
    import traceback
    traceback.print_exc(file=sys.stdout)

