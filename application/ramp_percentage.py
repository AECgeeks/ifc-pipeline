import os
import sys
import math
import json
import glob
import subprocess

import ifcopenshell
import ifcopenshell.geom

import OCC.Core.BRep
import OCC.Core.BRepGProp
import OCC.Core.BRepTools
import OCC.Core.TopTools
import OCC.Core.TopoDS
import OCC.Core.TopExp

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



id = sys.argv[1]
warning, error = map(float, sys.argv[2:4])
fs = list(map(ifcopenshell.open,sys.argv[4:]))
ramps = sum(map(lambda f: f.by_type("IfcRamp"), fs), [])

settings = ifcopenshell.geom.settings(
    USE_PYTHON_OPENCASCADE=True
)

results = []

for N, ramp in enumerate(ramps):
    
    try:
        shp = ifcopenshell.geom.create_shape(settings, ramp).geometry
    except: continue
        
    faces = list(yield_subshapes(shp, TopAbs_FACE))
    
    def get_norm(face):
        prop = OCC.Core.BRepGProp.BRepGProp_Face(face)
        uvb = OCC.Core.BRepTools.breptools_UVBounds(face)
        P = OCC.Core.gp.gp_Pnt()
        V = OCC.Core.gp.gp_Vec()
        prop.Normal(
            (uvb[0] + uvb[1]) / 2.,
            (uvb[1] + uvb[3]) / 2.,
            P, V
        )
        return V
        
    normals = list(map(get_norm, faces))
    
    nf_pairs = sorted(zip(faces, normals), key=lambda t: -t[1].Z())
    
    up_face, up_normal = nf_pairs[0]
    
    percentage = math.tan(math.acos(up_normal.Z())) * 100.
    
    if abs(percentage) < 0.01:
        continue
    
    if percentage >= error:
        st = "ERROR"
        clr = "red"
    elif percentage >= warning:
        st = "WARNING"
        clr = "orange"
    else:
        st = "NOTICE"
        clr = "green"
        
    up_wire = OCC.Core.BRepTools.breptools_OuterWire(up_face)
    vertices = list(yield_subshapes(up_wire, vertices=True))
    points = list(map(OCC.Core.BRep.BRep_Tool.Pnt, vertices))
    xyzs = list(map(to_tuple, points))
    
    obj = open("%s_%d.obj" % (id, N), "w")
    
    print('mtllib mtl.mtl\n', file=obj)
    print('usemtl %s\n' % clr, file=obj)
    
    for xyz in xyzs:
        print("v", *xyz, file=obj)
        
    print("f", *range(1, len(xyzs)+1), file=obj)
    
    desc = {
        "status": st,
        "percentage": percentage,
        "visualization": "/run/%s/result/resource/gltf/%d.glb" % (id, N),
        "guid": ramp.GlobalId
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

