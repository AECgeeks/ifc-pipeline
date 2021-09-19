import os
import sys
import ast
import json
import subprocess

import ifcopenshell
import ifcopenshell.geom

import numpy

from collections import defaultdict

id, fns = sys.argv[1:]
fns = ast.literal_eval(fns)
files = [ifcopenshell.open(fn) for fn in fns]

s = ifcopenshell.geom.settings(
    DISABLE_TRIANGULATION=True,
    DISABLE_OPENING_SUBTRACTIONS=True
)

tree = ifcopenshell.geom.tree()
for f in files:
    tree.add_file(f, s)

ifn = "simplified.obj"

def vertices():
    with open(ifn) as f:
        for l in f:
            if l.startswith("v "):
                yield tuple(map(float, l.split(" ")[1:]))

verts = numpy.array(list(vertices()))

def groups():
    current = []
    name = None
    with open(ifn) as f:
        for l in f:
            if l.startswith("g "):
                if current:
                    yield name, current
                    current[:] = []
                name = l[2:].strip()
            elif l.startswith("f "):
                vidx = l.split(" ")[1:]
                current.append(tuple(map(int, vidx)))
    if current:
        yield name, current
        current[:] = []
        
name_mapping = defaultdict(list)

for name, idxs in groups():
    pts = verts[(numpy.array(sum(idxs, ()))-1)]
    center = (pts.min(axis=0) + pts.max(axis=0)) / 2.
    
    lower = center - (0.2,0.2,0.2)
    upper = center + (0.2,0.2,0.2)
    box = (tuple(map(float, lower)), tuple(map(float, upper)))
    insts = tree.select_box(box)
    
    for inst in insts:
        if inst.Decomposes:
            inst = inst.Decomposes[0].RelatingObject
        if inst.is_a("IfcStair"):
            name_mapping[inst].append(name)

with open('colours.mtl', 'w') as f:
    f.write("newmtl red\n")
    f.write("Kd 1.0 0.0 0.0\n\n")

with open(ifn, 'r+') as f:
    ls = f.readlines()
    ls.insert(0, "mtllib colours.mtl\n")
    ls.insert(1, "usemtl red\n")
    f.seek(0)
    f.writelines(ls)

# --orient causes issues?
subprocess.check_call(["blender", "-b", "-P", os.path.join(os.path.dirname(__file__), "convert.py"), "--split", "--orient", "--components", "--", "simplified.obj", os.path.abspath("%s.dae")])

results = []
for i, (inst, names) in enumerate(name_mapping.items()):
    subprocess.check_call(["blender", "-b", "-P", os.path.join(os.path.dirname(__file__), "convert.py"), "--", *(os.path.abspath(n + ".dae") for n in names), os.path.abspath("%d.dae" % i)])
    subprocess.check_call(["COLLADA2GLTF-bin", "-i", "%d.dae" % i, "-o", "%s_%d.glb" % (id, i), "-b", "1"])
    
    results.append({
        "status": "ERROR",
        "visualization": "/run/%s/result/resource/gltf/%d.glb" % (id, i),
        "guid": inst.GlobalId
    })
        
with open(id + ".json", "w") as out:
    json.dump({
        "id": id,
        "results": results
    }, out)
