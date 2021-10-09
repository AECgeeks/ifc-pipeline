import os
import sys
import ast
import json
import functools
import subprocess

import ifcopenshell
import ifcopenshell.geom

import numpy

from collections import defaultdict, Counter

try:
    id, fns = sys.argv[1:]
    element_type = "IfcStair"
except:
    id, fns, element_type = sys.argv[1:]
    if element_type == "all":
        element_type = None
    
fns = ast.literal_eval(fns)
files = [ifcopenshell.open(fn) for fn in fns]

s = ifcopenshell.geom.settings(
    DISABLE_TRIANGULATION=True,
    DISABLE_OPENING_SUBTRACTIONS=True
)

tree = ifcopenshell.geom.tree()
include = {}
if element_type:

    def get_decompositions(x):
        """
        Generator that recursively yields decomposing elements
        """
        for rel in x.IsDecomposedBy:
            obs = rel.RelatedObjects
            yield from obs
            for ob in obs:
                yield from get_decompositions(ob)

    insts = sum((f.by_type(element_type) for f in files), [])
    for inst in list(insts):
        insts.extend(get_decompositions(inst))

    include["include"] = insts

iterators = list(map(functools.partial(ifcopenshell.geom.iterator, s, **include), files))
for it in iterators:
    tree.add_iterator(it)

ifn = "result.obj"

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
                vidx = [x.split("/")[0] for x in vidx]
                current.append(tuple(map(int, vidx)))
    if current:
        yield name, current
        current[:] = []
        
# obj name -> ifc insts
name_mapping = defaultdict(Counter)

import random
import time

for name, idxs in groups():
    t0 = time.perf_counter()
    # take set as multiple triangles use same position
    pts = verts[(numpy.array(list(set(sum(idxs, ()))))-1)]
    # select only z=0.1+ or z=0.05+ as we have bottom and
    # top faces for each vertex
    # a bit clumsy due to sign differences of fmod
    # @todo can we move them to the middle?
    select = numpy.abs(numpy.abs(numpy.fmod(pts[:,2], 0.1)) - 0.05) < 1.e-5
    pts = pts[select]
    
    # take random subset of 100 pts
    pts = list(pts)
    random.shuffle(pts)
    pts = pts[0:100]
    
    for pt in pts:
        pnt_t = tuple(map(float, pt))
    
        for radius in (0.1, 0.2, 0.3):
            insts = tree.select(pnt_t, extend=radius)
            if insts: break
        
        for inst in insts:
            if inst.Decomposes:
                inst = inst.Decomposes[0].RelatingObject
            if element_type is None or inst.is_a(element_type):
                name_mapping[name].update([inst])
    
    t1 = time.perf_counter()
    print("%.5f" % (t1 - t0), name, len(pts))
                
with open('colours.mtl', 'w') as f:
    f.write("newmtl red\n")
    f.write("Kd 1.0 0.0 0.0\n\n")

with open("simplified.obj", 'r+') as f:
    ls = f.readlines()
    ls.insert(0, "mtllib colours.mtl\n")
    ls.insert(1, "usemtl red\n")
    f.seek(0)
    f.writelines(ls)

# we can have multiple obj names corresponding to the same ifc element, so regroup
# ifc inst -> obj names
name_mapping_2 = defaultdict(list)
for i, (name, insts) in enumerate(name_mapping.items()):
    M = max(insts.values())
    for inst in [i for i,c in insts.items() if c > M // 4]:
        name_mapping_2[inst].append(name)
        
import pprint
pprint.pprint(name_mapping)
pprint.pprint(name_mapping_2)

# @nb we have generated associations using result.obj for dense points
# we actually use simplified.obj now for the visualization. The group
# names are identical in both files.

# --orient causes issues?
subprocess.check_call(["blender", "-b", "-P", os.path.join(os.path.dirname(__file__), "convert.py"), "--split", "--orient", "--components", "--", "simplified.obj", os.path.abspath("%s.dae")])

results = []
for i, (inst, names) in enumerate(name_mapping_2.items()):
            
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
