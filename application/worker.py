##################################################################################
#                                                                                #
# Copyright (c) 2020 AECgeeks                                                    #
#                                                                                #
# Permission is hereby granted, free of charge, to any person obtaining a copy   #
# of this software and associated documentation files (the "Software"), to deal  #
# in the Software without restriction, including without limitation the rights   #
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell      #
# copies of the Software, and to permit persons to whom the Software is          #
# furnished to do so, subject to the following conditions:                       #
#                                                                                #
# The above copyright notice and this permission notice shall be included in all #
# copies or substantial portions of the Software.                                #
#                                                                                #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE    #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  #
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  #
# SOFTWARE.                                                                      #
#                                                                                #
##################################################################################

import datetime
import os
from pathlib import Path
import sys
import json
import glob
import threading
import time
import platform
import traceback
import importlib
import subprocess
import operator
import shutil
import functools

from collections import defaultdict
import psutil

import requests

orig_print = print
def print(*args, **kwargs):
    current_time_s = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    orig_print(f"[{current_time_s}]", *args, **kwargs)


def queue_task(func):
    @functools.wraps(func)
    def wrapper(id, *args, **kwargs):
        txt = utils.storage_file_for_id(id, 'job.txt')
        os.makedirs(os.path.dirname(txt), exist_ok=True)
        with open(txt, 'w') as f:
            sys.stdout = f
            exc = None
            try:
                result = func(id, *args, **kwargs)
            except Exception as e:
                exc = e
            traceback.print_exc(file=sys.stdout)
            sys.stdout = sys.__stdout__
            utils.store_file(id, extension='job.txt')
            
            if exc:
                traceback.print_exc(file=sys.stdout)
                set_progress(id, -2)
                if (a := getattr(exc, 'args')) and a and (s := a[0]) and isinstance(s, str):
                    set_error(id, s)
                else:
                    set_error(id, 'unhandled exception')

                raise exc
            else:
                return result

    wrapper.original = func
    wrapper.original.__qualname__ += ".original"

    return wrapper


def run(*args, **kwargs):
    print("Executing", *args)
    for k, v in kwargs.items():
        print(f"With {k} = {v}")
    for ln in subprocess.run(args, capture_output=True, universal_newlines=True, check=True, **kwargs).stdout.split("\n"):
        print(ln)


on_windows = platform.system() == 'Windows'
ext = ".exe" if on_windows else ""
exe_path = os.path.join(os.path.dirname(__file__), "win" if on_windows else "nix")

IFCCONVERT = os.path.join(exe_path, "IfcConvert") + ext
if not os.path.exists(IFCCONVERT):
    IFCCONVERT = shutil.which("IfcConvert")

if VOXEC := os.environ.get("VOXEC_EXE"):
    pass
else:
    VOXEC = os.path.join(exe_path, "voxec") + ext
    if not os.path.exists(VOXEC):
        VOXEC = shutil.which("voxec")

assert VOXEC
assert IFCCONVERT

REPLACE_DETAILED_ELEMENTS = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'replace-detailed-elements.py')
assert os.path.exists(REPLACE_DETAILED_ELEMENTS)

import utils
import database


def set_progress(id, progress):
    session = database.Session()
   
    id = id.split("_")[0]

    model = session.query(database.model).filter(database.model.code == id).all()[0]
    model.progress = int(progress)
    session.commit()
    session.close()


def set_error(id, error):
    session = database.Session()
   
    id = id.split("_")[0]

    model = session.query(database.model).filter(database.model.code == id).all()[0]
    model.error = error
    session.commit()
    session.close()


class task(object):
    def __init__(self, progress_map):
        import inspect
        print(self.__class__.__name__, inspect.getfile(type(self)), *progress_map)
        self.begin, self.end = progress_map

    def sub_progress(self, i):
        set_progress(self.id, self.begin + (self.end - self.begin) * i / 100.)

    def __call__(self, directory, id, *args):
        self.id = id
        self.execute(directory, id, *args)
        self.sub_progress(100)


class ifc_validation_task(task):
    est_time = 1

    def execute(self, directory, id):
        ofn = os.path.join(directory, id + ".validate.log.json")
        with open(ofn, "w") as f:
            subprocess.call([sys.executable, "-m", "ifcopenshell.validate", utils.storage_file_for_id(id, "ifc"), "--json"], cwd=directory, stdout=f)
        utils.store_file(id, extension="validate.log.json")


class xml_generation_task(task):
    est_time = 1

    def execute(self, directory, id):
        subprocess.call([IFCCONVERT, utils.storage_file_for_id(id, "ifc"), id + ".xml", "-yv"], cwd=directory, stdout=subprocess.PIPE)
        utils.store_file(id, extension="xml")


class geometry_generation_task(task):
    est_time = 10

    def execute(self, directory, id):
        # @todo store this log in a separate file?
        proc = subprocess.Popen([
            IFCCONVERT,
            utils.storage_file_for_id(id, "ifc"),
            id + ".glb",
            "-qyv",
            "--log-format", "json",
            "--log-file", id + ".geometry.log.json",
            "--exclude", "entities", "IfcOpeningElement", "IfcSpace", "IfcFlowTerminal", "IfcDiscreteAccessory"
        ], cwd=directory, stdout=subprocess.PIPE)
        i = 0
        while True:
            ch = proc.stdout.read(1)

            if not ch and proc.poll() is not None:
                break

            if ch and ord(ch) == ord('.'):
                i += 1
                self.sub_progress(i)

        # GLB generation is mandatory to succeed
        if proc.poll() != 0:
            raise RuntimeError()
            
        utils.store_file(id, extension="glb")
        utils.store_file(id, extension="geometry.log.json")

                
class glb_optimize_task(task):
    est_time = 1
    
    def execute(self, directory, id):
        try:
            if subprocess.call(["gltf-pipeline" + ('.cmd' if on_windows else ''), "-i", id + ".glb", "-o", id + ".optimized.glb", "-b", "-d"], cwd=directory) == 0:
                os.rename(os.path.join(directory, id + ".glb"), os.path.join(directory, id + ".unoptimized.glb"))
                os.rename(os.path.join(directory, id + ".optimized.glb"), os.path.join(directory, id + ".glb"))
        except FileNotFoundError as e:
            pass
            
        utils.store_file(id, extension="glb")
        utils.store_file(id, extension="unoptimized.glb")


class gzip_task(task):
    est_time = 1
    order = 2000
    
    def execute(self, directory, id):
        import gzip
        for ext in ["glb", "xml", "svg"]:
            fn = os.path.join(directory, id + "." + ext)
            if os.path.exists(fn):
                with open(fn, 'rb') as orig_file:
                    with gzip.open(fn + ".gz", 'wb') as zipped_file:
                        zipped_file.writelines(orig_file)
                        
                utils.store_file(id, extension=ext + ".gz")
                        
                        
class svg_rename_task(task):
    """
    In case of an upload of multiple files copy the SVG file
    for an aspect model with [\w+]_[0-9].svg to [\w+].svg if
    and only if the second file does not exist yet or the
    first file is larger in terms of file size.
    """
    
    est_time = 1
    order = 1000
    
    def execute(self, directory, id):
        svg1_fn = os.path.join(directory, id + ".svg")
        svg2_fn = os.path.join(directory, id.split("_")[0] + ".svg")
        
        if os.path.exists(svg1_fn):
            if not os.path.exists(svg2_fn) or os.path.getsize(svg1_fn) > os.path.getsize(svg2_fn):
                shutil.copyfile(svg1_fn, svg2_fn)
                
        utils.store_file(id.split("_")[0], extension="svg")


class svg_generation_task(task):
    est_time = 10

    def execute(self, directory, id):
        proc = subprocess.Popen([IFCCONVERT, utils.storage_file_for_id(id, "ifc"), id + ".svg", "-qy", "--plan", "--model", "--section-height-from-storeys", "--door-arcs", "--print-space-names", "--print-space-areas", "--bounds=1024x1024", "--include", "entities", "IfcSpace", "IfcWall", "IfcWindow", "IfcDoor", "IfcAnnotation"], cwd=directory, stdout=subprocess.PIPE)
        i = 0
        while True:
            ch = proc.stdout.read(1)

            if not ch and proc.poll() is not None:
                break

            if ch and ord(ch) == ord('.'):
                i += 1
                self.sub_progress(i)
                
        utils.store_file(id, extension="svg")


class replace_detailed_elements_task(task):
    est_time = 1

    def execute(self, directory, id):
        fn0 = utils.storage_file_for_id(id, 'ifc')
        fn1 = utils.storage_file_for_id(id, 'tmp', ensure=False)
        fn2 = utils.storage_file_for_id(id, 'before.replacement.ifc', ensure=False)
        
        if os.path.getsize(fn0) > 100 * 1024 * 1024:
            run(sys.executable, REPLACE_DETAILED_ELEMENTS, fn0, fn1, cwd=directory)
            
            os.rename(fn0, fn2)
            os.rename(fn1, fn0)

            utils.store_file(id, extension="before.replacement.ifc")
            utils.store_file(id, extension="ifc")


def escape_string(s):
    """
    Escapes the characters "\", "'", and '"' in a string by adding a backslash before them.
    """
    escape_chars = ['\\', '\'', '\"']
    for char in escape_chars:
        s = s.replace(char, '\\' + char)
    return s


def do_process(id, translation=None):
    # @todo
    input_files = [utils.storage_file_for_id(id, "ifc")]
    
    if translation:
        if translation == 'auto':
            proc = subprocess.Popen([
                sys.executable,
            ], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            stdo, _ = proc.communicate(input=("""
import ifcopenshell
f = ifcopenshell.open('%(fn)s')
negate = lambda x: -x
print(*map(negate, f.by_type("IfcSite")[0].ObjectPlacement.RelativePlacement.Location.Coordinates))
""" % {'fn': escape_string(input_files[0])}).encode('ascii'))
            translation = dict(zip("xyz", map(float, stdo.decode('ascii').split(' '))))
            print("Translation auto =", *translation.items())
            
        for f in input_files:
            os.rename(f, f + ".old")
            proc = subprocess.Popen([
                sys.executable,
            ], stdin=subprocess.PIPE)
            proc.communicate(input=("""
import ifcopenshell
f = ifcopenshell.open('%(fn)s.old')
sites = f.by_type('IfcSite')
if len(sites) == 1:
    lp = f.createIfcLocalPlacement(
        RelativePlacement = f.createIfcAxis2Placement3D(
            Location=f.createIfcCartesianPoint((%(x)f, %(y)f, %(z)f))
        )
    )
    sites[0].ObjectPlacement.PlacementRelTo = lp
else:
    pass
f.write('%(fn)s')
""" % {'fn':escape_string(f), **translation}).encode('ascii'))
            utils.store_file(id, extension="ifc")            
    
    d = utils.storage_dir_for_id(id, output=True)
    if not os.path.exists(d):
        os.makedirs(d)

    tasks = [
        # ifc_validation_task,
        replace_detailed_elements_task,
        xml_generation_task,
        geometry_generation_task,
        # svg_generation_task,
        glb_optimize_task,
        gzip_task
    ]
    
    tasks_on_aggregate = []
    
    is_multiple = any("_" in n for n in input_files)
    # if is_multiple:
    #     tasks.append(svg_rename_task)
    
    """
    # Create a file called task_print.py with the following
    # example content to add application-specific tasks

    import sys
    
    from worker import task as base_task
    
    class task(base_task):
        est_time = 1    
        
        def execute(self, directory, id):
            print("Executing task 'print' on ", id, ' in ', directory, file=sys.stderr)
    """
    
    for fn in glob.glob("task_*.py"):
        mdl = importlib.import_module(fn.split('.')[0])
        if getattr(mdl.task, 'aggregate_model', False):
            tasks_on_aggregate.append(mdl.task)
        else:
            tasks.append(mdl.task)
        
    tasks.sort(key=lambda t: getattr(t, 'order', 10))
    tasks_on_aggregate.sort(key=lambda t: getattr(t, 'order', 10))

    elapsed = 0
    set_progress(id, elapsed)
    
    n_files = len(input_files)
    
    total_est_time = \
        sum(map(operator.attrgetter('est_time'), tasks)) * n_files + \
        sum(map(operator.attrgetter('est_time'), tasks_on_aggregate))
        
    def run_task(t, args, aggregate_model=False):
        nonlocal elapsed
        begin_end = (elapsed / total_est_time * 99, (elapsed + t.est_time) / total_est_time * 99)
        task = t(begin_end)
        try:
            task(d, *args)
        except:
            traceback.print_exc(file=sys.stdout)
            # Mark ID as failed
            with open(os.path.join(d, 'failed'), 'w') as f:
                pass
            return False
        elapsed += t.est_time
        return True
    
    for i in range(n_files):
        for t in tasks:
            if not run_task(t, ["%s_%d" % (id, i) if is_multiple else id]):
                break
        # to break out of nested loop
        else: continue
        break
            
    for t in tasks_on_aggregate:
        run_task(t, [id, input_files], aggregate_model=True)

    elapsed = 100
    set_progress(id, elapsed)


@queue_task
def process(id, callback_url, translation=None, **kwargs):
    do_process(id, translation=translation)


def assert_ifc_type(fn, ifc_type):
    proc = subprocess.Popen([
        sys.executable,
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate(input=("""
import ifcopenshell
f = ifcopenshell.open('%s')
exit(2 if len(f.by_type('%s')) == 0 else 0)
""" % (escape_string(fn), ifc_type)).encode('ascii'))
    for ln in (out+err).decode('utf-8').split('\n'):
        print(ln)
    if proc.returncode == 1:
        raise RuntimeError(f"Checking for existance on {ifc_type} on {fn} failed")
    return proc.returncode == 0


def empty_result(d, id, data=None):
    try: os.makedirs(d)
    except: pass
    with open(os.path.join(d, id + '.json'), 'w') as f:
        json.dump(data if data is not None else {
            'id': id,
            'results': []
        }, f)
    utils.store_file(id, extension="json")
    set_progress(id, 100)


@queue_task
def calculate_volume(id, config, **kwargs):
    
    def make_script(args):
        return """file = parse("*.ifc")
all_surfaces = create_geometry(file, exclude={"IfcSpace", "IfcOpeningElement", "IfcFlowTerminal", "IfcDiscreteAccessory"})
voxels = voxelize(all_surfaces)
external = exterior(voxels)
internal = invert(external)
plane_surf = plane(internal, 0.0, 0.0, 1.0, %f)
plane_voxels = voxelize(plane_surf)
two_components = subtract(internal, plane_voxels)
x = describe_components("components.json", two_components)
y = json_stats("internal.json", {"internal"})
""" % args['plane_z']
    
    def prepare(files):
        plane_z = 0.0
        
        if len(files) == 1:
            proc = subprocess.Popen([
                sys.executable,
            ], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            stdout, stderr = proc.communicate(input=("""
import ifcopenshell
from ifc_utils import get_unit
f = ifcopenshell.open('%s')
lu = get_unit(f, "LENGTHUNIT", 1.0)
s = f.by_type('IfcSite')[0]
if s.ObjectPlacement and s.ObjectPlacement.PlacementRelTo:
    print(s.ObjectPlacement.PlacementRelTo.RelativePlacement.Location.Coordinates[2] * lu)
""" % escape_string(files[0][1][0])).encode('ascii'))
            r = stdout.strip()
            if r:
                plane_z = -float(r)
                print("Using plane 0.0 0.0 1.0", plane_z)
        return {
            'plane_z': plane_z
        }
        
    def process(args, context : voxel_execution_context):           
        components = context.get_json("components.json")
        
        parse_bounds = lambda d: [tuple(map(float, s[1:-1].split(', '))) for s in d.get('world').split(' - ')]
        get_z_min = lambda d: parse_bounds(d)[0][2]
        to_volume = lambda c: c * context.voxel_size**3
        
        counts = list(map(to_volume, map(float, map(operator.itemgetter('count'), sorted(components, key=get_z_min)))))
        
        if len(counts) == 2:
            under, above = counts
        else:
            under = 0.
            above = sum(counts)
        
        with open(os.path.join(context.task_path, id + ".json"), 'w') as f:
            json.dump({
                'under_ground': under,
                'above_ground': above
            }, f)
            
        utils.store_file(id, extension="json")

    process_voxel_check(
        make_script,
        process,
        {},
        id,
        config['ids'],
        prepare=prepare
    )

def make_script_3_4(args):
    basis = """file = parse("*.ifc")
all_surfaces = create_geometry(file, exclude={"IfcSpace", "IfcOpeningElement", "IfcFlowTerminal", "IfcDiscreteAccessory"})
voxels = voxelize(all_surfaces)
spaces = create_geometry(file, include={"IfcSpace"})
space_ids_unaligned = voxelize(spaces, type="uint", method="volume")
space_ids_empty = constant_like(voxels, 0, type="uint")
space_ids = union(space_ids_unaligned, space_ids_empty)
free(space_ids_unaligned)
space_voxels = copy(space_ids, type="bit")
headroom = subtract(space_voxels, voxels)
headroom_height = collapse_count(headroom, 0, 0, -1)
space_footprint = collapse(space_voxels, 0, 0, -1)
space_footprint_offset = offset(space_footprint)
space_footprint_thick = union(space_footprint, space_footprint_offset)
space_footprint_thick_offset = offset(space_footprint_thick)
space_footprint_thicker = union(space_footprint_thick, space_footprint_thick_offset)
headroom_height_footprint = intersect(headroom_height, space_footprint_thicker)
free(headroom_height)
free(space_footprint_thicker)
free(space_footprint_thick_offset)
free(space_footprint_thick)
free(space_footprint)
free(space_footprint_offset)
"""
    
    threshold = """headroom_height_footprint_%(n)d = greater_than(headroom_height_footprint, %(n)d)
x = describe_group_by("data_%(n)d.json", headroom_height_footprint_%(n)d, space_ids, only_counts=1)
"""
    
    return ''.join((basis,) + tuple(threshold % {'n': t / 0.05} for t in args.get('thresholds', [])))


def process_3_4(args, context):
    import ifcopenshell
    
    space_guid_mapping = {}
    for fn in context.files:
        f = ifcopenshell.open(fn)
        for sp in f.by_type("IfcSpace"):
            space_guid_mapping[sp.id()] = sp.GlobalId
    
    d = defaultdict(list)
    
    # @todo the logic here needs to be reversed. starting from the
    # spaces so that when higher thresholds cause spaces to be left
    # out we can put zero in the list.
    
    for t in args.get('thresholds', []):
        n = t / context.voxel_size
        for di in context.get_json("data_%(n)d.json" % locals()):
            sid = int(di.get('id'))
            sgd = space_guid_mapping.get(sid)
            if sgd is None: continue
            # @todo ^ this appears to be only for 0
            # @note empiric value for getting results closer to geometric surface area
            # @todo don't do simple counts, but proper half counting for boundaries
            d[sgd].append((float(di.get('count')) * context.voxel_size ** 2) / 0.95)
    
    context.put_json(context.id + '.json', {
        'space_heights': d,
        'thresholds': args.get('thresholds', []),
        'id': context.id
    })
    
    utils.store_file(context.id, extension="json")
    
def make_script_3_26(entity, args):
    h = args['height']
    h = int(h / 0.05)
    
    basis = """function get_reachability(file)

    surfaces = create_geometry(file, exclude={"IfcOpeningElement", "IfcDoor", "IfcSpace", "IfcFlowTerminal", "IfcDiscreteAccessory"})
    surface_voxels = voxelize(surfaces)

    slabs = create_geometry(file, include={"IfcSlab"})
    slab_voxels_region = voxelize(slabs)

    doors = create_geometry(file, include={"IfcDoor"})
    door_voxels_region = voxelize(doors)

    empty = constant_like(surface_voxels, 0, type="bit")
    slab_voxels = union(empty, slab_voxels_region)
    door_voxels = union(empty, door_voxels_region)

    walkable = shift(slab_voxels, dx=0, dy=0, dz=1)
    walkable_minus = subtract(walkable, slab_voxels)
    walkable_seed = intersect(door_voxels, walkable_minus)

    surfaces_sweep = sweep(surface_voxels, dx=0, dy=0, dz=0.5)
    surfaces_padded = offset_xy(surface_voxels, 0.1)
    surfaces_obstacle = sweep(surfaces_padded, dx=0, dy=0, dz=-0.5)
    walkable_region = subtract(surfaces_sweep, surfaces_obstacle)
    walkable_seed_real = subtract(walkable_seed, surfaces_padded)
    reachable = traverse(walkable_region, walkable_seed_real)

return reachable

file = parse("*.ifc")
all_surfaces = create_geometry(file, exclude={"IfcSpace", "IfcOpeningElement", "IfcFlowTerminal", "IfcDiscreteAccessory"})
voxels = voxelize(all_surfaces)

stairs = create_geometry(file, include={"%s"})
stair_ids_region = voxelize(stairs, type="uint", method="surface")
stair_ids_empty = constant_like(voxels, 0, type="uint")
stair_ids = union(stair_ids_region, stair_ids_empty)
stair_ids_offset = shift(stair_ids, dx=0, dy=0, dz=1)

stair_voxels_region = voxelize(stairs)
stair_voxels_empty = constant_like(voxels, 0)
stair_voxels = union(stair_voxels_region, stair_voxels_empty)

railings = create_geometry(file, include={"IfcRailing"}, optional=1)
railing_voxels_orig = voxelize(railings)
railing_voxels_down = sweep(railing_voxels_orig, dx=0.0, dy=0.0, dz=-1.0)
stair_voxels_wo_railing = subtract(stair_voxels, railing_voxels_orig)

stair_offset = shift(stair_voxels_wo_railing, dx=0, dy=0, dz=1)
stair_offset_min_1 = subtract(stair_offset, stair_voxels_wo_railing)
stair_offset_min = subtract(stair_offset_min_1, railing_voxels_down)
extrusion = sweep(stair_voxels_wo_railing, dx=0.0, dy=0.0, dz=-0.4)
stair_top = subtract(stair_offset_min, extrusion)

reachable = get_reachability(file)
stair_top_reachable = intersect(stair_top, reachable, if_non_empty=1)

space = sweep(stair_top_reachable, dx=0, dy=0, dz=1, until=voxels, max=%d)
cnt = collapse_count(space, dx=0, dy=0, dz=-1)

valid = greater_than(cnt, %d)
invalid = less_than(cnt, %d)

mesh(valid, "valid.obj", groups=stair_ids_offset)
mesh(invalid, "invalid.obj", groups=stair_ids_offset)
describe_group_by("results.json", cnt, groups=stair_ids_offset, use_bits=0)
""" % (entity, h + 1, h, h + 1)

    return basis

def process_3_26(entity, args, context):
    import ifcopenshell
    
    stair_guid_mapping = {}
    
    # note that this also includes the decomposed element id
    stair_to_children = defaultdict(list)
    
    for fn in context.files:
        f = ifcopenshell.open(fn)
        for st in f.by_type(entity):
            stair_guid_mapping[st.id()] = st.GlobalId
            stair_to_children[st.GlobalId].append(st.id())
            
            for rel in st.IsDecomposedBy:
                for ch in rel.RelatedObjects:
                    stair_guid_mapping[ch.id()] = st.GlobalId
                    stair_to_children[st.GlobalId].append(ch.id())
    
    d = os.path.join(context.task_path, 'tmp')
    os.makedirs(d)
    
    context.get_file('valid.obj', target=os.path.join(d, 'valid.obj'))
    context.get_file('invalid.obj', target=os.path.join(d, 'invalid.obj'))
    components = context.get_json('results.json')
    
    min_height_component = {}
    for di in components:
        min_height_component[int(di["id"])] = float(int(di["min_value"])) * context.voxel_size
    
    with open(os.path.join(d, 'colours.mtl'), 'w') as f:
        f.write("newmtl red\n")
        f.write("Kd 1.0 0.0 0.0\n\n")
        f.write("newmtl green\n")
        f.write("Kd 0.0 1.0 0.0\n\n")
        
    def simplify():
        for fn in glob.glob(os.path.join(d, "*.obj")):
            fn2 = fn[:-4] + "_s.obj"
            run(sys.executable, "simplify_obj.py", fn, fn2)
            
            with open(fn2, 'r+') as f:
                ls = f.readlines()
                ls.insert(0, "mtllib colours.mtl\n")
                if "invalid" in fn:
                    ls.insert(1, "usemtl red\n")
                else:
                    ls.insert(1, "usemtl green\n")
                f.seek(0)
                f.truncate()
                # strip empty faces
                f.writelines([l for l in ls if not l.startswith("f") or len(l.split(" ")) > 2])
            
            yield fn2
        
    # split mesh into separate DAE files
    run("blender", "-b", "-P", "convert.py", "--split", "--orient", "--", *simplify(), os.path.join(d, "%s.dae"))
    
    collected = []
    min_height_by_guid = {}
    
    # join DAEs based on decomposition in IFC
    for iii, (gd, ch_ids) in enumerate(stair_to_children.items()):
        # note ch_ids also includes decomposed element id
        fns = [os.path.join(d, "%d.dae" % cid) for cid in ch_ids]
        fns = list(filter(os.path.exists, fns))
        
        min_height = None
        min_height_ids = set(min_height_component.keys()) & set(ch_ids)
        if not min_height_ids:
            print("WARNING: no min height recorded for", gd)
        else:
            min_height = min(map(min_height_component.__getitem__, min_height_ids))
        min_height_by_guid[gd] = min_height
        
        if len(fns) == 0:
            continue
        elif len(fns) == 1:
            collected.append((gd, fns[0]))
        else:
            joined = os.path.join(d, "%03d.dae" % iii)
            run("blender", "-b", "-P", "convert.py", "--", *fns, joined)
            collected.append((gd, joined))


    # check for red material name -> error
    # convert to glTF        
    def convert():
        for i, (gd, fn) in enumerate(collected):
            error = "red" in open(fn).read()
            id = int(os.path.basename(fn)[:-4])
            fn2 = "../" + context.id + "_%d" % i + ".glb"
            run("COLLADA2GLTF-bin", "-i", fn, "-o", fn2, "-b", "1", cwd=d)
            utils.store_file(context.id, filename=f"{context.id}_{i}.glb")
            yield i, error, gd


    # format result dict
    def create_issue(tup):
        i, is_error, g = tup
        return {
            "visualization": "/run/%s/result/resource/gltf/%d.glb" % (context.id, i),
            "status": ["NOTICE", "ERROR"][is_error],
            "guid": g,
            "minHeight": min_height_by_guid[g] if is_error else None
        }
        
    context.put_json(context.id + '.json', {
        'id': context.id,
        'results': list(map(create_issue, convert()))
    })
    
    utils.store_file(context.id, extension="json")


def make_script_connectivity_graph(exclude_external, args):
    base = """file = parse("*.ifc")
surfaces = create_geometry(file, exclude={"IfcOpeningElement", "IfcDoor", "IfcSpace", "IfcFlowTerminal", "IfcDiscreteAccessory"})
slabs = create_geometry(file, include={"IfcSlab"})
doors = create_geometry(file, include={"IfcDoor"})
door_filter = filter_properties(file, IsExternal=1)
external_doors = create_geometry(door_filter, include={"IfcDoor"})
all_surfaces = create_geometry(file, exclude={"IfcOpeningElement", "IfcSpace", "IfcFlowTerminal", "IfcDiscreteAccessory"})

surface_voxels_region = voxelize(surfaces)
slab_voxels_region = voxelize(slabs)
door_voxels_region = voxelize(doors)
voxels = voxelize(all_surfaces)
external_door_voxels_thin = voxelize(external_doors)
external_door_voxels_layer = offset(external_door_voxels_thin)
external_door_voxels = union(external_door_voxels_thin, external_door_voxels_layer)

empty = constant_like(voxels, 0, type="bit")
surface_voxels = union(empty, surface_voxels_region)
slab_voxels = union(empty, slab_voxels_region)
door_voxels_thin = union(empty, door_voxels_region)
door_voxels_layer = offset(door_voxels_thin)
door_voxels = union(door_voxels_thin, door_voxels_layer)

walkable = shift(slab_voxels, dx=0, dy=0, dz=1)
walkable_minus = subtract(walkable, slab_voxels)
surfaces_sweep = sweep(surface_voxels, dx=0, dy=0, dz=0.5)
surfaces_padded = offset_xy(surface_voxels, 0.1)
surfaces_obstacle = sweep(surfaces_padded, dx=0, dy=0, dz=-0.5)
walkable_region = subtract(surfaces_sweep, surfaces_obstacle)
walkable_seed = intersect(door_voxels, walkable_minus)
walkable_seed_real = intersect(walkable_seed, walkable_region)
reachable = traverse(walkable_region, walkable_seed_real)
reachable_shifted = shift(reachable, dx=0, dy=0, dz=1)
reachable_bottom = subtract(reachable, reachable_shifted)

seed = intersect(reachable, external_door_voxels)
flow = traverse(reachable, seed, connectedness=26, type="uint")
flow_bottom = intersect(flow, reachable_bottom)
"""

    epilogue_1 = """export_csv(flow_bottom, "flow.csv")
"""

    epilogue_2 = """low_res = resample(voxels, -4)
external = exterior(low_res)
flow_bit = copy(flow_bottom, type="bit")
flow_low_res = resample(flow_bit, -4)

external_offset = offset(external)
external_larger = union(external, external_offset)

internal = invert(external_larger)
cull_away_low_res = subtract(flow_low_res, internal)
cull_away_hi_res = resample(cull_away_low_res, 4)

flow_bottom_internal = subtract(flow_bottom, cull_away_hi_res)

export_csv(flow_bottom_internal, "flow.csv")
mesh(cull_away_hi_res, "cull_away_hi_res.obj")
"""

    print(exclude_external, exclude_external == "exclude_external")
    
    return base + (epilogue_2 if exclude_external == "exclude_external" else epilogue_1)

def process_connectivity_graph(command, args, context):
    context.get_file('flow.csv', target=os.path.join(context.task_path, 'flow.csv'))
    
    run(
        sys.executable,
        os.path.abspath(os.path.join(os.path.dirname(__file__), 'connectivity_graph.py')),
        context.id,
        repr(context.files),
        command,
        repr(args),
        cwd=context.task_path)
    
    # store json and gltfs
    utils.store_file(context.id, extension="json")
    for fn in glob.glob(os.path.join(context.task_path, "*.glb")):
        utils.store_file(context.id, filename=os.path.relpath(fn, context.task_path))


def process_non_voxel_check(command, args, id, ids, **kwargs):
    set_progress(id, 0)
    
    files = [utils.ensure_file(f, "ifc") for f in ids]
    
    path = utils.storage_dir_for_id(id, output=True)
    os.makedirs(path, exist_ok=True)

    run(
        sys.executable,
        os.path.abspath(os.path.join(os.path.dirname(__file__), command + '.py')),
        id,
        *map(str, args),
        *files,
        cwd=path)
    
    set_progress(id, 100)

    # store json and gltfs
    utils.store_file(id, extension="json")
    for fn in glob.glob(os.path.join(path, "*.glb")):
        utils.store_file(id, filename=os.path.relpath(fn, path))



def make_script_safety_barriers(args):
    return """file = parse("*.ifc")
surfaces = create_geometry(file, exclude={"IfcOpeningElement", "IfcDoor", "IfcSpace", "IfcFlowTerminal", "IfcDiscreteAccessory"})
slabs = create_geometry(file, include={"IfcSlab"})
doors = create_geometry(file, include={"IfcDoor"})

elevator_filter = filter_properties(file, nnn_Kategooria="Liftiruum")
elevator_shaft = create_geometry(elevator_filter, include={"IfcSpace"}, optional=1)
elevator_shaft_voxels = voxelize(elevator_shaft)

surface_voxels_excl = voxelize(surfaces)
surface_voxels = union(surface_voxels_excl, elevator_shaft_voxels)

slab_voxels_region = voxelize(slabs)
door_voxels_region = voxelize(doors)

empty = constant_like(surface_voxels, 0, type="bit")
slab_voxels = union(empty, slab_voxels_region)
door_voxels = union(empty, door_voxels_region)

walkable = shift(slab_voxels, dx=0, dy=0, dz=1)
walkable_minus = subtract(walkable, slab_voxels)
walkable_seed = intersect(door_voxels, walkable_minus)
surfaces_sweep = sweep(surface_voxels, dx=0, dy=0, dz=0.5)
surfaces_padded = offset_xy(surface_voxels, 0.1)
surfaces_obstacle = sweep(surfaces_padded, dx=0, dy=0, dz=-0.5)
walkable_region = subtract(surfaces_sweep, surfaces_obstacle)
walkable_seed_real = subtract(walkable_seed, surfaces_padded)
reachable = traverse(walkable_region, walkable_seed_real)
reachable_shifted = shift(reachable, dx=0, dy=0, dz=1)
reachable_bottom = subtract(reachable, reachable_shifted)
reachable_padded = offset_xy(reachable_bottom, 0.2)
full = constant_like(surface_voxels, 1)
surfaces_sweep_1m = sweep(surface_voxels, dx=0, dy=0, dz=1.0)
deadly = subtract(full, surfaces_sweep_1m)
really_reachable = subtract(reachable_padded, surfaces_obstacle)
result = intersect(really_reachable, deadly)
mesh(result, "result.obj")
"""


def process_safety_barriers(element_type, args, context):
    context.get_file('result.obj', target=os.path.join(context.task_path, 'result.obj'))

    run(
        sys.executable,
        "simplify_obj.py",
        os.path.join(context.task_path, 'result.obj'),
        os.path.join(context.task_path, 'simplified.obj')
    )
    
    run(
        sys.executable,
        os.path.abspath(os.path.join(os.path.dirname(__file__), 'annotate_safety_barriers.py')),
        context.id,
        repr(context.files),
        element_type,
        cwd=context.task_path)
    
    # store json and gltfs
    utils.store_file(context.id, extension="json")
    for fn in glob.glob(os.path.join(context.task_path, "*.glb")):
        utils.store_file(id, filename=os.path.relpath(fn, context.task_path))

    
class voxel_execution_context:
    def __init__(self, id, files, command):
        self.id = id
        self.files = files
        self.task_path = Path(utils.storage_dir_for_id(id, output=True))
        self.path = self.task_path / "voxel"
        self.path.mkdir()
        for fn in self.files:
            os.link(fn, str(self.path / os.path.basename(fn)))
        with (self.path / "voxelfile.txt").open('w') as f:
            f.write(command)

    def get_progress(self):
        try:
            p = (self.path / "progress").stat().st_size
        except: p = 0
        return p
    
    def get_log(self):
        lines = []
        try:
            with (self.path / "log.json").open() as f:
                for l in f:
                    try: lines.append(json.loads(l))
                    except Exception as e: print(e)
        except Exception as e: print(e)
        return lines

    def run(self, **args):
        self.voxel_size = args['size']
    
        def make_args(d):
            for kv in d.items():
                if kv[1] is True:
                    yield "--%s" % kv[0]
                else:
                    yield "--%s=%s" % kv
    
        def inner():
            f = (self.path / "progress").open("wb")
            print(VOXEC, "voxelfile.txt", "-q", "--no-vox", "--log-file", "log.json", *make_args(args or {}))
            process = subprocess.Popen(
                [VOXEC, "voxelfile.txt", "-q", "--no-vox", "--log-file", "log.json", *make_args(args or {})],
                cwd=self.path,
                stdout = f
            )

            if rlim_s := os.environ.get('VOXEC_RLIMIT_AS_MB'):
                pid = process.pid
                rl = int(rlim_s) * 1024 * 1024
                psutil.Process(pid).rlimit(psutil.RLIMIT_AS, (rl, rl))

            self.returncode = process.wait()
            
        thread = threading.Thread(target=inner)
        thread.start()
        while thread.is_alive():
            progress = self.get_progress() / 100. * 95.
            set_progress(self.id, progress)
            
            try:
                utils.store_file(self.id, filename="voxel/log.json")
            except: pass
            
            time.sleep(1.)

        def wrap_try(fn, default=None):
            def inner(*args):
                try:
                    return fn(*args)
                except:
                    return default
            return inner
            
        if self.returncode in (-6, -9):
            raise RuntimeError(f'insufficient resources')

        msgs = []
        try:
            msgs = list(filter(None, map(wrap_try(json.loads), list(open(os.path.join(utils.storage_dir_for_id(self.id), "voxel/log.json"))))))
        except: pass
        if msgs:
            if msgs[-1].get('severity') == 'fatal':
                # @todo add some assertions to script
                s = msgs[-1].get('error', {}).get('message')
                if not s:
                    s = msgs[-1]['message']
                raise RuntimeError(s)
            elif not msgs[-1].get('message', '').startswith("script finished"):
                raise RuntimeError(f'exit code {self.returncode}')
        else:
            raise RuntimeError(f'empty output exit code {self.returncode}')

    def get_file(self, path, target=None):
        s = (self.path / path).open('rb').read()
        if target:
            with open(target, 'wb') as f:
                f.write(s)
        else:
            return s
        
    def get_json(self, path):
        return json.loads(self.get_file(path))
    
    def put_json(self, path, obj):
        with (self.task_path / path).open('w') as f:
            json.dump(obj, f)
    

def process_voxel_check(script_fn, process_fn, args, id, files, **kwargs):

    from multiprocessing import cpu_count

    d = utils.storage_dir_for_id(id, output=True)
    os.makedirs(d, exist_ok=True)

    files = [utils.ensure_file(f, "ifc") for f in files]
    
    if p_fn := kwargs.get('prepare'):
        args.update(p_fn(files))

    command = script_fn(args)
    values = {
        'threads': cpu_count() // int(os.environ.get('NUM_WORKERS', '1')),
        'chunk': 16,
        'size': kwargs.get('size', 0.05)
    }
    
    context = voxel_execution_context(id, files, command)
    context.run(**values)
    process_fn(args, context)
    
    set_progress(id, 100)
        
        
def abort(id):
    print("Invalid arguments")
    set_progress(id, -2)
    set_error("Invalid arguments")


@queue_task
def space_heights(id, config, **kwargs):
    thresholds = config.get(
        'thresholds',
        # [0.0,1.6,1.8,2.0,2.1,2.2,2.3]
        [0.0,1.6]
    )
    
    try:
        thresholds = list(map(float, thresholds))
    except:
        return abort(id)
        
    files = [utils.ensure_file(f, "ifc") for f in config['ids']]
    d = utils.storage_dir_for_id(id, output=True)

    if not any(map(functools.partial(assert_ifc_type, ifc_type="IfcSpace"), files)):
        return empty_result(d, id, {
            'space_heights': {},
            'thresholds': thresholds,
            'id': id
        })
    
    process_voxel_check(
        make_script_3_4,
        process_3_4,
        {'thresholds': thresholds},
        id,
        config['ids'],
        # size=0.10,
        **kwargs)

@queue_task
def stair_headroom(id, config, **kwargs):
    height = config.get('height', 2.2)
    
    try:
        height = float(height)
    except:
        return abort(id)
        
    process_voxel_check(
        functools.partial(make_script_3_26, "IfcStair"),
        functools.partial(process_3_26, "IfcStair"),
        {'height': height},
        id,
        config['ids'],
        **kwargs)


@queue_task
def ramp_headroom(id, config, **kwargs):
    height = config.get('height', 2.2)
    
    try:
        height = float(height)
    except:
        return abort(id)

    files = [utils.ensure_file(f, "ifc") for f in config['ids']]
    d = utils.storage_dir_for_id(id, output=True)
        
    if not any(map(functools.partial(assert_ifc_type, ifc_type="IfcRamp"), files)):
        return empty_result(d, id)

    process_voxel_check(
        functools.partial(make_script_3_26, "IfcRamp"),
        functools.partial(process_3_26, "IfcRamp"),
        {'height': height},
        id,
        config['ids'],
        **kwargs)


@queue_task
def door_direction(id, config, **kwargs):

    files = [utils.ensure_file(f, "ifc") for f in config['ids']]
    d = utils.storage_dir_for_id(id, output=True)
        
    if not any(map(functools.partial(assert_ifc_type, ifc_type="IfcDoor"), files)):
        return empty_result(d, id)

    process_voxel_check(
        functools.partial(make_script_connectivity_graph, "exclude_external"),
        functools.partial(process_connectivity_graph, "doors"),
        {},
        id,
        config['ids'],
        **kwargs)


@queue_task
def landings(id, config, **kwargs):
    length = config.get('length', 1.5)

    try:
        length = float(length)
    except:
        return abort(id)


    process_voxel_check(
        functools.partial(make_script_connectivity_graph, "include_external"),
        functools.partial(process_connectivity_graph, "landings"),
        {'length': length * 0.5},
        id,
        config['ids'],
        **kwargs)


@queue_task
def risers(id, config, **kwargs):
    process_voxel_check(
        functools.partial(make_script_connectivity_graph, "include_external"),
        functools.partial(process_connectivity_graph, "risers"),
        {},
        id,
        config['ids'],
        **kwargs)

@queue_task
def escape_routes(id, config, **kwargs):
    lengths = config.get('lengths', [30., 30.])
    objects = config.get('objects', None)
    
    if not isinstance(lengths, list) or len(lengths) != 2 or set(map(type, lengths)) != {float}:
        return abort(id)
        
    if objects is not None:
        if not isinstance(objects, list):
            return abort(id)
        
    files = [utils.ensure_file(f, "ifc") for f in config['ids']]
    d = utils.storage_dir_for_id(id, output=True)

    if not any(map(functools.partial(assert_ifc_type, ifc_type="IfcSpace"), files)):
        return empty_result(d, id)

    process_voxel_check(
        functools.partial(make_script_connectivity_graph, "include_external"),
        functools.partial(process_connectivity_graph, "routes"),
        {'lengths': lengths, 'objects': objects},
        id,
        config['ids'],
        **kwargs)


@queue_task
def safety_barriers(id, config, **kwargs):
    element = config.get('element', 'IfcStair')
    
    if element not in {'IfcStair', 'IfcRamp', 'all'}:
        return abort(id)

    process_voxel_check(
        make_script_safety_barriers,
        functools.partial(process_safety_barriers, element),
        {},
        id,
        config['ids'],
        **kwargs)


@queue_task
def entrance_area(id, config, **kwargs):
    width = config.get('width', 1.5)
    depth = config.get('depth', 1.5)

    try:
        width = float(width)
        depth = float(depth)
    except:
        return abort(id)
        
    process_voxel_check(
        functools.partial(make_script_connectivity_graph, "exclude_external"),
        functools.partial(process_connectivity_graph, "entrance"),
        {width: width, depth:depth},
        id,
        config['ids'],
        **kwargs)
    

@queue_task
def ramp_percentage(id, config, **kwargs):
    warning = config.get('warning', 6)
    error = config.get('error', 10)

    try:
        warning = float(warning)
        error = float(error)
    except:
        return abort(id)
    
    process_non_voxel_check(
        'ramp_percentage',
        [warning, error],
        id,
        config['ids'],
        **kwargs)
