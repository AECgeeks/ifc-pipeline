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

import os
import sys
import json
import glob
import time
import platform
import traceback
import importlib
import subprocess
import tempfile
import operator
import shutil

import requests

on_windows = platform.system() == 'Windows'
ext = ".exe" if on_windows else ""
exe_path = os.path.join(os.path.dirname(__file__), "win" if on_windows else "nix")
IFCCONVERT = os.path.join(exe_path, "IfcConvert") + ext
if not os.path.exists(IFCCONVERT):
    IFCCONVERT = "IfcConvert"


import utils
import database


def set_progress(id, progress):
    session = database.Session()
   
    id = id.split("_")[0]

    model = session.query(database.model).filter(database.model.code == id).all()[0]
    model.progress = int(progress)
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
        ofn = os.path.join(directory, id + "_log.json")
        with open(ofn, "w") as f:
            subprocess.call([sys.executable, "-m", "ifcopenshell.validate", utils.storage_file_for_id(id, "ifc"), "--json"], cwd=directory, stdout=f)
        utils.store_file(id + "_log", "json")


class xml_generation_task(task):
    est_time = 1

    def execute(self, directory, id):
        subprocess.call([IFCCONVERT, utils.storage_file_for_id(id, "ifc"), id + ".xml", "-yv"], cwd=directory)
        utils.store_file(id, "xml")


class geometry_generation_task(task):
    est_time = 10

    def execute(self, directory, id):
        # @todo store this log in a separate file?
        proc = subprocess.Popen([IFCCONVERT, utils.storage_file_for_id(id, "ifc"), id + ".glb", "-qyv", "--log-format", "json", "--log-file", id + "_log.json"], cwd=directory, stdout=subprocess.PIPE)
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
            
        utils.store_file(id, "glb")
        utils.store_file(id + "_log", "json")

                
class glb_optimize_task(task):
    est_time = 1
    
    def execute(self, directory, id):
        try:
            if subprocess.call(["gltf-pipeline" + ('.cmd' if on_windows else ''), "-i", id + ".glb", "-o", id + ".optimized.glb", "-b", "-d"], cwd=directory) == 0:
                os.rename(os.path.join(directory, id + ".glb"), os.path.join(directory, id + ".unoptimized.glb"))
                os.rename(os.path.join(directory, id + ".optimized.glb"), os.path.join(directory, id + ".glb"))
        except FileNotFoundError as e:
            pass
            
        utils.store_file(id, "glb")
        utils.store_file(id, "unoptimized.glb")


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
                        
                utils.store_file(id, ext + ".gz")
                        
                        
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
                
        utils.store_file(id.split("_")[0], "svg")


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
                
        utils.store_file(id, "svg")


def do_process(id):
    # @todo
    input_files = [utils.storage_file_for_id(id, "ifc")]
    d = utils.storage_dir_for_id(id, output=True)
    if not os.path.exists(d):
        os.makedirs(d)

    tasks = [
        ifc_validation_task,
        xml_generation_task,
        geometry_generation_task,
        svg_generation_task,
        glb_optimize_task,
        gzip_task
    ]
    
    tasks_on_aggregate = []
    
    is_multiple = any("_" in n for n in input_files)
    if is_multiple:
        tasks.append(svg_rename_task)
    
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


def process(id, callback_url, **kwargs):
    try:
        do_process(id)
        status = "success"
    except Exception as e:
        traceback.print_exc(file=sys.stdout)
        status = "failure"        

    if callback_url is not None:       
        r = requests.post(callback_url, data={"status": status, "id": id})


def escape_routes(id, files, **kwargs):

        d = utils.storage_dir_for_id(id, output=True)
        os.makedirs(d)

        if kwargs.get('development'):
            VOXEL_HOST = "http://localhost:5555"
        else:
            VOXEL_HOST = "http://voxel:5000" 
            
        ESCAPE_ROUTE_LENGTH = 8.0
        
        files = [utils.ensure_file(f, "ifc") for f in files]
        files = [('ifc', (fn, open(fn))) for fn in files]

        command = """file = parse("*.ifc")
surfaces = create_geometry(file)
voxels = voxelize(surfaces, VOXELSIZE=0.05)
external = exterior(voxels)
shell = offset(external)
shell_inner_outer = offset(shell)
three_layers = union(shell, shell_inner_outer)
x = json_stats("vars.json")
xx = mesh(three_layers, "interior.obj")
"""
        
        command = """file = parse("*.ifc")
fire_door_filter = filter_attributes(file, OverallWidth=">1.2")
surfaces = create_geometry(file, exclude={"IfcOpeningElement", "IfcDoor", "IfcSpace"})
slabs = create_geometry(file, include={"IfcSlab"})
doors = create_geometry(file, include={"IfcDoor"})
fire_doors = create_geometry(fire_door_filter, include={"IfcDoor"})
surface_voxels = voxelize(surfaces)
slab_voxels = voxelize(slabs)
door_voxels = voxelize(doors)
fire_door_voxels = voxelize(fire_doors)
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
all_surfaces = create_geometry(file)
voxels = voxelize(all_surfaces)
external = exterior(voxels)
walkable_region_offset = offset_xy(walkable_region, 1)
walkable_region_incl = union(walkable_region, walkable_region_offset)
seed_external = intersect(walkable_region_incl, external)
seed_fire_doors = intersect(walkable_region_incl, fire_door_voxels)
seed = union(seed_external, seed_fire_doors)
safe = traverse(walkable_region_incl, seed, %(ESCAPE_ROUTE_LENGTH)f, connectedness=26)
safe_bottom = intersect(safe, reachable_bottom)
unsafe = subtract(reachable_bottom, safe)
safe_interior = subtract(safe_bottom, external)
x = mesh(unsafe, "unsafe.obj")
x = mesh(safe_interior, "safe.obj")
""" % locals()

        values = {'voxelfile': command}
        try:

            objfn_0 = os.path.join(d, id + "_0.obj")
            objfn_1 = os.path.join(d, id + "_1.obj")
            objfn_0_s = os.path.join(d, id + "_0_s.obj")
            objfn_1_s = os.path.join(d, id + "_1_s.obj")
            mtlfn = objfn_0[:-5] + '0.mtl'
            daefn = objfn_0[:-5] + '0.dae'
            glbfn = daefn[:-4] + '.glb'

            r = requests.post("%s/run" % VOXEL_HOST, files=files, data=values, headers={'accept':'application/json'})
            vid = json.loads(r.text)['id']
            
            # @todo store in db
            with open(os.path.join(d, "vid"), "w") as vidf:
                vidf.write(vid)
            
            while True:
                r = requests.get("%s/progress/%s" % (VOXEL_HOST, vid))
                progress = r.json()
                set_progress(id, progress)
                
                try:
                    r = requests.get("%s/log/%s" % (VOXEL_HOST, vid))
                    msgs = r.json()
                    json.dump(msgs, open(os.path.join(d, id + "_log.json"), "w"))
                    utils.store_file(id + "_log", "json")
                except: pass
                
                # @todo this a typo scripted -> script
                if len(msgs) and msgs[-1]['message'].startswith("scripted finished"):
                    break
                else:
                    time.sleep(1.)
            
            with open(mtlfn, 'w') as f:
                f.write("newmtl red\n")
                f.write("Kd 1.0 0.0 0.0\n\n")
                f.write("newmtl green\n")
                f.write("Kd 0.0 1.0 0.0\n\n")
                
            """
            r = requests.get("%s/run/%s/interior.obj" % (VOXEL_HOST, vid))
            r.raise_for_status()
            with open(objfn_0, 'w') as f:
                f.write('mtllib 0.mtl\n')
                f.write('usemtl red\n')
                f.write(r.text) 
            """            

            r = requests.get("%s/run/%s/unsafe.obj" % (VOXEL_HOST, vid))
            r.raise_for_status()
            with open(objfn_0, 'w') as f:
                f.write('mtllib %s_0.mtl\n' % id)
                f.write('usemtl red\n')
                f.write(r.text)
                
            r = requests.get("%s/run/%s/safe.obj" % (VOXEL_HOST, vid))
            r.raise_for_status()
            with open(objfn_1, 'w') as f:
                f.write('mtllib %s_0.mtl\n' % id)
                f.write('usemtl green\n')
                f.write(r.text)
            
            for fn in (objfn_0, objfn_1):
                subprocess.check_call([sys.executable, "simplify_obj.py", fn, fn[:-4] + "_s.obj"])

            subprocess.check_call(["blender", "-b", "-P", "convert.py", "--", objfn_0_s, objfn_1_s, daefn])
            
            subprocess.check_call(["COLLADA2GLTF-bin", "-i", daefn, "-o", glbfn, "-b", "1"])
            
            utils.store_file(id + "_0", "glb")
            
        except (requests.exceptions.ConnectionError, requests.exceptions.HTTPError) as e:
            traceback.print_exc()
