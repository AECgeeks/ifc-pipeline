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
import platform
import traceback
import importlib
import subprocess
import tempfile
import operator

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
    model = session.query(database.model).filter(database.model.code == id).all()[0]
    model.progress = int(progress)
    session.commit()
    session.close()


class task(object):
    def __init__(self, progress_map):
        print(self.__class__.__name__, progress_map)
        self.begin, self.end = progress_map

    def sub_progress(self, i):
        set_progress(self.id, self.begin + (self.end - self.begin) * i / 100.)

    def __call__(self, directory, id):
        self.id = id
        self.execute(directory, id)
        self.sub_progress(100)


class ifc_validation_task(task):
    est_time = 1

    def execute(self, directory, id):
        with open(os.path.join(directory, "log.json"), "w") as f:
            subprocess.call([sys.executable, "-m", "ifcopenshell.validate", id + ".ifc", "--json"], cwd=directory, stdout=f)


class xml_generation_task(task):
    est_time = 1

    def execute(self, directory, id):
        subprocess.call([IFCCONVERT, id + ".ifc", id + ".xml", "-yv"], cwd=directory)


class geometry_generation_task(task):
    est_time = 10

    def execute(self, directory, id):
        proc = subprocess.Popen([IFCCONVERT, id + ".ifc", id + ".glb", "-qyv", "--log-format", "json", "--log-file", "log.json"], cwd=directory, stdout=subprocess.PIPE)
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

                
class glb_optimize_task(task):
    est_time = 1
    
    def execute(self, directory, id):
        if subprocess.call(["gltf-pipeline" + ('.cmd' if on_windows else ''), "-i", id + ".glb", "-o", id + ".optimized.glb", "-b", "-d"], cwd=directory) == 0:
            os.rename(os.path.join(directory, id + ".glb"), os.path.join(directory, id + ".unoptimized.glb"))
            os.rename(os.path.join(directory, id + ".optimized.glb"), os.path.join(directory, id + ".glb"))


class gzip_task(task):
    est_time = 1
    
    def execute(self, directory, id):
        import gzip
        for ext in ["glb", "xml", "svg"]:
            fn = os.path.join(directory, id + "." + ext)
            if os.path.exists(fn):
                with open(fn, 'rb') as orig_file:
                    with gzip.open(fn + ".gz", 'wb') as zipped_file:
                        zipped_file.writelines(orig_file)


class svg_generation_task(task):
    est_time = 10

    def execute(self, directory, id):
        proc = subprocess.Popen([IFCCONVERT, id + ".ifc", id + ".svg", "-qy", "--section-height-from-storeys", "--door-arcs", "--print-space-names", "--print-space-areas", "--bounds=1024x1024", "--include", "entities", "IfcSpace", "IfcWall", "IfcWindow", "IfcDoor", "IfcAnnotation"], cwd=directory, stdout=subprocess.PIPE)
        i = 0
        while True:
            ch = proc.stdout.read(1)

            if not ch and proc.poll() is not None:
                break

            if ch and ord(ch) == ord('.'):
                i += 1
                self.sub_progress(i)



def do_process(id):
    d = utils.storage_dir_for_id(id)

    tasks = [
        ifc_validation_task,
        xml_generation_task,
        geometry_generation_task,
        svg_generation_task,
        glb_optimize_task,
        gzip_task
    ]
    
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
        tasks.append(mdl.task)
        
    tasks.sort(key=lambda t: getattr(t, 'order', 10))

    elapsed = 0
    set_progress(id, elapsed)
    
    total_est_time = sum(map(operator.attrgetter('est_time'), tasks))
    
    for t in tasks:
        begin_end = (elapsed / total_est_time * 99, (elapsed + t.est_time) / total_est_time * 99)
        task = t(begin_end)
        try:
            task(d, id)
        except:
            # Mark ID as failed
            with open(os.path.join(d, 'failed'), 'w') as f:
                pass
            break
        elapsed += t.est_time
        
    elapsed = 100
    set_progress(id, elapsed)
    
def process(id, callback_url):
    try:
        do_process(id)
        status = "success"
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        status = "failure"        
        
    if callback_url is not None:
        r = requests.post(callback_url, data={"status": status, "id": id})
