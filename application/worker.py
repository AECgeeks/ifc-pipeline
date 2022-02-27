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
import itertools
import shutil

from multiprocessing import Process

class ifcopenshell_file_dict(dict):
    def __missing__(self, key):
        import ifcopenshell
        self[key] = ifcopenshell.open(utils.storage_file_for_id(key, 'ifc'))
        return self[key]

import requests

import utils
import config
import database


on_windows = platform.system() == 'Windows'
ext = ".exe" if on_windows else ""
exe_path = os.path.join(os.path.dirname(__file__), "win" if on_windows else "nix")
IFCCONVERT = os.path.join(exe_path, "IfcConvert") + ext
if not os.path.exists(IFCCONVERT):
    IFCCONVERT = "IfcConvert"



def set_progress(id, progress):
    session = database.Session()
   
    id = id.split("_")[0]

    model = session.query(database.model).filter(database.model.code == id).all()[0]
    model.progress = int(progress)
    session.commit()
    session.close()


class task(object):
    def __init__(self, id, progress_map):
        import inspect
        print(self.__class__.__name__, inspect.getfile(type(self)), *progress_map)
        self.id = id
        self.begin, self.end = progress_map

    def sub_progress(self, i):
        set_progress(self.id, self.begin + (self.end - self.begin) * i / 100.)

    def __call__(self, *args):
        self.execute(*args)
        self.sub_progress(100)


class ifc_validation_task(task):
    est_time = 1

    def execute(self, context, id):
        import ifcopenshell.validate
        
        logger = ifcopenshell.validate.json_logger()
        f = context.models[id]
        
        ifcopenshell.validate.validate(f, logger)
                
        with open(os.path.join(context.directory, "log.json"), "w") as f:
            print("\n".join(json.dumps(x, default=str) for x in logger.statements), file=f)


class xml_generation_task(task):
    est_time = 1

    def execute(self, context, id):
        import ifcopenshell.geom
        f = context.models[id]
        sr = ifcopenshell.geom.serializers.xml(f, os.path.join(context.directory, f"{id}.xml"))
        sr.finalize()


class xml_to_json_conversion(task):
    est_time = 1
    
    def execute(self, context, id):
        subprocess.check_call([
            sys.executable,
            os.path.join(os.path.dirname(__file__), "process_xml_to_json.py"),
            id
        ], cwd=context.directory)


class geometry_generation_task(task):
    est_time = 10

    def execute(self, context, id):
        import ifcopenshell.geom
        
        ifcopenshell.ifcopenshell_wrapper.turn_off_detailed_logging()
        ifcopenshell.ifcopenshell_wrapper.set_log_format_json()
        
        f = context.models[id]
        settings = ifcopenshell.geom.settings(APPLY_DEFAULT_MATERIALS=True)
        sr = ifcopenshell.geom.serializers.gltf(utils.storage_file_for_id(id, "glb"), settings)
        sr.writeHeader()
        for progress, elem in ifcopenshell.geom.iterate(settings, f, with_progress=True, exclude=("IfcSpace", "IfcOpeningElement"), cache=utils.storage_file_for_id(id, "cache.h5")):
            sr.write(elem)
            self.sub_progress(progress)
        sr.finalize()
        
        with open(os.path.join(context.directory, f"log.json"), "w") as f:
            f.write(ifcopenshell.get_log())
                
class glb_optimize_task(task):
    est_time = 1
    
    def execute(self, context, id):
        try:
            if subprocess.call(["gltf-pipeline" + ('.cmd' if on_windows else ''), "-i", id + ".glb", "-o", id + ".optimized.glb", "-b", "-d"], cwd=context.directory) == 0:
                os.rename(os.path.join(context.directory, id + ".glb"), os.path.join(context.directory, id + ".unoptimized.glb"))
                os.rename(os.path.join(context.directory, id + ".optimized.glb"), os.path.join(context.directory, id + ".glb"))
        except FileNotFoundError:
            pass


class gzip_task(task):
    est_time = 1
    order = 2000
    
    def execute(self, context, id):
        import gzip
        for ext in ["glb", "xml", "svg"]:
            fn = os.path.join(context.directory, id + "." + ext)
            if os.path.exists(fn):
                with open(fn, 'rb') as orig_file:
                    with gzip.open(fn + ".gz", 'wb') as zipped_file:
                        zipped_file.writelines(orig_file)
                        
                        
class svg_generation_task(task):
    est_time = 10
    aggregate_model = True

    def execute(self, context):
        import ifcopenshell.geom

        settings = ifcopenshell.geom.settings(
            INCLUDE_CURVES=True,
            EXCLUDE_SOLIDS_AND_SURFACES=False,
            APPLY_DEFAULT_MATERIALS=True,
            DISABLE_TRIANGULATION=True
        )

        # cache = ifcopenshell.geom.serializers.hdf5("cache.h5", settings)
        
        sr = ifcopenshell.geom.serializers.svg(utils.storage_file_for_id(self.id, "svg"), settings)

        # @todo determine file to select here or unify building storeys accross files somehow
        sr.setFile(context.models[context.input_ids[0]])
        sr.setSectionHeightsFromStoreys()

        sr.setDrawDoorArcs(True)
        sr.setPrintSpaceAreas(True)
        sr.setPrintSpaceNames(True)
        sr.setBoundingRectangle(1024., 1024.)
        
        sr.setProfileThreshold(128)
        sr.setPolygonal(True)
        sr.setAlwaysProject(True)
        sr.setAutoElevation(True)

        # sr.setAutoSection(True)
        
        sr.writeHeader()
        for ii in context.input_ids:
            f = context.models[ii]
            for progress, elem in ifcopenshell.geom.iterate(settings, f, with_progress=True, exclude=("IfcOpeningElement",), cache=utils.storage_file_for_id(self.id, "cache.h5")):
                sr.write(elem)
                self.sub_progress(progress)

        sr.finalize()


class task_execution_context:
    
    def __init__(self, id):
        self.id = id
        self.directory = utils.storage_dir_for_id(id)
        self.input_files = [name for name in os.listdir(self.directory) if os.path.isfile(os.path.join(self.directory, name)) and name.endswith(".ifc")]
        self.models = ifcopenshell_file_dict()
    
        tasks = [
            ifc_validation_task,
            xml_generation_task,
            xml_to_json_conversion,
            geometry_generation_task,
            glb_optimize_task,
            gzip_task
        ]
        
        tasks_on_aggregate = [
            svg_generation_task
        ]
        
        self.is_multiple = any("_" in n for n in self.input_files)
        
        self.n_files = len(self.input_files)
        
        self.input_ids = ["%s_%d" % (self.id, i) if self.is_multiple else self.id for i in range(self.n_files)]
        
        """
        # Create a file called task_print.py with the following
        # example content to add application-specific tasks

        import sys
        
        from worker import task as base_task
        
        class task(base_task):
            est_time = 1    
            
            def execute(self, context):
                print("Executing task 'print' on ", context.id, ' in ', context.directory, file=sys.stderr)
        """
        
        for fn in glob.glob("task_*.py"):
            mdl = importlib.import_module(fn.split('.')[0])
            if getattr(mdl.task, 'aggregate_model', False):
                tasks_on_aggregate.append(mdl.task)
            else:
                tasks.append(mdl.task)
                
        self.tasks = list(filter(config.task_enabled, tasks))
        self.tasks_on_aggregate = list(filter(config.task_enabled, tasks_on_aggregate))

        self.tasks.sort(key=lambda t: getattr(t, 'order', 10))
        self.tasks_on_aggregate.sort(key=lambda t: getattr(t, 'order', 10))


    def run(self):
        elapsed = 0
        set_progress(self.id, elapsed)
        
        
        total_est_time = \
            sum(map(operator.attrgetter('est_time'), self.tasks)) * self.n_files + \
            sum(map(operator.attrgetter('est_time'), self.tasks_on_aggregate))
            
        def run_task(t, *args, aggregate_model=False):
            nonlocal elapsed
            begin_end = (elapsed / total_est_time * 99, (elapsed + t.est_time) / total_est_time * 99)
            task = t(self.id, begin_end)
            try:
                task(self, *args)
            except:
                traceback.print_exc(file=sys.stdout)
                # Mark ID as failed
                with open(os.path.join(self.directory, 'failed'), 'w') as f:
                    pass
                return False
            elapsed += t.est_time
            return True

        with_failure = False
                
        for t, ii in itertools.product(self.tasks, self.input_ids):
            if not run_task(t, ii):
                with_failure = True
                break
                
        if not with_failure:
            for t in self.tasks_on_aggregate:
                run_task(t, aggregate_model=True)

        elapsed = 100
        set_progress(self.id, elapsed)        


def do_process(id):
    tec = task_execution_context(id)
    # for local development
    # tec.run()

    p = Process(target=task_execution_context.run, args=(tec,))
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError()


def process(id, callback_url):
    try:
        do_process(id)
        status = "success"
    except Exception as e:
        traceback.print_exc(file=sys.stdout)
        status = "failure"        

    if callback_url is not None:       
        r = requests.post(callback_url, data={"status": status, "id": id})

