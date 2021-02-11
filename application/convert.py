# blender -b -P dae_convert.py -- <obj_file.obj>

import sys
import bpy

assert sys.argv[4] == '--'

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
for fn in sys.argv[5:-1]:
    bpy.ops.import_scene.obj(filepath=fn, axis_forward='Y', axis_up='Z')
bpy.ops.wm.collada_export(filepath=sys.argv[-1])
