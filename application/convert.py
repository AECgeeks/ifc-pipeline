# blender -b -P dae_convert.py [--split] -- <obj_file.obj>

import sys
import bpy

assert '--' in sys.argv
sep = sys.argv.index('--')
split = '--split' in sys.argv
components = '--components' in sys.argv
orient = "--orient" in sys.argv

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

for fn in sys.argv[sep + 1:-1]:
    if fn.endswith(".obj"):
        bpy.ops.import_scene.obj(filepath=fn, axis_forward='Y', axis_up='Z', use_split_groups=True)
    elif fn.endswith(".dae"):
        bpy.ops.wm.collada_import(filepath=fn)
    else:
        print(fn)
        exit()
        
if orient:
    for obj in list(bpy.data.objects):
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.remove_doubles()
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.editmode_toggle()

if split:
    for ob in bpy.data.objects:
        print(ob.name)
    get_id = lambda nm: nm.split('.')[0].split('-')[-1]
    for id in {get_id(ob.name) for ob in bpy.data.objects}:
        if components:
            pass
        else:
            try:
                int(id)
            except:
                continue

        for ob in bpy.data.objects:
            if hasattr(ob, 'select'):
                ob.select = get_id(ob.name) == id
            else:
                ob.select_set(get_id(ob.name) == id)
        bpy.ops.wm.collada_export(filepath=sys.argv[-1] % id, selected=True)
else:
    bpy.ops.wm.collada_export(filepath=sys.argv[-1])

exit()
