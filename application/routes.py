import os
import io
import gzip
import shutil

from flask import abort, jsonify, request, send_file

import utils
import database

from main import application, process_upload, queue_task, DEVELOPMENT

@application.route("/file", methods=['POST'])
def accept_file_gzip():
    application.logger.debug("Request Headers %s", request.headers)

    if request.headers['Content-Encoding'] != 'gzip':
        abort(400)
    
    file = io.BytesIO(request.data)
    gz = gzip.GzipFile(fileobj=file, mode='r')
    
    def process(fn):
        with open(fn, 'wb') as f:
            try:
                shutil.copyfileobj(gz, f)
            except OSError:
                abort(400)
    
    id = process_upload(process)
    return jsonify({
        "status": "ok",
        "id": id
    })

@application.route("/file/<id>/status", methods=['GET'])
def get_file_progress(id):
    if not utils.validate_id(id):
        abort(404)
    session = database.Session()
    models = session.query(database.model).filter(database.model.code == id).all()
    session.close()
    if len(models) != 1:
        abort(404)
    p = models[0].progress
    if p in (-1, -2):
        return jsonify({
            "status": ["queued", "errored"][p == -2],
            "id": id
        })
    elif p == 100:
        return jsonify({
            "status": "done",
            "id": id
        })
    else:
        return jsonify({
            "status": "progress",
            "progress": p,
            "id": id
        })
   
@application.route("/run/<check>", methods=['POST'])
def initiate_check_escape_routes(check):
    if check not in {'escape_routes', 'calculate_volume', 'space_heights', 'stair_headroom', 'door_direction', 'landings', 'safety_barriers', 'entrance_area', 'ramp_percentage'}:
        abort(404)
        
    id = utils.generate_id()
    # d = utils.storage_dir_for_id(id, output=True)
    # os.makedirs(d)
    
    config = request.json
    
    session = database.Session()
    session.add(database.model(id, ''))
    session.commit()
    session.close()
    
    queue_task(check, id, config)
    return jsonify({
        "status": "ok",
        "id": id
    })


@application.route("/run/<id>/status", methods=['GET'])
def get_check_progress(id):
    if not utils.validate_id(id):
        abort(404)
        
    session = database.Session()
    models = session.query(database.model).filter(database.model.code == id).all()
    session.close()
    
    if len(models) != 1:
        abort(404)
        
    p = models[0].progress
    
    fn = utils.storage_file_for_id(id + "_0", "glb", output=True)
    fn2 = utils.storage_file_for_id(id, "json", output=True)
    
    if p == 100 and (os.path.exists(fn) or os.path.exists(fn2)):
        return jsonify({
            "status": "done",
            "id": id
        })
    elif p in (-1, -2):
        return jsonify({
            "status": ["queued", "errored"][p == -2],
            "id": id
        })
    else:
        return jsonify({
            "status": "progress",
            "progress": p,
            "id": id
        })
        
@application.route("/run/<id>/log", methods=['GET'])
def get_check_log(id):
    if not utils.validate_id(id):
        abort(404)
        
    fn = utils.storage_file_for_id(id + "_log", "json", output=True)
    return send_file(fn)

   
@application.route("/run/<id>/result", methods=['GET'])
def get_check_results(id):
    if not utils.validate_id(id):
        abort(404)

    fn = utils.storage_file_for_id(id + "_0", "glb", output=True)
    fn2 = utils.storage_file_for_id(id, "json", output=True)

    if not os.path.exists(fn) and not os.path.exists(fn2):
        abort(409)
        
    if os.path.exists(fn2):
        return send_file(fn2)
    else:    
        return jsonify({
            "id": id,
            "results": [{"visualization": "/run/%s/result/resource/gltf/0.glb" % id}]
        })

@application.route("/run/<id>/result/resource/gltf/<i>.glb", methods=['GET'])
def get_gltf(id, i):
    try:
        i = int(i)
    except:
        import traceback
        traceback.print_exc()
        abort(404)
    
    fn = utils.storage_file_for_id(id + "_%d" % i, "glb", output=True)
    print(fn)
    if not os.path.exists(fn):
        abort(404)
    return send_file(fn)
