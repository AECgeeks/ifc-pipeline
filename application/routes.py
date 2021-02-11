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
            print(gz.tell(), f.tell())
            print("copying data")
            shutil.copyfileobj(gz, f)
            print(gz.tell(), f.tell())
    
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
    if p == -1:
        return jsonify({
            "status": "queued",
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
   
@application.route("/run/escape_routes", methods=['POST'])
def initiate_check():
    id = utils.generate_id()
    d = utils.storage_dir_for_id(id)
    os.makedirs(d)
    files = [os.path.join(utils.storage_dir_for_id(i), i+".ifc") for i in request.json['ids']]
    
    session = database.Session()
    session.add(database.model(id, ''))
    session.commit()
    session.close()
    
    queue_task('escape_routes', id, files)
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
        
    d = utils.storage_dir_for_id(id)
    fn = os.path.join(d, "0.glb")
    if os.path.exists(fn):
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
        
@application.route("/run/<id>/log", methods=['GET'])
def get_check_log(id):
    if not utils.validate_id(id):
        abort(404)
        
    d = utils.storage_dir_for_id(id)
    fn = os.path.join(d, "log.json")
    return send_file(fn)

   
@application.route("/run/<id>/result", methods=['GET'])
def get_check_results(id):
    if not utils.validate_id(id):
        abort(404)
    d = utils.storage_dir_for_id(id)
    fn = os.path.join(d, "0.glb")
    if not os.path.exists(fn):
        abort(409)
    return jsonify({
        "id": id,
        "results": [{"visualization": "/run/%s/result/resource/gltf/0.glb" % id}]
    })

@application.route("/run/<id>/result/resource/gltf/<i>.glb", methods=['GET'])
def get_gltf(id, i):
    if i != "0":
        abort(404)
    d = utils.storage_dir_for_id(id)
    fn = os.path.join(d, "0.glb")
    if not os.path.exists(fn):
        abort(404)
    return send_file(fn)
