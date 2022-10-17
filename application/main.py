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

from __future__ import print_function

import os
import glob
import json
import operator
import threading

from collections import defaultdict, namedtuple
from flask_dropzone import Dropzone

from werkzeug.middleware.proxy_fix import ProxyFix
from flask import Flask, request, send_file, render_template, abort, jsonify, redirect, url_for, make_response, send_from_directory
from flask_cors import CORS
from flask_basicauth import BasicAuth
from flasgger import Swagger

import utils
import worker
import config
import database

# We have a custom static file handler that serves two directories,
# but None cannot be supplied here because flask-dropzone depends on
# it.
application = Flask(__name__, static_folder="non-existant")
dropzone = Dropzone(application)

# application.config['DROPZONE_UPLOAD_MULTIPLE'] = True
# application.config['DROPZONE_PARALLEL_UPLOADS'] = 3

DEVELOPMENT = os.environ.get('environment', 'production').lower() == 'development'
WITH_REDIS = os.environ.get('with_redis', 'false').lower() == 'true'


if not DEVELOPMENT and os.path.exists("/version"):
    PIPELINE_POSTFIX = "." + open("/version").read().strip()
else:
    PIPELINE_POSTFIX = ""


if not DEVELOPMENT:
    # In some setups this proved to be necessary for url_for() to pick up HTTPS
    application.wsgi_app = ProxyFix(application.wsgi_app, x_proto=1)

CORS(application)
application.config['SWAGGER'] = {
    'title': os.environ.get('APP_NAME', 'ifc-pipeline request API'),
    'openapi': '3.0.2',
    "specs": [
        {
            "version": "0.1",
            "title": os.environ.get('APP_NAME', 'ifc-pipeline request API'),
            "description": os.environ.get('APP_NAME', 'ifc-pipeline request API'),
            "endpoint": "spec",
            "route": "/apispec",
        },
    ]
}
swagger = Swagger(application)

if DEVELOPMENT and not WITH_REDIS:
    redis_queue = None
else:
    from redis import Redis
    from rq import Queue
    
    redis = Redis(host=os.environ.get("REDIS_HOST", "localhost"))
    redis_queue = Queue(connection=redis, default_timeout=3600)


@application.route('/', methods=['GET'])
def get_main():
    return render_template('index.html')



def process_upload(filewriter, callback_url=None):
    id = utils.generate_id()
    d = utils.storage_dir_for_id(id)
    os.makedirs(d)
    
    filewriter(os.path.join(d, id+".ifc"))
    
    session = database.Session()
    session.add(database.model(id, ''))
    session.commit()
    session.close()
    
    if redis_queue is None:
        t = threading.Thread(target=lambda: worker.process(id, callback_url))
        t.start()
    else:
        redis_queue.enqueue(worker.process, id, callback_url)

    return id
    


def process_upload_multiple(files, callback_url=None):
    id = utils.generate_id()
    d = utils.storage_dir_for_id(id)
    os.makedirs(d)
   
    file_id = 0
    session = database.Session()
    m = database.model(id, '')   
    session.add(m)
  
    for file in files:
        fn = file.filename
        filewriter = lambda fn: file.save(fn)
        filewriter(os.path.join(d, id+"_"+str(file_id)+".ifc"))
        file_id += 1
        m.files.append(database.file(id, ''))
    
    session.commit()
    session.close()
    
    if redis_queue is None:
        t = threading.Thread(target=lambda: worker.process(id, callback_url))
        t.start()        
    else:
        redis_queue.enqueue(worker.process, id, callback_url)

    return id



@application.route('/', methods=['POST'])
def put_main():
    """
    Upload model
    ---
    requestBody:
      content:
        multipart/form-data:
          schema:
            type: object
            properties:
              ifc:
                type: string
                format: binary
    responses:
      '200':
        description: redirect
    """
    ids = []
   
    files = []
    for key, f in request.files.items():
        if key.startswith('file'):
            file = f
            files.append(file)    

       
    id = process_upload_multiple(files)
    url = url_for('check_viewer', id=id) 

    if request.accept_mimetypes.accept_json:
        return jsonify({"url":url})
    else:
        return redirect(url)


@application.route('/p/<id>', methods=['GET'])
def check_viewer(id):
    if not utils.validate_id(id):
        abort(404)
    return render_template('progress.html', id=id)    
    
    
@application.route('/pp/<id>', methods=['GET'])
def get_progress(id):
    if not utils.validate_id(id):
        abort(404)
    session = database.Session()
    model = session.query(database.model).filter(database.model.code == id).all()[0]
    session.close()
    return jsonify({"progress": model.progress})


@application.route('/log/<id>.<ext>', methods=['GET'])
def get_log(id, ext):
    log_entry_type = namedtuple('log_entry_type', ("level", "message", "instance", "product"))
    
    if ext not in {'html', 'json'}:
        abort(404)
        
    if not utils.validate_id(id):
        abort(404)
    logfn = os.path.join(utils.storage_dir_for_id(id), "log.json")
    if not os.path.exists(logfn):
        abort(404)
            
    if ext == 'html':
        log = []
        for ln in open(logfn):
            l = ln.strip()
            if l:
                log.append(json.loads(l, object_hook=lambda d: log_entry_type(*(d.get(k, '') for k in log_entry_type._fields))))
        return render_template('log.html', id=id, log=log)
    else:
        return send_file(logfn, mimetype='text/plain')


@application.route('/v/<id>', methods=['GET'])
@application.route('/live/<id>/<channel>', methods=['GET'])
def get_viewer(id, channel=None):
    if not utils.validate_id(id):
        abort(404)
    d = utils.storage_dir_for_id(id)
    
    if not os.path.exists(d):
        abort(404)
    
    ifc_files = [os.path.join(d, name) for name in os.listdir(d) if os.path.isfile(os.path.join(d, name)) and name.endswith('.ifc')]
    
    if len(ifc_files) == 0:
        abort(404)
    
    failedfn = os.path.join(utils.storage_dir_for_id(id), "failed")
    if os.path.exists(failedfn):
        return render_template('error.html', id=id)

    for ifc_fn in ifc_files:
        glbfn = ifc_fn.replace(".ifc", ".glb")
        if not os.path.exists(glbfn):
            abort(404)
            
    n_files = len(ifc_files) if "_" in ifc_files[0] else None
                    
    return render_template(
        'viewer.html',
        id=id,
        n_files=n_files,
        postfix=PIPELINE_POSTFIX,
        with_screen_share=config.with_screen_share,
        live_share_id=channel or utils.generate_id(),
        mode='listen' if channel else 'view'
    )


@application.route('/m/<fn>', methods=['GET'])
def get_model(fn):
    """
    Get model component
    ---
    parameters:
        - in: path
          name: fn
          required: true
          schema:
              type: string
          description: Model id and part extension
          example: BSESzzACOXGTedPLzNiNklHZjdJAxTGT.glb
    """
    
 
    id, ext = fn.split('.', 1)
    
    if not utils.validate_id(id):
        abort(404)
  
    if ext not in {"xml", "svg", "glb", "unoptimized.glb", "tree.json"}:
        abort(404)
   
    path = utils.storage_file_for_id(id, ext)    

    if not os.path.exists(path):
        abort(404)
        
    if os.path.exists(path + ".gz"):
        import mimetypes
        response = make_response(
            send_file(path + ".gz", 
                mimetype=mimetypes.guess_type(fn, strict=False)[0])
        )
        response.headers['Content-Encoding'] = 'gzip'
        return response
    else:
        return send_file(path)

        
@application.route('/live/<channel>', methods=['POST'])
def post_live_viewer_update(channel):
    # body = request.get_json(force=True)
    # @todo validate schema?
    # body = json.dumps(body)
    body = request.data.decode('ascii');
    redis.publish(channel=f"live_{channel}", message=body)
    return ""


@application.route('/static/<path:filename>')
def static_handler(filename):
    # filenames = [os.path.join(root, fn)[len("static")+1:] for root, dirs, files in os.walk("static", topdown=False) for fn in files]
    if filename.startswith("bimsurfer/"):
        return send_from_directory("bimsurfer", "/".join(filename.split("/")[1:]))
    else:
        return send_from_directory("static", filename)


@application.route('/live/<channel>', methods=['GET'])
def get_viewer_update(channel):
    def format(obj):
        return f"data: {obj.decode('ascii')}\n\n"

    def stream():
        pubsub = redis.pubsub()
        pubsub.subscribe(f"live_{channel}")
        try:
            msgs = pubsub.listen()
            yield from map(format, \
                map(operator.itemgetter('data'), \
                filter(lambda x: x.get('type') == 'message', msgs)))
        finally:
            import traceback
            traceback.print_exc()
            try: pubsub.unsubscribe(channel)
            except: pass
    
    return application.response_class(
        stream(),
        mimetype='text/event-stream',
        headers={'X-Accel-Buffering': 'no', 'Cache-Control': 'no-cache'},
    )

"""
# Create a file called routes.py with the following
# example content to add application-specific routes

from main import application

@application.route('/test', methods=['GET'])
def test_hello_world():
    return 'Hello world'
"""
try:
    import routes
except ImportError as e:
    pass
