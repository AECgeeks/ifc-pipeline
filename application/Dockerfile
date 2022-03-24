FROM ubuntu:latest

WORKDIR /
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update && apt-get -y --no-install-recommends --no-install-suggests install python3 python3-pip unzip wget libpq-dev build-essential libssl-dev libffi-dev libxml2-dev libxslt1-dev zlib1g-dev npm python3-setuptools python3-dev python3-wheel supervisor libjpeg-dev
RUN npm install -g jsdoc gltf-pipeline requirejs npx uglify-js

# IfcConvert v0.7.0
RUN wget https://s3.amazonaws.com/ifcopenshell-builds/IfcConvert-v0.7.0-b5133c6-linux64.zip -O /tmp/IfcConvert.zip
RUN unzip /tmp/IfcConvert.zip -d /usr/bin

RUN wget -O /tmp/ifcopenshell_python.zip https://s3.amazonaws.com/ifcopenshell-builds/ifcopenshell-python-`python3 -c 'import sys;print("".join(map(str, sys.version_info[0:2])))'`-v0.7.0-b5133c6-linux64.zip
RUN mkdir -p `python3 -c 'import site; print(site.getusersitepackages())'`
RUN unzip -d `python3 -c 'import site; print(site.getusersitepackages())'` /tmp/ifcopenshell_python.zip

# Server
WORKDIR /www
COPY application/*.py application/*.txt application/config.json application/config.schema /www/
COPY application/templates /www/templates

# Python dependencies
RUN python3 -m pip install -r requirements-production.txt

COPY .git/HEAD /tmp/.git/HEAD
COPY .git/refs/ /tmp/.git/refs/
RUN /bin/bash -c '(cat /tmp/.git/$(cat /tmp/.git/HEAD | cut -d \  -f 2)) || cat /tmp/.git/HEAD' > /version
RUN sed -i "4i<script>console.log('pipeline version: $(cat /version)');</script>" /www/templates/*.html
RUN rm -rf /tmp/.git

RUN npm i -D @babel/preset-env @babel/core @babel/cli

COPY application/static /www/static/
COPY application/bimsurfer/bimsurfer /www/static/bimsurfer

WORKDIR /www/static
# RUN jsdoc -r -d docs bimsurfer
RUN r.js -o baseUrl=. name=App out=App.built.js optimize=none
RUN npx babel App.built.js -o App.built.es2015.js --presets=@babel/preset-env --compact false
RUN uglifyjs --compress -o App.$(cat /version).js -- App.built.es2015.js 
RUN npx babel bimsurfer/lib/DRACOLoader.js -o bimsurfer/lib/DRACOLoader.es2015.js --presets=@babel/preset-env --compact false
RUN uglifyjs --compress -o bimsurfer/lib/three.gltf.orbit.draco.min.js -- bimsurfer/lib/three.js bimsurfer/lib/GLTFLoader.js bimsurfer/lib/DRACOLoader.es2015.js bimsurfer/lib/OrbitControls.js 
RUN rm -rf node_modules package-lock.json

COPY application/queue.conf /etc/supervisord.conf
RUN sed -i s/NUM_WORKERS/`python3 -c "import json; print(json.load(open('/www/config.json'))['performance']['num_workers'])"`/g /etc/supervisord.conf

WORKDIR /www
