FROM ubuntu:latest

WORKDIR /

RUN apt-get -y update && apt-get -y --no-install-recommends --no-install-suggests install \
	 unzip wget libpq-dev build-essential libssl-dev libffi-dev libxml2-dev libxslt1-dev \
	 zlib1g-dev supervisor blender build-essential cmake git ca-certificates

RUN wget -qO /tmp/mc.sh https://github.com/conda-forge/miniforge/releases/download/4.13.0-1/Mambaforge-Linux-x86_64.sh && chmod +x /tmp/mc.sh && \
	bash /tmp/mc.sh -b -p /conda && \
	/conda/bin/mamba install -y python=3.9 pythonocc-core numpy numba matplotlib scipy scikit-image pandas psutil && \
	rm /tmp/mc.sh

RUN wget -qO- https://nodejs.org/dist/v16.18.0/node-v16.18.0-linux-x64.tar.gz | tar -xzC /opt

ENV PATH /opt/node-v16.18.0-linux-x64/bin:/conda/bin:$PATH

RUN python3 -m pip install "Flask<2.0.0" flask-cors Pillow gunicorn rq redis sqlalchemy psycopg2 sqlalchemy-utils Flask-BasicAuth flasgger requests flask-dropzone flask-healthz sknw rdp bresenham scikit-learn minio aabbtree MarkupSafe==2.0.1
RUN npm install -g jsdoc gltf-pipeline requirejs uglify-js

RUN git clone --recursive https://github.com/KhronosGroup/COLLADA2GLTF && \
	cd /COLLADA2GLTF/GLTF/dependencies/draco/ && \
	wget -qO- https://github.com/google/draco/commit/d9006bfc730761616c4bbcaaa5f78cf2d33f8190.patch | patch --fuzz 10 -p1 && \
	wget -qO- https://github.com/google/draco/commit/d09e0008680ba12af75f47a9a176b11af0f5d96f.patch | patch --fuzz 10 -p1 && \
	mkdir /COLLADA2GLTF/build && cd /COLLADA2GLTF/build && \
	cmake .. && make install -j `nproc` && \
	mv COLLADA2GLTF-bin /usr/local/bin && \
	cd / && \
	rm -rf COLLADA2GLTF

ARG IFOPSH_VERSION=v0.7.0-64d98f1

# IfcConvert v0.7.0
RUN wget https://s3.amazonaws.com/ifcopenshell-builds/IfcConvert-${IFOPSH_VERSION}-linux64.zip -O /tmp/IfcConvert.zip && \
	unzip /tmp/IfcConvert.zip -d /usr/bin && \
	rm /tmp/IfcConvert.zip

# IfcOpenShell-Python v0.7.0
RUN wget -O /tmp/ifcopenshell_python.zip https://s3.amazonaws.com/ifcopenshell-builds/ifcopenshell-python-`python3 -c 'import sys;print("".join(map(str, sys.version_info[0:2])))'`-${IFOPSH_VERSION}-linux64.zip  && \
	mkdir -p `python3 -c 'import site; print(site.getusersitepackages())'` && \
	unzip -d `python3 -c 'import site; print(site.getusersitepackages())'` /tmp/ifcopenshell_python.zip && \
	wget https://raw.githubusercontent.com/IfcOpenShell/IfcOpenShell/v0.7.0/src/ifcopenshell-python/ifcopenshell/geom/occ_utils.py -O `python3 -c 'import site; print(site.getusersitepackages())'`/ifcopenshell/geom/occ_utils.py && \
	rm /tmp/ifcopenshell_python.zip

# Voxec IfcOpenShell v0.7.0
RUN wget https://s3.amazonaws.com/ifcopenshell-builds/voxec-${IFOPSH_VERSION}-linux64.zip -O /tmp/voxec.zip && \
	unzip /tmp/voxec.zip -d /usr/bin && \
	rm /tmp/voxec.zip

# Get git commit sha in /version
COPY .git/HEAD /tmp/.git/HEAD
COPY .git/refs/ /tmp/.git/refs/
RUN /bin/bash -c '(cat /tmp/.git/$(cat /tmp/.git/HEAD | cut -d \  -f 2)) || cat /tmp/.git/HEAD' > /version
RUN rm -rf /tmp/.git

COPY application/static /www/static/
COPY application/bimsurfer/bimsurfer /www/static/bimsurfer

WORKDIR /www/static
RUN npm i -D @babel/preset-env @babel/core @babel/cli && \
	r.js -o baseUrl=. name=App out=App.built.js optimize=none && \
	npx babel App.built.js -o App.built.es2015.js --presets=@babel/preset-env --compact false && \
	uglifyjs --compress -o App.$(cat /version).js -- App.built.es2015.js && \
	npx babel bimsurfer/lib/DRACOLoader.js -o bimsurfer/lib/DRACOLoader.es2015.js --presets=@babel/preset-env --compact false && \
	uglifyjs --compress -o bimsurfer/lib/three.gltf.orbit.draco.min.js -- bimsurfer/lib/three.js bimsurfer/lib/GLTFLoader.js bimsurfer/lib/DRACOLoader.es2015.js bimsurfer/lib/OrbitControls.js && \
	rm -rf node_modules package-lock.json

COPY application/queue.conf /etc/supervisord.conf
COPY application/clean_tmp.sh /clean_tmp.sh
RUN chmod +x /clean_tmp.sh

# Server
WORKDIR /www
COPY application/*.py /www/
COPY application/templates /www/templates
RUN sed -i "4i<script>console.log('pipeline version: $(cat /version)');</script>" /www/templates/*.html

WORKDIR /www
