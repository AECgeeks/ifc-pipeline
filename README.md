ifc-pipeline
------------

A processing queue that uses [IfcOpenShell](https://github.com/IfcOpenShell/IfcOpenShell/) to convert IFC input files into a graphic display using glTF 2.0 and [BIMSurfer2](https://github.com/AECgeeks/BIMsurfer2/) for visualization.

There is a small web application in Flaks that accepts file uploads. HTTPS is provided by Nginx. Everything is tied together using Docker Compose.

~~~
./init.sh my.domain.name.com
docker-compose up -d
~~~
