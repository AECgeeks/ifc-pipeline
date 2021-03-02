import sys
import argparse
import subprocess

from jinja2 import Environment, FileSystemLoader

parser = argparse.ArgumentParser()
parser.add_argument('--with-https', dest='with_https', action='store_const', const=True, default=True)
parser.add_argument('--without-https', dest='with_https', action='store_const', const=False, default=True)
parser.add_argument('--db-host', dest='db_host')
parser.add_argument('--db-user', dest='db_user')
parser.add_argument('--db-pass', dest='db_pass')
(args, compose_args) = parser.parse_known_args()

env = Environment(
    loader=FileSystemLoader('.'),
    trim_blocks=True,
    lstrip_blocks=True
)
template = env.get_template('docker-compose-template.yml')
with open('docker-compose.yml', 'w') as f:
    print(template.render(**vars(args)), file=f)

subprocess.call(['docker-compose', '-f', 'docker-compose.yml'] + compose_args)
