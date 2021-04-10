import io
import sys
import argparse
import subprocess

from jinja2 import Environment, FileSystemLoader

parser = argparse.ArgumentParser()
parser.add_argument('--with-https', dest='with_https', action='store_const', const=True, default=True)
parser.add_argument('--without-https', dest='with_https', action='store_const', const=False, default=True)
parser.add_argument('--with-minio', dest='with_minio', action='store_const', const=True, default=False)
parser.add_argument('--without-minio', dest='with_minio', action='store_const', const=False, default=False)
parser.add_argument('--convert', dest='convert', action='store_const', const=True, default=False)
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

if args.convert:

    import os    
    import copy
    import glob
    
    from collections import defaultdict

    def di_to_fs(x):
        if isinstance(x, dict):
            return frozenset((k, di_to_fs(v)) for k, v in x.items())
        if isinstance(x, list):
            return tuple(di_to_fs(v) for v in x)
        return x
        
    def fs_to_di(x):
        if isinstance(x, frozenset):
            return {k: fs_to_di(v) for k, v in x}
        if isinstance(x, tuple):
            return list(fs_to_di(v) for v in x)
        return x
        
    volume_handling = {
        './docker-volumes/certbot/conf' : 'emptyDir',
        './docker-volumes/certbot/www'  : 'emptyDir',
        './docker-volumes/redis'        : 'emptyDir',
        './docker-volumes/nginx/log'    : 'emptyDir',
        './docker-volumes/db'           : 'persistentVolumeClaim',
        './docker-volumes/minio'        : 'persistentVolumeClaim',
        './models'                      : 'persistentVolumeClaim',
        './output'                      : 'persistentVolumeClaim',
        './nginx'                       : 'configMap',
        './nginx_http_only'             : 'configMap'
    }
    
    s = io.StringIO()
    print(template.render(**vars(args)), file=s)
    s.seek(0)

    from yaml import load, dump, FullLoader
    cfg = load(s, Loader=FullLoader)
    
    def remove(v, to_remove=()):
        if isinstance(v, dict):
            return {k: remove(v, to_remove) for k, v in v.items() if k not in to_remove}
        else:
            return v
            
    # remove the build: and command: keys from the config
    cfg_ = remove(cfg, ("build", "command"))
    
    for H in set(volume_handling.values()):
    
        cfg = copy.deepcopy(cfg_)
    
        volume_lists = [[d.get('volumes') for d in v.values()] for v in cfg.values() if isinstance(v, dict)][0]
        
        global_volume_names = []
        
        for li in volume_lists:
            if li:
                for i, v in reversed(list(enumerate(list(li)))):
                    host_path, cont_path, *rest = v.split(":")
                    handler = volume_handling[host_path]
                    if handler != H:
                        li[i:i+1] = []
                    elif handler != "configMap":
                        # we need to remap to a volume not on the host
                        name = host_path.replace("./", "").replace("/", "-")
                        global_volume_names.append(name)
                        li[i] = ":".join((name, cont_path) + tuple(rest))
                        
        volume_lists = [[d.get('volumes') for d in v.values()] for v in cfg.values() if isinstance(v, dict)][0]

        cfg['volumes'] = {k:None for k in global_volume_names}
                    
        with open('docker-compose-%s.yml' % H, 'w') as f:
            dump(cfg, f, default_flow_style=False)

        subprocess.call(['kompose', '-v', 'convert', '-c', "--file", "docker-compose-%s.yml" % H, "--volumes", H, "-o", H] + compose_args)
        
    merge = defaultdict(list)
        
    for H in set(volume_handling.values()):
        files = glob.glob(H+"/**", recursive=True)
        names = [s[len(H)+1:] for s in files]
        for f, n in zip(files, names):
            if n.endswith('yaml'):
                merge[n].append(load(open(f), Loader=FullLoader))
               
    try: 
        os.makedirs('merged/templates')
    except: pass
    
    preset = {
        'Chart.yaml.name': 'ifc-pipeline-vx',
        'Chart.yaml.description': 'IFC-pipeline with voxelization',
    }
    
    def do_merge(args, key=None):
        key = key or []
        key_s = ".".join(key)
        preset_v = preset.get(key_s)

        if len(args) == 1:
            return args[0]
            
        if key_s.endswith("spec.containers"):
            # instead of merging the lists, we merge the first item of the lists, which is a dict
            # and wrap it in a list
            return [do_merge([a[0] for a in args], key=key + ["0"])]
            
        if key_s.endswith("spec.containers.0.command"):
            # is this a Kompose convert bug?
            # or an issue in compose-template.yml?
            # import shlex
            # return [shlex.join(args[0])]
            pass
        
        if all(isinstance(a, dict) or a is None for a in args):
            if key_s.endswith('metadata.annotations'):
                return None
            dicts = list(filter(None, args))
            if len(dicts) == 0:
                return None
            keys = set(sum((list(a.keys()) for a in dicts), []))
            return {k: do_merge([a.get(k) for a in dicts], key=key + [k]) for k in keys}
        elif preset_v:
            return preset_v
        elif all(isinstance(a, str) or a is None for a in args) and len(set(filter(None, args))) == 1:
            # note that None values are skipped here, so ['Recreate' None 'Recreate'] -> 'Recreate'
            return args[0]
        elif all(isinstance(a, list) or a is None for a in args):
            li = sum([a or [] for a in args], [])
            
            # remove duplicate items (e.g port bindings)
            li = list(map(di_to_fs, li))
            li = list({v:1 for v in li}.keys())
            li = list(map(fs_to_di, li))
            
            return li
        elif len(set(args)) == 1:
            # can be merged with rule above:
            return args[0]
        else:
            import pdb; pdb.set_trace()

    for k, cfgs in merge.items():
        with open(os.path.join('merged', k), 'w') as f:
            cfg = do_merge(cfgs, [k])
            
            # We need to manually patch the access mode. Kompose's approach is to generate accessMode based on
            # the volume first encountered and then drop further usage of it. This fails for the output/
            # volume where the first usage is RO which is then translated to ReadOnlyMany
            # https://github.com/kubernetes/kompose/blob/76565d80b2dccfa2de4b4612557788b1861cc48a/pkg/transformer/kubernetes/kubernetes.go#L533
            if "persistentvolumeclaim" in k:
                cfg['spec']['accessModes'] = ['ReadWriteOnce']
            
            dump(cfg, f, default_flow_style=False)
    
else:

    with open('docker-compose.yml', 'w') as f:
        print(template.render(**vars(args)), file=f)

    subprocess.call(['docker-compose', '-f', 'docker-compose.yml'] + compose_args)
