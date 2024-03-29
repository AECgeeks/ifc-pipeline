import os
import json

from multiprocessing import cpu_count

import jsonschema

# parse configuration file
_config = json.load(open(os.path.join(os.path.dirname(__file__), "config.json")))

# parse configuration schema and validate _config
schema = json.load(open(os.path.join(os.path.dirname(__file__), "config.schema")))
jsonschema.validate(schema=schema, instance=_config)

# returns true if task is enabled in _config
task_enabled = lambda nm: nm.__name__ in _config['tasks']
treeview_label = _config['treeview']['label']
with_screen_share = _config['features'].get("screen_share", {}).get("enabled", False)
num_threads = _config['performance']['num_threads']
if num_threads == 'auto':
    num_threads = max(1, cpu_count() // _config['performance']['num_workers'])
