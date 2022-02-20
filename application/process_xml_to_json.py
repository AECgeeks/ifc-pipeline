import os
import re
import sys
import json

from functools import lru_cache, partial

import ifcopenshell
import ifcopenshell.util.element
import lxml.etree as ET

from jinja2 import Environment, BaseLoader, Undefined

import config

class Ignore(Undefined):
    def _fail_with_undefined_error(self, *args, **kwargs):
        return None


T = Environment(
    loader=BaseLoader,
    undefined=Ignore
).from_string(config.treeview_label)


def shorten_namespace(k):
    ns = re.findall(r"\{(.+?)\}(\w+)", k)
    if ns:
        return f"{ns[0][0].split('/')[-1]}:{ns[0][1]}"
    else:
        return k


def attempt(fn):
    try:
        return fn()
    except:
        pass


def map_attribute(k):
    return {'Name': 'name', 'id': 'guid'}.get(k, k)

def to_dict(ifc, t):
    di = {map_attribute(shorten_namespace(k)): v for k, v in (t.attrib or {}).items()}
    is_product = attempt(lambda: ifc[di.get('guid')].is_a("IfcProduct"))
    is_project = attempt(lambda: ifc[di.get('guid')].is_a("IfcProject"))
    if is_product or is_project:
        di['label'] = T.render(**instance_template_lookup(ifc[di.get('guid')]))
    di['type'] = t.tag
    cs = list(map(partial(to_dict, f), t))
    if cs:
        di['children'] = cs
    t = (t.text or '').strip()
    if t:
        di['text'] = t
    return di


def dict_to_namespace(di):
    methods = {
        '__repr__': lambda self: ''
    }
    methods.update(di)
    return type('anonymous_', (object,), methods)()


class instance_template_lookup:
    def __init__(self, inst):
        self.inst = inst
        di = {k: v for k, v in inst.get_info(include_identifier=False, recursive=False).items() if isinstance(v, (str, int, float)) and k != 'type'}
        self.attrs = dict_to_namespace(di)
        
    def __getitem__(self, k):
        if k == 'attr':
            return self.attrs
        elif k == 'prop':
            return instance_property_lookup(self.inst)
        elif k == 'type':
            return self.inst.is_a()
        else:
            raise KeyError(k)
            
    def keys(self):
        return 'attr', 'prop', 'type'


class instance_property_lookup:
    def __init__(self, inst):
        self.inst = inst
        
    def __getattr__(self, k):
        v = self.props().get(k, {})
        if isinstance(v, dict):
            return dict_to_namespace(v)
        else:
            return v

    @lru_cache(maxsize=32)
    def props(self):
        props = ifcopenshell.util.element.get_psets(self.inst)

        if ifcopenshell.util.element.get_type(self.inst):
            props.update(ifcopenshell.util.element.get_psets(ifcopenshell.util.element.get_type(self.inst)))
            
        for v in list(props.values()):
            props.update(v)
            
        return props


if __name__ == "__main__":

    id = sys.argv[1]
    
    f = ifcopenshell.open(f"{id}.ifc")

    json.dump(
        to_dict(f, ET.parse(
            f"{id}.xml",
            parser=ET.XMLParser(encoding="utf-8")
        ).getroot()),
        open(f"{id}.tree.json", "w", encoding="utf-8")
    )