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

import os
import string
import tempfile

from random import SystemRandom
choice = lambda seq: SystemRandom().choice(seq)
letter_set = set(string.ascii_letters)

from minio import Minio


if os.environ.get("MINIO_HOST"):
    STORAGE_DIR = tempfile.gettempdir()
    OUTPUT_DIR = tempfile.gettempdir()
else:
    STORAGE_DIR = os.environ.get("MODEL_DIR", tempfile.gettempdir())
    OUTPUT_DIR = os.environ.get("OUTPUT_DIR", STORAGE_DIR)
    
    
def generate_id():
    return "".join(choice(string.ascii_letters) for i in range(32))


def storage_dir_for_id(id, output=False):
    id = id.split("_")[0]
    return os.path.join([STORAGE_DIR, OUTPUT_DIR][output], id[0:1], id[0:2], id[0:3], id)


def storage_file_for_id(id, ext, ensure=True, **kwargs):
    if ensure:
        ensure_file(id, ext, **kwargs)
    return os.path.join(storage_dir_for_id(id, **kwargs), id + "." + ext)
    
    
def ensure_file(id, ext, **kwargs):
    path = storage_file_for_id(id, ext, ensure=False, **kwargs)
    if os.environ.get("MINIO_HOST"):
        if not os.path.exists(path):
            if not os.path.exists(storage_dir_for_id(id)):
                os.makedirs(storage_dir_for_id(id))
            client = Minio(os.environ.get("MINIO_HOST"), "minioadmin", "minioadmin", secure=False)
            if not client.bucket_exists("ifc-pipeline"):
                client.make_bucket("ifc-pipeline")
            print("ensure_file", id, ext)
            try:
                client.fget_object("ifc-pipeline", id.split("_")[0] + "/" + id + "." + ext, path)
            except: pass
    return path        
            
def store_file(id, ext):
    if os.environ.get("MINIO_HOST"):
        path = storage_file_for_id(id, ext)
        client = Minio(os.environ.get("MINIO_HOST"), "minioadmin", "minioadmin", secure=False)
        if not client.bucket_exists("ifc-pipeline"):
            client.make_bucket("ifc-pipeline")
        client.fput_object("ifc-pipeline", id.split("_")[0] + "/" + id + "." + ext, path)
        print("store", id, ext)
    else:
        print("not storing", id, ext)

            
def validate_id(id):
    id_num = id.split("_")
    
    if len(id_num) == 1:
        id = id_num[0]
    elif len(id_num) == 2:
        id, num = id_num
        num = str(int(num))
    else:
        return False

    return len(set(id) - set(string.ascii_letters)) == 0

