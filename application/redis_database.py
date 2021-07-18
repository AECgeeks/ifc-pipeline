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

import redis


class redis_serializable:
    """
    Constructs key from table name, primary key and column name
    and sets and retrieves values.
    """

    def __init__(self, session, obj):
        self.session, self.obj = session, obj

    def make_key(self, k):
        return "/".join((self.obj.tablename, self.obj.id, k))

    def get(self, k):
        return self.session.redis.get(self.make_key(k)).decode("utf-8")

    def set(self, k, v):
        return self.session.redis.set(self.make_key(k), v)


class id_obj:
    """
    A helper object used in queries to fix an id to a specific
    value.
    """

    def __init__(self, table, value):
        self.table, self.id = table, value

    tablename = property(lambda self: self.table.__name__)


class query_obj_all:
    """
    A bit silly, we already fetched the results, but
    just in order to be compatible with sqlalchemy.
    """

    def __init__(self, results):
        self.results = results

    def all(self):
        return self.results


class query_obj:
    """
    A query object with filter(). It can only filter on primary key.
    """

    def __init__(self, session, table):
        self.session, self.table = session, table

    def filter(self, query):
        rs = redis_serializable(self.session, id_obj(self.table, query.value))
        di = {}
        for x in dir(self.table):
            v = getattr(self.table, x)
            if isinstance(v, column_desc):
                di[x] = v.type(rs.get(x))
        el = self.table(**di)
        self.session.add(el)
        return query_obj_all([el])


class Session:
    """
    A session object with add() commit() and query() methods. Note
    that always all parameters of add()'ed objects are set to Redis,
    it is not recorded which fields has been changed. There is also
    no transaction or pipeline in place.
    """

    def __init__(self):
        self.redis = redis.Redis(host=os.environ.get("REDIS_HOST", "localhost"))
        self.elems = []

    def add(self, elem):
        self.elems.append(elem)

    def commit(self):
        for elem in self.elems:
            rs = redis_serializable(self, elem)
            for x in dir(type(elem)):
                v = getattr(type(elem), x)
                if isinstance(v, column_desc):
                    rs.set(x, getattr(elem, x))

    def close(self):
        pass

    def query(self, table):
        return query_obj(self, table)


class filter_obj:
    """
    Pair of column and value used in querying
    """

    def __init__(self, column, value):
        self.column, self.value = column, value


class column_desc:
    """
    Pair of name and type. Can be used with == operator
    to create query filters
    """

    def __init__(self, name, type):
        self.name, self.type = name, type

    def __eq__(self, other):
        return filter_obj(self, other)


class model:
    code = column_desc("code", str)
    filename = column_desc("filename", str)
    progress = column_desc("progress", int)

    def __init__(self, code, filename, progress=-1):
        self.code, self.filename, self.progress = code, filename, progress

    def _get_id(self):
        return self.code

    id = property(_get_id)
    tablename = property(lambda self: type(self).__name__)
