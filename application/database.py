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

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func
from sqlalchemy.inspection import inspect
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy_utils import database_exists, create_database
import os

DEVELOPMENT = os.environ.get('environment', 'production').lower() == 'development'

if DEVELOPMENT:
    engine = create_engine('sqlite:///ifc-pipeline.db', connect_args={'check_same_thread': False})
else:
    engine = create_engine('postgresql://postgres:postgres@%s:5432/bimsurfer2' % os.environ.get('POSTGRES_HOST', 'localhost'))
    
Session = sessionmaker(bind=engine)

Base = declarative_base()


class Serializable(object):
    def serialize(self):
        return {c: getattr(self, c) for c in inspect(self).attrs.keys()}


class model(Base, Serializable):
    __tablename__ = 'models'

    id = Column(Integer, primary_key=True)
    code = Column(String)
    filename = Column(String)
    progress = Column(Integer, default=-1)
    date = Column(DateTime, server_default=func.now())

    def __init__(self, code, filename):
        self.code = code
        self.filename = filename


def initialize():
    if not database_exists(engine.url):
        create_database(engine.url)
    Base.metadata.create_all(engine)


if __name__ == "__main__" or DEVELOPMENT:
    initialize()
