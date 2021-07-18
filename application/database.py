import os

if os.environ.get('NO_DATABASE', 'false').lower() in ('1', 'true'):
    from redis_database import *
else:
    from sql_database import *
