import json
import time, os
import pandas as pd
from redis import Redis
import datetime
import numpy as np
from threading import Thread
# from concurrent.futures import ThreadPoolExecutor
import queue
import json
import logging
from events import Events
import logging

logger = logging.getLogger("__main__."+__name__)
logger.setLevel(logging.DEBUG)
from os import environ
import socket

class redis_conn(object):
    def __init__(self,ip,port,db=0):
        self.hostname = environ.get("REDIS_HOSTNAME", ip)
        self.port = environ.get("REDIS_PORT", port)
        self.db = db

        self.conn = Redis(self.hostname, port, db=db,
                          retry_on_timeout=True, decode_responses=True,
                          socket_keepalive=True
                          )

class redis_subscriber(redis_conn):
    def __init__(self, ip, port, db=0, verbose=False):
        super(redis_subscriber, self).__init__(ip=ip, port=port, db=db)
        # self.pubsub = self.conn.pubsub()
        self.list_of_thread = []
        self.block_ms = 100
        self._verbose =verbose

    def Listen_on_Thread(self, key, last_id='$'):
        print(f"starting thread listener: {key} id:{last_id}")
        athread = Thread(target=self.listen_to_stream, args=(key, last_id), daemon=True)
        athread.start()
        self.list_of_thread.append(athread)

    def listen_to_stream(self, key, last_id='$'):
        self.listening = True
        while self.listening:
            try:
                # print(f"try: {key} id:{last_id}")
                resp = self.conn.xread({key: last_id}, count=1, block=self.block_ms)
                # if self._verbose:
                #     print(resp)
                if resp:
                    # if len(resp)>0:
                    if self._verbose:
                        print(resp)
                    key, messages = resp[0]
                    last_id, data = messages[0]
                    # print(key, resp)
                    self.process_message(key, last_id, data)
            except ConnectionError as e:
                logger.debug(f"ERROR REDIS_DB {e}")
                print("ERROR REDIS_DB CONNECTION: {}".format(e))
            except Exception as e:
                print(e)
                logging.debug(f"Other Error: {e}")

    def stop_listen(self):
        self.listening = False

    def process_message(self, key, stream_id, data):
        pass
        '''
        Handle data here
        convert dict to whatever and append to list
        
        '''
