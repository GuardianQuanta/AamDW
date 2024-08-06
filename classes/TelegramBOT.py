
import queue
from threading import Thread
import asyncio
class TG_BOT_Helper():
    def __init__(self,bot_instances):
        self._bot = bot_instances
        self._group_map = {}
        self.default_group_name = ""
        self.stream_write_queue = queue.Queue()

    def __del__(self):
        self.allow_write_to_stream = False

    def stop(self):
        self.allow_write_to_stream = False

    def add_group(self,group_name,group_id):
        self._group_map[group_name] = group_id

    def set_default_group_name(self,group_name):
        self.default_group_name = group_name


    def start_loop(self):

        athread = Thread(target=self.loop_async, args=(), daemon=True)
        athread.start()

        # asyncio.get_event_loop().run_until_complete()
        # self.list_of_thread.append(athread)

    def loop_async(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.loop_notification())

    async def loop_notification(self):
        self.allow_write_to_stream = True

        while self.allow_write_to_stream:
            text,group_name = self.stream_write_queue.get()
            print(f"Sending: {text} to {group_name}")
            await self.send_notification(text,group_name)

    def push_message_to_queue(self,text,group_name=""):
        self.stream_write_queue.put( (text,group_name) )


    # def send_notification(self,text,group_name=''):
    #
    #     #    asyncio.get_event_loop().run_until_complete(bot.send_message(text="Gold data Update starting", chat_id=group_id))
    #     if group_name == '':
    #         if self.default_group_name == "":
    #             return
    #         else:
    #             await (self._bot.send_message(text=text, chat_id=self._group_map[self.default_group_name]))
    #     else:
    #         await self._bot.send_message(text=text, chat_id=self._group_map[group_name])


    async def send_notification(self,text,group_name=''):

        #    asyncio.get_event_loop().run_until_complete(bot.send_message(text="Gold data Update starting", chat_id=group_id))
        if group_name == '':
            if self.default_group_name == "":
                return
            else:
                await (self._bot.send_message(text=text, chat_id=self._group_map[self.default_group_name]))
        else:
            await self._bot.send_message(text=text, chat_id=self._group_map[group_name])

