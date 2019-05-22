"""
Abstract Base Class to be implemented by an AI Engine to represent the
AI Engine itself
(that can be controlled over HTTP)
"""
import abc
import asyncio
import collections.abc
import logging
import sys

import ai_training as ait


def _get_logger():
    logger = logging.getLogger('hu.ai_training.interface')
    return logger


class AiTrainingProviderABC(abc.ABC, collections.abc.Mapping):
    """Abstract base class for training provider such as WNET or RNN"""

    def __init__(self):
        self.__controller = None
        self.training_list = {}

    @property
    @abc.abstractmethod
    def ai_engine_name(self) -> str:
        """Get the name of the AI Engine (backend) for status updates to API"""

    @property
    @abc.abstractmethod
    def config(self) -> ait.Config:
        """Get the config object"""

    @abc.abstractmethod
    async def on_startup(self):
        """Called when server is started"""

    async def on_shutdown(self):
        """Called when server is shut down"""
        for key, item in self.training_list.items():
            await item.shutdown()

    @abc.abstractmethod
    def training_item_factory(self, dev_id, ai_id) -> ait.AiTrainingItemABC:
        """Called when need to create a new training item"""

    async def load_training_data_async(self, dev_id,
                                       ai_id) -> ait.AiTrainingItemABC:
        """Load training data"""
        if ait.check_training_exists(self.config.training_data_root, dev_id,
                                     ai_id):
            item = self.lookup_item(dev_id, ai_id)
            if item:
                item.update_status_from_file()
            else:
                item = self.create(dev_id, ai_id)
            return item

        return None

    def kill_running_process(self):
        """This is a function to kill the process. This aborts testing really badly.
        so by putting it here it can be overridden for test purposes.
        This is designed to be called from the shutdown watchdog"""
        sys.exit()

    # concrete methods
    @property
    def controller(self) -> ait.AiTrainingControllerABC:
        """Controller callback object"""
        return self.__controller

    @controller.setter
    def controller(self, value: ait.AiTrainingControllerABC):
        """Controller callback object"""
        self.__controller = value

    # required implement MutableMapping ABC so that this class can be
    # iterated for
    # the AIs available
    def __getitem__(self, key):
        return self.training_list[key]

    def __iter__(self):
        return iter(self.training_list)

    def __len__(self):
        return len(self.training_list)

    def create(self, dev_id, ai_id) -> ait.AiTrainingItemABC:
        """Creates a new AI with IDs"""
        if self.lookup_item(dev_id, ai_id):
            raise ait.TrainingAlreadyExistsError(dev_id, ai_id)

        item = self.training_item_factory(dev_id, ai_id)
        item.max_chat_lock_seconds = self.config.max_chat_lock_seconds
        if self.controller:
            item.controller = self.controller
        self.training_list[(dev_id, ai_id)] = item
        return item

    async def delete_ai(self, dev_id, ai_id):
        item = self.lookup_item(dev_id, ai_id)
        if item is None:
            raise ait.TrainingNotFoundError(dev_id, ai_id)
        del self.training_list[(dev_id, ai_id)]
        coro = self._delete_ai(item)
        asyncio.create_task(coro)

    async def delete_dev(self, dev_id):
        keys_to_delete = []
        for key, value in self.training_list.items():
            if value.dev_id == dev_id:
                keys_to_delete.append(key)

        if len(keys_to_delete) == 0:
            raise ait.TrainingNotFoundError(dev_id)
        for key in keys_to_delete:
            await self.delete_ai(key[0], key[1])

    def lookup_item(self, dev_id, ai_id):
        try:
            item = self.training_list[(dev_id, ai_id)]
        except KeyError:
            return None
        return item

    async def _delete_ai(self, item):
        await item.shutdown()
        ait.delete_ai_files(
            self.config.training_data_root, item.dev_id, item.ai_id)
