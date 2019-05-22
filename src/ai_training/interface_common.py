"""
Common code to be used by an AI Engine for communication with the HTTP layer
"""

import abc
import multiprocessing

import ai_training.save_controller as save_controller


class AiTrainingControllerABC(abc.ABC):
    """The common Training Controller class for the AI Engine to communicate
    with the HTTP hosting layer"""

    @abc.abstractmethod
    async def training_callback(self, item):
        """Callback from an AI that there is status information to send to API"""

    @abc.abstractmethod
    def start_registration_with_api(self):
        """Ask controller to (re)register with API server"""

    @abc.abstractmethod
    async def create_training_process_pool(self, training_processes: int,
                                           training_queue_size: int,
                                           worker_type: type):
        """Create a training process pool"""

    @property
    @abc.abstractmethod
    def multiprocessing_manager(self) -> multiprocessing.Manager:
        """The multiprocessing manager instance"""

    @property
    @abc.abstractmethod
    def save_controller(self) -> save_controller.SaveController:
        """The multiprocessing manager instance"""
