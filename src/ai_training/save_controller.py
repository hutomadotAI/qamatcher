import logging
import multiprocessing


def _get_logger():
    logger = logging.getLogger('hu.ai_training.save')
    return logger


class SaveController:
    """A multiprocess aware class that can be passed into processes to keep shared state"""

    def __init__(self, manager: multiprocessing.Manager):
        self.__ai_save_status = manager.dict()
        self.logger = _get_logger()

    def track_ai(self, ai_id: str):
        """track AI"""
        self.__ai_save_status[ai_id] = False
        self.logger.info("track AI save state: {}".format(ai_id))

    def forget_ai(self, ai_id: str):
        """forget AI"""
        if ai_id in self.__ai_save_status:
            del self.__ai_save_status[ai_id]
            self.logger.info("forget AI save state: {}".format(ai_id))

    def set_save_state(self, ai_id: str, value: bool):
        """Set save state for an AI"""
        if ai_id in self.__ai_save_status:
            existing_state = self.get_save_state(ai_id)
            if value != existing_state:
                self.__ai_save_status[ai_id] = value
                self.logger.info("set_save_state: AI:{} is now {}".format(
                    ai_id, value))

    def get_save_state(self, ai_id: str):
        """Get save state for an AI"""
        save_state = self.__ai_save_status.get(ai_id, False)
        return save_state

    def __getstate__(self):
        """This function is used to control pickling of this object so it can
        be shared across processes"""
        state = self.__dict__.copy()
        del state['logger']
        return state

    def __setstate__(self, state):
        """This function is used to control pickling of this object so it can
        be shared across processes"""
        # We need this function to be present but it just needs to do the standard job
        # of updating the internal dictionary based on state
        self.__dict__.update(state)
        self.logger = _get_logger()
