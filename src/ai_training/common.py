"""
Common definitions for AI training
"""

import enum
import logging
import pickle


def _get_logger():
    logger = logging.getLogger('hu.ai_training.common')
    return logger


class AiTrainingState(enum.Enum):
    """AI training status enum
    Simplified state list with no duplicates or ambiguity"""
    ai_undefined = enum.auto()
    ai_ready_to_train = enum.auto()
    ai_training_queued = enum.auto()
    ai_training = enum.auto()
    ai_training_stopped = enum.auto()
    ai_training_complete = enum.auto()
    ai_error = enum.auto()


class Error(Exception):
    """Base error for this module"""
    pass


class TrainingAlreadyExistsError(Error):
    """Exception raised if training already exists"""

    def __init__(self, dev_id, ai_id):
        Error.__init__(self)
        self.dev_id = dev_id
        self.ai_id = ai_id


class TrainingNotFoundError(Error):
    """Exception raised if training not found"""

    def __init__(self, dev_id=None, ai_id=None):
        Error.__init__(self)
        self.dev_id = dev_id
        self.ai_id = ai_id


class TrainingFailedError(Error):
    """Exception raised if training fails"""

    def __init__(self, training_status):
        Error.__init__(self)
        self.training_status = training_status


class TrainingSyntaxError(Error):
    """Exception raised if training fails"""
    pass


class TrainingStatusError(Error):
    """Exception raised if training status on file is invalid"""
    pass


class ChatStateError(Error):
    """Exception raised if training status is invalid for chat"""

    def __init__(self, message, aiid=None, state=None):
        super().__init__()
        self.message = message
        self.aiid = aiid
        self.state = state


class ChatAiHashError(Error):
    """Exception raised if AI hash is invalid for chat"""
    pass


class ChatOverloadedError(Error):
    """Exception raised if chat is overloaded"""

    def __init__(self, message, error_data=None):
        super().__init__()
        self.message = message
        if error_data is None:
            error_data = {}
        self.error_data = error_data


STATUS_VERSION = 20170215


class AiTrainingStatusWithProgress:
    """Simple object that represents a training state,
    easily stored to pickle"""

    def __init__(self,
                 state,
                 training_progress=None,
                 training_error=None,
                 training_file_hash=None,
                 training_data_hash=None):
        self.version = STATUS_VERSION
        self.state = state
        self.training_progress = training_progress
        self.training_error = training_error
        self.training_file_hash = training_file_hash
        self.training_data_hash = training_data_hash

    def __repr__(self):
        if isinstance(self.training_progress, float):
            training_progress_str = "{:.2f}".format(self.training_progress)
        else:
            training_progress_str = str(self.training_progress)

        if isinstance(self.training_error, float):
            training_error_str = "{:.2e}".format(self.training_error)
        else:
            training_error_str = str(self.training_error)

        ss = "AiTrainingStatusWithProgress({}, {}, {}, {}, {})".format(
            self.state, training_progress_str, training_error_str,
            self.training_file_hash, self.training_data_hash)
        return ss

    def __eq__(self, other):
        if not isinstance(other, AiTrainingStatusWithProgress):
            return False

        is_equal = (self.state == other.state
                    and self.training_progress == other.training_progress
                    and self.training_error == other.training_error
                    and self.training_file_hash == other.training_file_hash
                    and self.training_data_hash == other.training_data_hash)
        return is_equal

    @property
    def is_training(self):
        return self.state == AiTrainingState.ai_training

    @property
    def is_stopped(self):
        return (self.state == AiTrainingState.ai_ready_to_train
                or self.state == AiTrainingState.ai_training_queued
                or self.state == AiTrainingState.ai_training_stopped
                or self.state == AiTrainingState.ai_training_complete)

    @property
    def can_chat(self):
        return (self.state == AiTrainingState.ai_training
                or self.state == AiTrainingState.ai_training_queued
                or self.state == AiTrainingState.ai_training_stopped
                or self.state == AiTrainingState.ai_training_complete)

    @property
    def training_hash(self):
        if self.training_file_hash is None:
            return None
        if self.training_data_hash is None:
            return None
        combined_hash = "{}-{}".format(self.training_file_hash,
                                       self.training_data_hash)
        return combined_hash

    @staticmethod
    def load(file):
        try:
            with file.open('rb') as fp:
                file_value = pickle.load(fp)
        except (pickle.PickleError, ImportError, OSError, EOFError) as exc:
            raise TrainingStatusError from exc

        if not isinstance(file_value, AiTrainingStatusWithProgress):
            raise TrainingStatusError
        if file_value.version != STATUS_VERSION:
            logger = _get_logger()
            logger.warning(
                "Mismatched version of AiTrainingStatusWithProgress")
            raise TrainingStatusError
        return file_value

    @staticmethod
    def load_safe(file):
        try:
            return AiTrainingStatusWithProgress.load(file)
        except TrainingStatusError:
            return AiTrainingStatusWithProgress(AiTrainingState.ai_undefined)

    def save(self, file):
        """Save status to disk"""
        try:
            with file.open('wb') as fp:
                pickle.dump(self, fp)
        except OSError:
            logger = _get_logger()
            logger.warning("Failed to write AiTrainingStatusWithProgress "
                           "- read-only mode?")
