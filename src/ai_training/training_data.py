"""
Utilities for handling training files
"""
import base64
import copy
import enum
import io
import logging
import hashlib

from pathlib import Path

from ai_training import TrainingSyntaxError

TOPIC_TAG = "@topic_out="


def _get_logger():
    logger = logging.getLogger('hu.training_file')
    return logger


# basic unit to store a Q&A sequence
class TrainingEntry:
    def __init__(self):
        self.question = None
        self.answer = None
        self.topic_out = None
        self.multi_turn_previous = None


class Topic:
    def __init__(self, topic_name):
        self.topic_name = topic_name
        self.entries = []
        self.sub_topics = {}

    def find_sub_topic(self, name):
        try:
            sub_topic = self.sub_topics[name]
        except KeyError:
            sub_topic = Topic(name)
            self.sub_topics[name] = sub_topic
        return sub_topic

    def is_empty(self):
        return not self.entries and not self.sub_topics


class TrainingV1ParseState(enum.Enum):
    """State machine to parse training file"""
    # cleared state, no entries in process - needs a blank line or
    # start of file
    cleared = enum.auto()
    # question expected next
    question = enum.auto()
    # answer expected next
    answer = enum.auto()

    @property
    def ready_for_question(self):
        if (self == TrainingV1ParseState.question
                or self == TrainingV1ParseState.cleared):
            return True
        return False


class ParseStatus:
    def __init__(self):
        self.current_topic = None
        self.current_entry = None
        self.previous_entry = None
        self.parse_state = TrainingV1ParseState.cleared
        self.line_counter = 0
        self.entry_counter = 0

    def reset_topic(self, topic):
        if not TrainingV1ParseState.ready_for_question:
            raise TrainingSyntaxError(
                'Invalid state {} for reset_topic line at #{}'.format(
                    self.parse_state, self.line_counter))
        self.current_topic = topic
        self.current_entry = TrainingEntry()
        self.previous_entry = None
        self.parse_state = TrainingV1ParseState.question


def file_load_training_data_v1(training_file: Path) -> Topic:
    with training_file.open(mode='r', encoding='utf-8') as training_handle:
        topic = _stream_load_training_data_v1(training_handle)

    return topic


def str_load_training_data_v1(training_data: str) -> Topic:
    with io.StringIO(training_data) as training_handle:
        topic = _stream_load_training_data_v1(training_handle)

    return topic


def hash_topic_name(topic_raw_name) -> str:
    md5 = hashlib.md5()
    topic_name_bytes = topic_raw_name.encode(encoding='utf8')
    md5.update(topic_name_bytes)
    digest = md5.digest()
    topic_hash_bytes = base64.b64encode(digest)
    topic_name = topic_hash_bytes.decode("utf-8")
    return topic_name


def _stream_load_training_data_v1(training_stream) -> Topic:  # noqa: C901
    # TODO(paulan): make this less complex
    logger = _get_logger()
    root_topic = Topic(None)
    status = ParseStatus()
    status.reset_topic(root_topic)
    status.parse_state = TrainingV1ParseState.cleared

    # walk the file generating the topic tree
    for line_with_line_endings in training_stream:
        line = line_with_line_endings.strip()
        if len(line) == 0:
            if status.parse_state is TrainingV1ParseState.answer:
                raise TrainingSyntaxError(
                    "Blank line after question - expected answer")
            status.reset_topic(root_topic)
            status.parse_state = TrainingV1ParseState.cleared
        elif line.startswith(TOPIC_TAG):
            if status.parse_state != TrainingV1ParseState.cleared:
                logger.error("Topic start must follow a blank line")
                raise TrainingSyntaxError(
                    "Topic start must follow a blank line")
            topic_raw_name = line.split(TOPIC_TAG)[1]
            topic_name = hash_topic_name(topic_raw_name)
            current_topic = root_topic.find_sub_topic(topic_name)
            status.reset_topic(current_topic)
        elif status.parse_state.ready_for_question:
            status.current_entry.topic_out = status.current_topic.topic_name
            status.current_entry.question = line
            if status.previous_entry is not None:
                status.current_entry.multi_turn_previous = \
                    status.previous_entry.answer
            status.parse_state = TrainingV1ParseState.answer
        elif status.parse_state == TrainingV1ParseState.answer:
            status.current_entry.answer = line
            status.current_topic.entries.append(status.current_entry)
            status.previous_entry = status.current_entry
            status.current_entry = TrainingEntry()
            status.parse_state = TrainingV1ParseState.question
            status.entry_counter += 1
        status.line_counter += 1

    if not status.parse_state.ready_for_question:
        logger.error("File ended with a question with no answer")
        raise TrainingSyntaxError("File ended with a question with no answer")

    # Post-processing of the tree of topics
    for topic_name, topic in root_topic.sub_topics.items():
        if len(root_topic.entries) == 0:
            logger.error(
                "Training file has no root entries, but has subtopics")
            raise TrainingSyntaxError("Training file has no root entries")

        if len(topic.entries) == 0:
            err_str = "Training file topic '{}' has no entries".format(
                topic_name)
            logger.error(err_str)
            raise TrainingSyntaxError(err_str)

        # post-processing to make sure that we can enter a topic
        # first entry of a topic is the way into a topic
        # *** make sure you take a deep copy (in case it is also LAST entry)
        first_entry = copy.deepcopy(topic.entries[0])
        root_topic.entries.append(first_entry)

        # post-processing to make sure that we can exit a topic
        # or multi-turn dialog
        # last entry of a topic is the way out of a topic / sub-topic
        last_entry = topic.entries[len(topic.entries) - 1]
        last_entry.topic_out = None

    num_topics = len(root_topic.sub_topics) + 1
    logger.info(
        "Parsed training file with {} lines, {} entries, {} topics".format(
            status.line_counter, status.entry_counter, num_topics))
    return root_topic
