"""
Common AI HTTP code (asyncio/aiohttp)
"""

import json
import logging

from aiohttp import web

import ai_training as ait


def _get_logger():
    logger = logging.getLogger('hu.ai_training.http')
    return logger


def create_response_from_item(item: ait.AiTrainingItemABC):
    """Create web response"""
    status = item.status
    data = {
        'dev_id': item.dev_id,
        'ai_id': item.ai_id,
        'status': status.state.name
    }
    if status.training_progress is not None:
        data['training_progress'] = status.training_progress
    if status.training_error is not None:
        data['training_error'] = status.training_error
    data['ai_hash'] = status.training_hash
    resp = web.json_response(data)
    return resp


def raise_bad_request(message: str, error_type: type = web.HTTPBadRequest):
    """Create web response"""
    logger = _get_logger()
    logger.error('Bad request received error is: {}'.format(message))
    data = {'error': message}
    data_str = json.dumps(data)
    raise error_type(text=data_str, content_type="application/json")


def raise_bad_request_status(message: str, status: ait.AiTrainingState,
                             ai_id: str):
    """Create web response"""
    logger = _get_logger()
    logger.error(
        'Bad request received for {} in status {}, error is: {}'.format(
            ai_id, status, message))
    data = {'error': message, 'status': status.name}
    data_str = json.dumps(data)
    raise web.HTTPBadRequest(text=data_str)
