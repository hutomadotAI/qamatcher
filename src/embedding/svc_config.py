import os


class SvcConfig(object):

    __instance = None

    def __init__(self):
        self._w2v_server_url = os.environ.get('W2V_SERVER_URL',
                                              'http://ai-word2vec:9090')
        self._er_server_url = os.environ.get('ER_SERVER_URL',
                                             'http://api-entity:9095')
        self._emb_server_url = os.environ.get('EMB_SERVER_URL',
                                              'http://10.8.0.26:8097')

        self._server_port = os.environ.get('EMB_SERVER_PORT', '9090')

    @staticmethod
    def get_instance():
        if SvcConfig.__instance is None:
            SvcConfig.__instance = SvcConfig()
        return SvcConfig.__instance

    @property
    def w2v_server_url(self):
        return self._w2v_server_url

    @property
    def er_server_url(self):
        return self._er_server_url

    @property
    def emb_server_url(self):
        return self._emb_server_url

    @property
    def server_port(self):
        return int(self._server_port)
