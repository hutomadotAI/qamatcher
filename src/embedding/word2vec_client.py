import numpy
import json
import aiohttp


class Word2VecClient(object):
    def __init__(self, service_url):
        self.service_url = service_url
        self.client_session = aiohttp.ClientSession()

    async def get_vectors_for_words(self, words):
        list = []
        for word in words:
            list.append(word)
        json_payload = json.dumps({'words': list})
        headers = {'Content-type': 'application/json'}

        async with self.client_session.post(
                self.service_url + '/words', data=json_payload,
                headers=headers) as resp:
            status = resp.status
            response = await resp.json()

        if status == 200 and response is not None:
            json_payload = json.loads(response)
            if 'vectors' in json_payload:
                words_dict = json_payload['vectors']
                vectors_dict = {}
                for word, vecs in words_dict.items():
                    if vecs is not None:
                        vectors_dict[word] = numpy.array(vecs, dtype='float32')
                return vectors_dict
        return None
