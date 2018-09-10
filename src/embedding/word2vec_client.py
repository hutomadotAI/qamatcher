import numpy
import aiohttp


class Word2VecError(Exception):
    pass


class Word2VecClient():
    def __init__(self, service_url, client_session: aiohttp.ClientSession):
        self.service_url = service_url
        self.client_session = client_session

    async def w2v_call(self, payload, endpoint='words'):
        try:
            async with self.client_session.post(
                    self.service_url + "/" + endpoint, json=payload) as resp:
                status = resp.status
                if status != 200:
                    raise Word2VecError(
                        "Word2Vec call failed with status {}".format(status))
                response_json = await resp.json()

            if response_json is None:
                raise Word2VecError("Response was none")

            return response_json
        except (aiohttp.client_exceptions.ClientConnectorError,
                aiohttp.client_exceptions.ContentTypeError) as exc:
            raise Word2VecError("aiohttp error", exc)

    async def get_unknown_words(self, words):
        word_list = [word for word in words]
        payload = {'words': word_list}

        response = await self.w2v_call(payload, endpoint='unk_words')

        unk_words = response['unk_words']
        return unk_words

    async def get_vectors_for_words(self, words):
        word_list = [word for word in words]
        payload = {'words': word_list}

        response = await self.w2v_call(payload)

        words_dict = response['vectors']
        vectors_dict = {}
        for word, vecs in words_dict.items():
            vectors_dict[word] = numpy.array(vecs, dtype='float32')
        return vectors_dict
