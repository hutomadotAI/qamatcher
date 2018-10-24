import numpy as np
import aiohttp


class EmbeddingError(Exception):
    pass


class EmbeddingClient:
    def __init__(self, service_url, client_session: aiohttp.ClientSession):
        self.service_url = service_url
        self.client_session = client_session

    async def embedding_call(self, payload, endpoint='get_embedding'):
        try:
            async with self.client_session.post(
                    self.service_url + "/" + endpoint, json=payload) as resp:
                status = resp.status
                if status != 200:
                    raise EmbeddingError(
                        "Embedding call failed with status {}".format(status))
                response_json = await resp.json()

            if response_json is None:
                raise EmbeddingError("Response was none")

            return response_json
        except (aiohttp.client_exceptions.ClientConnectorError,
                aiohttp.client_exceptions.ContentTypeError) as exc:
            raise EmbeddingError("aiohttp error", exc)

    async def get_sentence_embedding(self, sentence):
        payload = {'sen': sentence}

        response = await self.embedding_call(payload)

        sen_emb = np.array(response['predictions'][0], dtype='float32')
        return sen_emb
