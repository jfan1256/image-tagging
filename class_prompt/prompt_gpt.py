import numpy as np
import ray
import json
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from openai import OpenAI
from tqdm import tqdm
from tenacity import retry, wait_random_exponential, stop_after_attempt, before_sleep_log

from utils.system import *

class PromptGPT:
    def __init__(self,
                 items=None,
                 model=None,
                 batch_size=None,
                 ):

        '''
        tags (list): List of items pass through GPT
        model (str): GPT Model used for Prompt Engineering
        batch_size (int): Batch size for parallelization
        '''

        self.items = items
        self.model = model
        self.batch_size = batch_size

    @staticmethod
    @ray.remote
    def gpt_topic(item):
        api_key = json.load(open(get_config() / 'api.json'))['api_key']
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f"You are a clueless user of a digital asset management software that helps you find assets on your "
                                            f"computer by searching for content using AI rather that metadata. Generate ONE short search query " 
                                            f"using only the noun of the image you're looking for with adjectives that describe it in plain " 
                                            f"english that you could use to find an image file with the given tags: {item}. "
                                            f"Only use the information from a few of the tags to generate the prompt." 
                                            f"Do not search for anything but the information in tags list provided"
                                            f"Do not speak to the quality of the image, only the tags. Put some randomness in the "
                                            f"structure of your query, in lowercase with no punctuation. They should not all sound exactly the same. Make it concise."}
            ],
            temperature=0.75,
            max_tokens=500,
            top_p=0.75,
            frequency_penalty=0,
            presence_penalty=0,
            seed=1
        )
        summary = response.choices[0].message.content.translate(str.maketrans('', '', '."'))
        return summary

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6), before_sleep=before_sleep_log(logger, logging.INFO))
    def retry_gpt_topic(self, item):
        # Call the remote function from within the retry logic
        return self.gpt_topic.remote(item)

    def get_gpt_topic(self):
        ray.init(num_cpus=16, ignore_reinit_error=True)
        num_batches = np.ceil(len(self.items) / self.batch_size)
        all_summary = []
        for i in tqdm(range(int(num_batches)), desc='Processing batches'):
            start_index = i * self.batch_size
            end_index = min(start_index + self.batch_size, len(self.items))
            batch = self.items[start_index:end_index]

            # Start asynchronous tasks for the batch
            futures = [self.retry_gpt_topic(item) for item in batch]
            batch_summaries = ray.get(futures)

            # Update lists
            all_summary.extend(batch_summaries)
            time.sleep(1)

        ray.shutdown()
        return all_summary