import random
import re


class DynamicPromptParser:
    def __init__(self, seed=None):
        self._seed = seed

    def _replace_with_random_word(self, match):
        words = match.group(1).split("|")
        return random.choice(words)

    def parse(self, input_string):
        rng_state = None
        if self._seed is not None:
            rng_state = random.getstate()
            random.seed(self._seed)

        pattern = r"\{([^{}]+)\}"

        while re.search(pattern, input_string):
            input_string = re.sub(pattern, self._replace_with_random_word, input_string)

        if rng_state:
            random.setstate(rng_state)
        return input_string
