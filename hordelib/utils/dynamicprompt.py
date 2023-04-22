import random
import re


class DynamicPromptParser:
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)

    def _replace_with_random_word(self, match):
        words = match.group(1).split("|")
        return random.choice(words)

    def parse(self, input_string):
        pattern = r"\{([^{}]+)\}"

        while re.search(pattern, input_string):
            input_string = re.sub(pattern, self._replace_with_random_word, input_string)

        return input_string
