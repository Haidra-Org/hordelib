import re
import string

from unidecode import unidecode


class Sanitizer:
    VERSION_REMOVER = re.compile(r"([Vv][0-9][0-9.]*)$")

    @staticmethod
    def sanitise_model_name(name):
        """Ensure we store model names in ascii"""
        # We just remove exotic unicode characters
        return unidecode(name)

    @staticmethod
    def sanitise_filename(filename):
        """Don't allow crazy filenames, there are a lot"""
        # First remove exotic unicode characters
        filename = unidecode(filename)
        # Now exploit characters
        valid_chars = f"-_.(){string.ascii_letters}{string.digits}"
        filename.replace(" ", "_")
        valid_name = "".join(c for c in filename if c in valid_chars)
        return valid_name.strip()

    @staticmethod
    def remove_version(string):
        return Sanitizer.VERSION_REMOVER.sub("", string)

    @staticmethod
    def has_unicode(string):
        for char in string:
            codepoint = ord(char)

            if codepoint > 0x80:
                return True
        return False
