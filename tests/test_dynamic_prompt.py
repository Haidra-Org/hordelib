from hordelib.utils.dynamicprompt import DynamicPromptParser


class TestDynamicPrompt:
    def test_basic(self):
        seed = 4434
        instr = "the {dog|cat|big monkey} is {near|beside|around} the {house|building|garden}."
        expected = "the big monkey is beside the house."
        result = DynamicPromptParser(seed).parse(instr)
        assert result == expected

    def test_nested(self):
        seed = 4434453134
        instr = (
            "the {dog|cat|big monkey|{frog|toad}} is "
            "{near|{beside|{inside|on}}|around} the {house|building|garden}."
        )
        expected = "the frog is on the building."
        result = DynamicPromptParser(seed).parse(instr)
        assert result == expected

    def test_empty(self):
        seed = 44344531
        instr = "the {dog|cat|big monkey|{frog|toad}} is {here|there|} and not {}"
        expected = "the toad is  and not {}"
        result = DynamicPromptParser(seed).parse(instr)
        assert result == expected

    def test_random(self):
        seed = None
        instr = "the {dog|cat|big monkey} is {near|beside|around} the {house|building|garden}."
        result = DynamicPromptParser(seed).parse(instr)
        assert "{" not in result
