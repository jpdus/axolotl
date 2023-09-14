from axolotl.prompt_tokenizers import (
    AlpacaPromptTokenizingStrategy,
    InstructionPromptTokenizingStrategy,
)
from axolotl.prompters import AlpacaPrompter, PromptStyle, UnpromptedPrompter


class AIROBOROSPrompter(AlpacaPrompter):
    system_prompt = ""
    system_no_input_prompt = ""
    system_format: str = ""
    turn_format = "{input}\nUSER: {instruction}\nASSISTANT: "
    turn_no_input_format = "USER: {instruction}\nASSISTANT: "

    def __init__(self):  # pylint: disable=super-init-not-called
        pass


class AiroborosTokenizingStrategy(InstructionPromptTokenizingStrategy):
    def parse_instruction_fields(self, prompt):
        return (
            prompt["instruction"],  # instruction
            prompt["system"] if "system" in prompt else "",  # input
            prompt["response"],  # response
        )


def load(tokenizer, cfg):
    return AiroborosTokenizingStrategy(
        AIROBOROSPrompter(),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )
