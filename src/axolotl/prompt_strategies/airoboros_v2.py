"""Module containing the SimpleShareGPTPromptTokenizingStrategy class"""

import logging
from dataclasses import dataclass, field
from typing import Generator, List, Tuple

from axolotl.prompt_strategies.sharegpt_simple import (
    SimpleShareGPTPromptTokenizingStrategy,
)

LOG = logging.getLogger("axolotl")


@dataclass
class ARConversation:
    """A class that keeps all conversation history."""

    # The name of this template
    name: str = "airoboros_v2"
    # The template of the system prompt
    # SYSTEM: is only a divider and is removed by strategy
    system_template: str = "{system_message}"
    # The system message
    system_message: str = "A chat."
    # The names of two roles
    roles = ("USER", "ASSISTANT")
    messages: List[List[str]] = field(default_factory=list)
    sep = "\n"
    sep2 = "\n"  # EOS token wird von tokenizing strat gesetzt

    def get_prompt(self) -> Generator[Tuple[str, str], None, None]:
        system_prompt = self.system_template.format(system_message=self.system_message)
        seps = [self.sep, self.sep2]
        yield ("SYSTEM:", system_prompt + seps[0])
        for i, (role, message) in enumerate(self.messages):
            if message:
                yield (role + ":", " " + message + seps[i % 2])
            else:
                LOG.warning(f"role with empty message: {role}")
                yield (role + ":", "")

    def copy(self):
        return ARConversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            messages=[[x, y] for x, y in self.messages],
        )

    def append_message(self, role, message):
        self.messages.append([role, message])


SHAREGPT_ASSERTION_FAILED_ROLE = (
    "Role did not alternate between turns (gpt and human). Please check your data."
)


class ARShareGPTPrompter:  # pylint: disable=too-few-public-methods
    """
    A prompter that generates prompts for the ShareGPT
    """

    def __init__(self):
        self._conversation = ARConversation()

    def build_prompt(self, source) -> Generator[str, None, None]:
        if len(source) < 2:
            # If there isn't a back and forth conversation, ignore it
            # also happens on the data splitting leaving empty conversations
            raise IndexError(
                f"A conversation entry has less than 2 messages :\n{source}"
            )

        conv = self._conversation.copy()

        # Add the conversation system prompt if provided, otherwise use the default one
        if source[0]["from"] == "system":
            conv.system_message = source[0]["value"]
            source.pop(0)

        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        try:
            # Apply prompt templates
            if (
                source[0]["from"] not in roles
                or roles[source[0]["from"]] != conv.roles[0]
            ):
                # Skip the first one if it is not from human
                source = source[1:]
        except IndexError as err:
            # sometimes there is a bing or system chat
            raise err

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], SHAREGPT_ASSERTION_FAILED_ROLE
            conv.append_message(role, sentence["value"])

        for part in conv.get_prompt():
            yield part


def load(tokenizer, cfg):
    return SimpleShareGPTPromptTokenizingStrategy(
        ARShareGPTPrompter(),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )
