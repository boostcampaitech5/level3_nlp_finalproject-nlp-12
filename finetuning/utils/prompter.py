"""
A dedicated helper to manage templates and prompt building.
"""

import re
import json
import os.path as osp
from typing import Union


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with 'kullm' and will not break.
            template_name = "multi"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from optional input
        # if a label (=response, =output) is provided, it's also appended.

        # 멀티 턴 대화 처리하는 부분
        def converter(sentence):
            result = re.sub(r"질문\s*", "input", sentence)
            result = re.sub(r"답변\s*", "response", result)

            return result
        
        instruction = converter(instruction)
        new_instruction = instruction.split('\n')[-1]
        history = instruction[:-len(new_instruction)]
        try:
            new_instruction = new_instruction.split('input: ')[1]
        except:
            new_instruction = new_instruction.split('input: ')[0]

        res = self.template["prompt"].format(history=history, instruction=new_instruction)

        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()
