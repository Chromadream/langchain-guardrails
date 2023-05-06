from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Extra

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from langchain.output_parsers import GuardrailsOutputParser
from langchain.prompts import PromptTemplate

class RAILChain(Chain):
    """
    RAIL Chain for non-chat model, like `text-davinci-003`
    """

    query: str
    """Business logic prompt object to use."""
    llm: BaseLanguageModel
    rail_output_spec: str
    """RAIL output spec"""
    output_key: str = "text"  #: :meta private:

    @property
    def prompt(self):
        full_prompt = """
<rail version="0.1">

<output>
{rail_output_spec}
</output>

<prompt>

{query}

@complete_json_suffix_v2
</prompt>
</rail>
""".format(rail_output_spec=self.rail_output_spec, query=self.query)
        import re
        variable_names = re.findall(r"\{\{(.*?)\}\}", full_prompt, re.MULTILINE | re.DOTALL)
        output_parser = GuardrailsOutputParser.from_rail_string(full_prompt)
        prompt = PromptTemplate(template=output_parser.guard.base_prompt, input_variables=variable_names)
        return prompt


    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return self.prompt.input_variables

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        # Your custom chain logic goes here
        # This is just an example that mimics LLMChain
        prompt_value = self.prompt.format_prompt(**inputs)
        # Whenever you call a language model, or another chain, you should pass
        # a callback manager to it. This allows the inner run to be tracked by
        # any callbacks that are registered on the outer run.
        # You can always obtain a callback manager for this by calling
        # `run_manager.get_child()` as shown below.
        response = self.llm.generate_prompt(
            [prompt_value],
            callbacks=run_manager.get_child() if run_manager else None
        )
        
        return {self.output_key: response.generations[0][0].text}

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        # Your custom chain logic goes here
        # This is just an example that mimics LLMChain
        prompt_value = self.prompt.format_prompt(**inputs)
        
        # Whenever you call a language model, or another chain, you should pass
        # a callback manager to it. This allows the inner run to be tracked by
        # any callbacks that are registered on the outer run.
        # You can always obtain a callback manager for this by calling
        # `run_manager.get_child()` as shown below.
        response = await self.llm.agenerate_prompt(
            [prompt_value],
            callbacks=run_manager.get_child() if run_manager else None
        )

        return {self.output_key: response.generations[0][0].text}

    @property
    def _chain_type(self) -> str:
        return "rail_chain"