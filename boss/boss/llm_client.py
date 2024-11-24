from dotenv import load_dotenv

load_dotenv()

import logging
import os
from typing import Dict, List

from openai import AsyncOpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def call_openai_api_structured(
    messages: List[Dict[str, str]],
    response_model: type[BaseModel],
    model: str = "gpt-4o-mini",
) -> BaseModel:
    """sel
    Make a structured call to OpenAI API with Pydantic model parsing
    """
    try:
        for msg in messages:
            if not isinstance(msg, dict):
                raise ValueError("Each message must be a dict")
            if "role" not in msg or "content" not in msg:
                raise ValueError("Messages must have 'role' and 'content' keys")

        # Make API call
        completion = await openai_client.beta.chat.completions.parse(
            messages=messages,
            response_format=response_model,
            model=model,
        )

        # Parse response into model
        response_text = completion.choices[0].message.content
        return response_model.model_validate_json(response_text)

    except Exception as e:
        logger.error(f"Error in structured OpenAI API call: {e}")
        raise
