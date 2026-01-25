# tests/integration/test_openai_real.py
import os
import pytest
from openai import OpenAI


@pytest.mark.integration
def test_openai_real_call():
    if "OPENAI_API_KEY" not in os.environ:
        pytest.skip("OPENAI_API_KEY not set")

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello in one word"}],
    )

    assert len(response.choices[0].message.content) > 0
