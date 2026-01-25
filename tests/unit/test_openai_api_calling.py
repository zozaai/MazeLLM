# tests/scripts/test_openai_api_calling.py
from dotenv import load_dotenv
from openai import OpenAI


def get_chat_response(client: OpenAI, prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are concise and helpful."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def test_openai_chat_response_mocked(mocker):
    """
    Unit test with OpenAI fully mocked (no network call).
    """

    # ---- mock OpenAI response structure ----
    mock_choice = mocker.Mock()
    mock_choice.message.content = "Mocked response"

    mock_response = mocker.Mock()
    mock_response.choices = [mock_choice]

    mock_client = mocker.Mock()
    mock_client.chat.completions.create.return_value = mock_response

    # ---- call function ----
    result = get_chat_response(mock_client, "Hello")

    # ---- assertions ----
    assert result == "Mocked response"
    mock_client.chat.completions.create.assert_called_once()
