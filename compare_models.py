import os
from time import perf_counter

import openai
from dotenv import load_dotenv
from openai import OpenAI

from main import (
    BASE_URL,
    MAX_OUTPUT_TOKENS,
    MODEL,
    TEMPERATURE,
    build_messages,
    validate_summary,
)


MODEL_A = MODEL
MODEL_B = "nex-agi/nex-n2.5-pro:free"

TEST_CASES = [
    "Клиент дважды оплатил один заказ.",
    (
        "Покупатель получил товар в поврежденной упаковке. "
        "Сам товар цел, но клиент хочет заменить заказ."
    ),
]


def evaluate_model(
    client: OpenAI,
    model_name: str,
    user_text: str,
) -> None:
    started_at = perf_counter()

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=build_messages(user_text),
            temperature=TEMPERATURE,
            max_completion_tokens=MAX_OUTPUT_TOKENS,
        )
    except openai.APIError as error:
        print(f"Модель: {model_name}")
        print(f"Ошибка API: {error}")
        print("-" * 60)
        return

    elapsed_seconds = perf_counter() - started_at
    choice = response.choices[0]

    if choice.finish_reason != "stop":
        answer = f"[Незавершенный результат: {choice.finish_reason}]"
    else:
        try:
            answer = validate_summary(choice.message.content)
        except ValueError as error:
            answer = f"[Некорректный результат: {error}]"

    print(f"Модель: {model_name}")
    print(f"Обращение: {user_text}")
    print(f"Ответ: {answer}")
    print(f"Завершение: {choice.finish_reason}")
    print(f"Время: {elapsed_seconds:.2f} с")

    if response.usage is not None:
        print(f"Входные токены: {response.usage.prompt_tokens}")
        print(f"Выходные токены: {response.usage.completion_tokens}")

    print("-" * 60)


def main() -> None:
    load_dotenv()

    token = os.getenv("LLM_API_KEY")
    if not token:
        raise SystemExit(
            "Не найдена переменная LLM_API_KEY. "
            "Проверьте файл .env."
        )

    client = OpenAI(base_url=BASE_URL, api_key=token)

    for model_name in (MODEL_A, MODEL_B):
        for user_text in TEST_CASES:
            evaluate_model(client, model_name, user_text)


if __name__ == "__main__":
    main()