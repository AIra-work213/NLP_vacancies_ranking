"""Нагрузка через настоящий HTTP: новые запросы, повторы и фильтры"""

import os
import random

from locust import HttpUser, between, task

TECHNOLOGIES = (
    "Python",
    "Java",
    "PostgreSQL",
    "Docker",
    "Linux",
    "SQL",
    "FastAPI",
    "Spring",
    "Redis",
    "Git",
    "JavaScript",
    "React",
    "Kubernetes",
    "Go",
)
REPEATED_QUERY = "Python backend разработчик FastAPI PostgreSQL"


class SearchUser(HttpUser):
    """Проверяет успешную выдачу и отдельно регистрирует управляемую перегрузку"""

    wait_time = between(0.05, 0.15)

    @task
    def search(self) -> None:
        """Смешивает сценарии либо повторяет один запрос для проверки кеша"""
        mode = os.environ.get("NLP_RANKER_LOAD_MODE", "mixed")
        body: dict = {"query": REPEATED_QUERY, "limit": 10}
        name = "Поиск: повтор"
        if mode == "mixed" and random.random() < 0.7:
            body["query"] = "Разработчик " + " ".join(random.sample(TECHNOLOGIES, 3))
            name = "Поиск: новый запрос"
            if random.random() < 0.3:
                body["filters"] = {"work_format": "remote"}
                name = "Поиск: фильтр remote"

        with self.client.post(
            "/v1/search", json=body, name=name, catch_response=True, timeout=15
        ) as response:
            if response.status_code == 200:
                value = response.json()
                if not isinstance(value.get("hits"), list) or not value.get(
                    "release_version"
                ):
                    response.failure("Ответ не соответствует поисковому контракту")
            elif response.status_code == 429:
                response.failure("429: ограничение очереди")
            else:
                response.failure(f"HTTP {response.status_code}")
