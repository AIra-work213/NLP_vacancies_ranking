"""Отложенные этапы отбора примеров и дообучения"""


def mine_negatives(config: dict[str, object]) -> None:
    """Обозначает будущий отбор отрицательных примеров без утечки тестовых данных"""
    raise NotImplementedError("Отбор отрицательных примеров ещё не реализован")


def train(config: dict[str, object]) -> None:
    """Обозначает отложенный этап дообучения модели"""
    raise NotImplementedError("Дообучение пока отложено")
