from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class StrictSchemaError(ValueError):
    pass


def json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrictSchemaError(f"json: duplicate key {key!r}")
        result[key] = value
    return result


def reject_json_constant(value: str) -> Any:
    raise StrictSchemaError(f"json: invalid constant {value}")


def mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StrictSchemaError(f"{path}: expected mapping")
    for key in value:
        if type(key) is not str:
            raise StrictSchemaError(f"{path}: expected string keys")
    return value


def fields(
    value: Any,
    required: set[str] | frozenset[str],
    optional: set[str] | frozenset[str] = frozenset(),
    path: str = "value",
) -> Mapping[str, Any]:
    obj = mapping(value, path)
    keys = set(obj)
    missing = set(required) - keys
    unknown = keys - set(required) - set(optional)
    if missing:
        raise StrictSchemaError(f"{path}: missing fields {sorted(missing)}")
    if unknown:
        raise StrictSchemaError(f"{path}: unknown fields {sorted(unknown)}")
    return obj


def string(value: Any, path: str, *, allow_empty: bool = False) -> str:
    if type(value) is not str:
        raise StrictSchemaError(f"{path}: expected string")
    if not allow_empty and not value:
        raise StrictSchemaError(f"{path}: empty string")
    return value


def boolean(value: Any, path: str) -> bool:
    if type(value) is not bool:
        raise StrictSchemaError(f"{path}: expected bool")
    return value


def integer(
    value: Any,
    path: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise StrictSchemaError(f"{path}: expected int")
    if minimum is not None and value < minimum:
        raise StrictSchemaError(f"{path}: expected >= {minimum}")
    if maximum is not None and value > maximum:
        raise StrictSchemaError(f"{path}: expected <= {maximum}")
    return value


def number(
    value: Any,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if type(value) not in (int, float):
        raise StrictSchemaError(f"{path}: expected number")
    result = float(value)
    if result != result or result in (float("inf"), float("-inf")):
        raise StrictSchemaError(f"{path}: expected finite number")
    if minimum is not None and result < minimum:
        raise StrictSchemaError(f"{path}: expected >= {minimum}")
    if maximum is not None and result > maximum:
        raise StrictSchemaError(f"{path}: expected <= {maximum}")
    return result


def sequence(value: Any, path: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise StrictSchemaError(f"{path}: expected sequence")
    return value


def string_tuple(value: Any, path: str, *, nonempty: bool = True) -> tuple[str, ...]:
    result = tuple(string(item, f"{path}[{index}]") for index, item in enumerate(sequence(value, path)))
    if nonempty and not result:
        raise StrictSchemaError(f"{path}: empty sequence")
    if len(set(result)) != len(result):
        raise StrictSchemaError(f"{path}: duplicate values")
    return result


def enum(value: Any, choices: set[str] | frozenset[str], path: str) -> str:
    result = string(value, path)
    if result not in choices:
        raise StrictSchemaError(f"{path}: expected one of {sorted(choices)}")
    return result
