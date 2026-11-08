"""Typed field declarations for YAML-driven entry points, independent of a CLI."""
from dataclasses import dataclass


class ConfigError(ValueError):
    """An input YAML is missing a field or contains an incompatible value."""


@dataclass(frozen=True)
class ConfigField:
    name: str
    path: str
    type: object = None
    choices: object = None
    nargs: object = None
    action: str = "store"
    nullable: bool = False
    inherit: object = None
    reference: bool = False
    help: str = ""

    def convert(self, value):
        if value is None:
            if self.nullable:
                return None
            raise ConfigError(f"{self.path}: null is not allowed")
        if self.action == "append":
            if not isinstance(value, list):
                raise ConfigError(f"{self.path}: expected a YAML list")
            return [self._occurrence(item) for item in value]
        return self._occurrence(value)

    def _occurrence(self, value):
        if self.action == "store_true":
            if not isinstance(value, bool):
                raise ConfigError(f"{self.path}: expected true or false")
            return value
        if self.nargs is not None:
            if not isinstance(value, list):
                raise ConfigError(f"{self.path}: expected a YAML list")
            if self.nargs == "+" and not value:
                raise ConfigError(f"{self.path}: expected at least one value")
            if isinstance(self.nargs, int) and len(value) != self.nargs:
                raise ConfigError(f"{self.path}: expected {self.nargs} values")
            return [self._scalar(item) for item in value]
        return self._scalar(value)

    def _scalar(self, value):
        if self.type is int and (isinstance(value, bool) or not isinstance(value, int)):
            raise ConfigError(f"{self.path}: expected an integer")
        if self.type is float and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ConfigError(f"{self.path}: expected a number")
        if self.type is str and not isinstance(value, str):
            raise ConfigError(f"{self.path}: expected a string")
        try:
            result = self.type(value) if self.type is not None else value
        except (TypeError, ValueError) as error:
            raise ConfigError(f"{self.path}: {error}") from error
        if self.choices is not None and result not in self.choices:
            raise ConfigError(f"{self.path}: expected one of {list(self.choices)}, got {result!r}")
        return result


class ConfigSchema:
    """Declare only the YAML fields consumed by an entry point.

    Numerical defaults belong in YAML, not in this schema. ``inherit`` can name
    a field in a referenced GB-SAE recipe; no model values are copied into jobs.
    Other fields read directly by an experiment remain in its configuration.
    """

    def __init__(self, prog=None, *, description=None, **_):
        self.prog = prog
        self.description = description
        self.fields = []

    def add_field(self, name, **options):
        if any(field.name == name for field in self.fields):
            raise ValueError(f"Duplicate configuration field: {name}")
        field = ConfigField(name, **options)
        self.fields.append(field)
        return field

    def add_group(self, _description):
        """Allow reusable field groups to append to the same explicit schema."""
        return self

    def error(self, message):
        raise ConfigError(message)
