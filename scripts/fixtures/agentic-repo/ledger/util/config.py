"""Configuration from a TOML-like ``key = value`` file with sections."""
from __future__ import annotations

from dataclasses import dataclass, field


class ConfigError(ValueError):
    pass


@dataclass
class Config:
    sections: dict[str, dict[str, str]] = field(default_factory=dict)

    @classmethod
    def parse(cls, text: str) -> "Config":
        sections: dict[str, dict[str, str]] = {"": {}}
        current = ""
        for number, raw in enumerate(text.splitlines(), start=1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                current = line[1:-1].strip()
                sections.setdefault(current, {})
                continue
            if "=" not in line:
                raise ConfigError(f"line {number}: expected key = value")
            key, value = (part.strip() for part in line.split("=", 1))
            sections[current][key] = value.strip('"')
        return cls(sections)

    def get(self, section: str, key: str, default: str | None = None) -> str | None:
        return self.sections.get(section, {}).get(key, default)

    def integer(self, section: str, key: str, default: int = 0) -> int:
        value = self.get(section, key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            raise ConfigError(f"[{section}] {key} is not an integer: {value!r}") from None
