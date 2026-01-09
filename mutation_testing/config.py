"""Configuration loading for mutation testing."""

import yaml
from pathlib import Path
from typing import List, Optional, Union
from dataclasses import dataclass

from .core import Mutation


@dataclass
class MutationConfig:
    """Configuration loaded from a YAML file."""

    mutations: List[Mutation]
    module: str
    timeout: int = 30

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "MutationConfig":
        """Load configuration from a YAML file."""
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)

        mutations = []
        module = None
        timeout = data.get("settings", {}).get("timeout", 30)

        for target in data.get("targets", []):
            module = target.get("module")
            for m in target.get("mutations", []):
                mutations.append(Mutation(
                    id=m["id"],
                    function=m["function"],
                    original=m["original"],
                    mutant=m["mutant"],
                    description=m.get("description", ""),
                    line=m.get("line"),
                ))

        if not module:
            raise ValueError("No module specified in config")

        return cls(mutations=mutations, module=module, timeout=timeout)
