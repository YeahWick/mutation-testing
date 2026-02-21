"""Configuration loading for mutation testing."""

import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field

from .core import Mutation


@dataclass
class CoverageConfig:
    """Coverage report settings from the YAML config."""

    threshold: float = 100.0
    fail_under: bool = False
    test_paths: List[str] = field(default_factory=list)
    test_mappings: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class MutationConfig:
    """Configuration loaded from a YAML file."""

    mutations: List[Mutation]
    module: str
    timeout: int = 30
    coverage: Optional[CoverageConfig] = None

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

        # Parse coverage settings
        coverage = None
        cov_data = data.get("coverage")
        if cov_data:
            coverage = CoverageConfig(
                threshold=float(cov_data.get("threshold", 100.0)),
                fail_under=bool(cov_data.get("fail_under", False)),
                test_paths=cov_data.get("test_paths", []),
                test_mappings={
                    k: v if isinstance(v, list) else [v]
                    for k, v in cov_data.get("test_mappings", {}).items()
                },
            )

        return cls(mutations=mutations, module=module, timeout=timeout, coverage=coverage)
