import os
from dataclasses import dataclass


@dataclass
class Config:
    cores_available: int


config = Config(
    cores_available=len(os.sched_getaffinity(0)),
)
