from dataclasses import make_dataclass, fields, field
from loguru import logger
import numpy as np


class Struct(object):
    def __new__(cls, **kwargs):
        class_fields = []
        for key, val in kwargs.items():
            if isinstance(val, dict):
                class_fields.append([
                    key, type(val), field(
                        default_factory=lambda v=val: v.copy() if hasattr(v, 'copy') else dict(v),
                    )]
                )
            elif isinstance(val, np.ndarray):
                # Handle numpy arrays with default_factory
                class_fields.append([
                    key, type(val), field(
                        default_factory=lambda v=val: v.copy(),
                    )]
                )
            elif isinstance(val, (list, set, tuple)) and len(val) > 0:
                # Handle other mutable collections
                class_fields.append([
                    key, type(val), field(
                        default_factory=lambda v=val: type(v)(v),
                    )]
                )
            else:
                # Handle immutable types normally
                class_fields.append([
                    key, type(val), field(default=val)]
                )

        object_type = make_dataclass(
            'Struct',
            class_fields,
            namespace={
                'keys': lambda self: [f.name for f in fields(self)],
            },
        )
        return object_type()