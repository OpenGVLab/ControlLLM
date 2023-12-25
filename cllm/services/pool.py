from collections import OrderedDict
from functools import wraps


class ModelPool:
    def __init__(self):
        self.pool = OrderedDict()
        self.device_map = {}

    def register(self, model_fn):
        @wraps(model_fn)
        def wrapper(func):
            @wraps(func)
            async def innner_wrapper(*args, **kwargs):
                while True:
                    try:
                        model = self._load_model(model_fn)
                        func.model = model
                        return await func(*args, **kwargs)
                    except RuntimeError as e:
                        self._move_oldest_to_cpu(e)
                        model = self._load_model(model_fn)
                        func.model = model
            return innner_wrapper
        return wrapper

    def _load_model(self, model_fn):
        if model_fn not in self.pool:
            while True:
                try:
                    self.pool[model_fn] = model_fn()
                    break
                except RuntimeError as e:
                    self._move_oldest_to_cpu(e)

        model = self.pool[model_fn]
        self.pool.move_to_end(model_fn)

        while True:
            try:
                model.to(model.device)
                break
            except RuntimeError as e:
                self._move_oldest_to_cpu(e)

        return model

    def _move_oldest_to_cpu(self, error):
        remove_at_least_one = False

        for model in self.pool.values():
            if str(model.device) != 'cpu':
                model.to('cpu')
                remove_at_least_one = True
                break

        if not remove_at_least_one:
            raise error
