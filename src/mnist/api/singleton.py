import typing


T = typing.TypeVar('T')


def singleton(cls: type[T]) -> typing.Callable[..., T]:
    instances: typing.Dict[type[T], T] = dict()

    def get_instance(*args: typing.Any, **kwargs: typing.Any) -> T:
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance
