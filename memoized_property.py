import functools


def memoized_property(method):
    "Memoizes the output of method and makes it available as a property"
    n = '_'+method.__name__

    @functools.wraps(method)
    def wrapper(self):
        try:
            v = getattr(self, n)
            return v
        except AttributeError:
            v = method(self)
            setattr(self, n, v)
            return v
    return property(wrapper)
