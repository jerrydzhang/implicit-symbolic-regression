def __getattr__(name: str):
    if name == "gplearn":
        from . import gplearn

        return gplearn

    raise AttributeError(f"module {__name__} has no attribute {name}")
