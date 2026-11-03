import asyncio

from viam.module.module import Module

# Imported for its side effect: EasyResource registers the model on import.
try:
    from models.orbbec import Orbbec  # noqa: F401
except ModuleNotFoundError:
    # when running as local module with run.sh
    from .models.orbbec import Orbbec  # noqa: F401


if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())
