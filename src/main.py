import asyncio
from viam.module.module import Module
try:
    from models.orbbec import Orbbec
except ModuleNotFoundError:
    # when running as local module with run.sh
    from .models.orbbec import Orbbec


if __name__ == '__main__':
    asyncio.run(Module.run_from_registry())
