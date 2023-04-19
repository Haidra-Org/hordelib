# This tests running hordelib standalone, as an external caller would use it.
# Call with: python -m test.run_txt2img
# You need all the deps in whatever environment you are running this.
import os

import hordelib


def load_and_free():
    from hordelib.shared_model_manager import SharedModelManager

    SharedModelManager.manager.load("Deliberate")
    SharedModelManager.manager.load("Realistic Vision")
    SharedModelManager.manager.load("URPM")

    SharedModelManager.manager.compvis.move_to_disk_cache("URPM")
    print(SharedModelManager.manager.loaded_models)

    SharedModelManager.manager.unload_model("Deliberate")
    SharedModelManager.manager.unload_model("Realistic Vision")
    SharedModelManager.manager.unload_model("URPM")


def main():
    hordelib.initialise(setup_logging=False)

    from hordelib.horde import HordeLib
    from hordelib.shared_model_manager import SharedModelManager

    HordeLib()
    SharedModelManager.loadModelManagers(compvis=True)

    load_and_free()


if __name__ == "__main__":
    main()
