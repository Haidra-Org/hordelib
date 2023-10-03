import hordelib.initialisation
from hordelib.shared_model_manager import SharedModelManager


def main():
    hordelib.initialisation.initialise(setup_logging=True, logging_verbosity=5)

    SharedModelManager.load_model_managers(["compvis"])
    SharedModelManager.manager.compvis.download_all_models()


if __name__ == "__main__":
    main()
