class TileModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "seamless_tiling_enabled": ("<bool>",),
            },
        }

    CATEGORY = "spinagon"

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "run"

    def run(self, model, seamless_tiling_enabled: bool = False):
        if seamless_tiling_enabled:
            make_circular(model.model)
        return (model,)


def make_circular(m):
    for child in m.children():
        if "Conv2d" in str(type(child)):
            child.padding_mode = "circular"
        make_circular(child)


def make_regular(m):
    for child in m.children():
        if "Conv2d" in str(type(child)):
            child.padding_mode = "zeros"
        make_regular(child)


NODE_CLASS_MAPPINGS = {"HordeSeamlessTiling": TileModel}
