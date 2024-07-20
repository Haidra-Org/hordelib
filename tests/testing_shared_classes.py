class ResolutionTestCase:
    def __init__(
        self,
        *,
        width: int,
        height: int,
        ddim_steps: int,
        hires_fix_denoise_strength: float,
        model_native_resolution: int,
        max_expected_steps: int | None,
        min_expected_steps: int | None,
    ):
        self.width = int(width)
        self.height = int(height)
        self.ddim_steps = int(ddim_steps)
        self.hires_fix_denoise_strength = hires_fix_denoise_strength
        self.model_native_resolution = int(model_native_resolution)
        self.max_expected_steps = int(max_expected_steps)
        self.min_expected_steps = int(min_expected_steps)
