import hordelib

# Remove any command line args passed to pytest. ComfyUI hates
# the pytest args being in argv, we we hack them out purely for testing.
hordelib.initialise()
