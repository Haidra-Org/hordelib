import subprocess
import sys

from loguru import logger

# The shell command you want to run
tests = [
    "run_stress_test_mixed",
    "run_stress_test_cnet_preproc",
    "run_stress_test_cnet",
    "run_stress_test_dynamic",
    "run_stress_test_img2img",
    "run_stress_test_pp",
    "run_stress_test_txt2img_hiresfix",
    "run_stress_test_txt2img",
]

if len(sys.argv) > 2:
    print(f"Usage: {sys.argv[0]} [<iterations>]")
    sys.exit(1)
if len(sys.argv) == 2:
    try:
        ITERATIONS = int(sys.argv[1])
    except ValueError:
        print("Please provide an integer as the argument.")
        sys.exit(1)
else:
    ITERATIONS = 50

# Run the shell command and store the result in a CompletedProcess instance
try:
    for test in tests:
        logger.warning(f"Running stress test: {test}")
        result = subprocess.run(
            f"python -m examples.{test} {ITERATIONS}",
            text=True,
            shell=True,
            check=True,
        )
except subprocess.CalledProcessError as e:
    print(f"Error occurred: {e}")
    print(f"Return code: {e.returncode}")
    print(f"Error message: {e.stderr}")
else:
    print("Tests executed successfully")
    print(f"Return code: {result.returncode}")
    print(f"Output: {result.stdout}")
