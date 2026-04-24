from pathlib import Path
import json
import platform
import sys
try:
    import pkg_resources
except ImportError:
    pkg_resources = None


def generate_reproducibility_package(
    output_dir: Path,
):
    """
    Generate reproducibility information for the experiment.
    Saves environment information, Python version, platform,
    and installed packages.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
    }

    env_file = output_dir / "environment.json"

    with open(env_file, "w") as f:
        json.dump(info, f, indent=2)

    # Save installed packages

    if pkg_resources is not None:
        packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}

    packages_file = output_dir / "requirements_snapshot.json"

    with open(packages_file, "w") as f:
        json.dump(packages, f, indent=2)