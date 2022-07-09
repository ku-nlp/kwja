from pathlib import Path

import pytest

here = Path(__file__).parent


@pytest.fixture()
def fixture_data_dir():
    return here / "data"
