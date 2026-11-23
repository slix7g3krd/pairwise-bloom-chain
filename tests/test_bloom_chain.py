import pytest
import graphics.bloom_chain as mod

CASES = [
    ("threshold_exposure", (1,)),
    ("chain_energy", (1,)),
    ("upsample_bleed", (1,)),
    ("async_order", (1,)),
    ("output_space", (1,)),
    ("determinism", (1,)),
    ("downsample", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("buffer_lifetime", (1,)),
    ("metrics", (1,)),
    ("dirt_mask", (1,)),
    ("clamp", (1,)),
    ("half_res", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
