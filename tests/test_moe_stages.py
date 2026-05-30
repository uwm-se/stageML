from stageml.moe_stages import ADAPTER, BASE, REQUEST, ROUTING, TENANT, TOKEN, join, stage_name, can_fold_at


def test_moe_lattice_join_order():
    assert join(BASE, ADAPTER) == ADAPTER
    assert join(TENANT, REQUEST) == REQUEST
    assert join(ROUTING, TOKEN) == TOKEN
    assert join(BASE, TOKEN, ADAPTER) == TOKEN


def test_stage_names():
    assert stage_name(BASE) == "base"
    assert stage_name("request") == "request"


def test_fold_boundary():
    assert can_fold_at(BASE, ADAPTER)
    assert can_fold_at(ADAPTER, ADAPTER)
    assert not can_fold_at(REQUEST, ADAPTER)
