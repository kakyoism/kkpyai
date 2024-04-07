import os.path as osp
import sys


# project
_script_dir = osp.abspath(osp.dirname(__file__))
sys.path.insert(0, repo_root := osp.abspath(f'{_script_dir}/../../kkpyai'))
import kkpyutil as util
import torch as tc
import kktorch as ktc

_case_dir = _script_dir
_src_dir = osp.abspath(osp.dirname(_case_dir))
_org_dir = osp.join(_case_dir, '_org')
_gen_dir = osp.join(_case_dir, '_gen')
_ref_dir = osp.join(_case_dir, '_ref')
_skip_slow_tests = osp.isfile(osp.join(_case_dir, 'skip_slow_tests.cfg.txt'))
_skip_reason = 'tests requires long network or file i/o are temporarily skipped during tdd'


def test_factory_ramp():
    fact = ktc.TensorFactory()
    got = fact.ramp((1, 3, 3), 1)
    assert tc.allclose(got, tc.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], device=got.device))


def test_factory_rand_repro():
    fact = ktc.TensorFactory()
    got1 = fact.rand_repro((2, 2), 42)
    got2 = fact.rand_repro((2, 2), 42)
    assert tc.allclose(got1, got2)
