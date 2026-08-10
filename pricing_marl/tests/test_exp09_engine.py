"""
Validation for the exp09 Calvano-ladder engine (src_calvano/).

Runnable directly (`python tests/test_exp09_engine.py`) or under pytest.

The checks that matter for the ladder's internal consistency:

  1. full-vector state encode/decode round-trips.
  2. C5 (market_min + rules, K=30) block dynamics reproduce src_atomic
     exactly -- the engine is the same physics, only the Q-table indexing
     differs.
  3. C4/C5 Q-init reproduces src.agent.calculate_heuristic_init_values, i.e.
     exp09's rule cells carry the same prior as exp05/exp07.
  4. C1/C2 Q-init equals Calvano Eq. (8) computed independently, and is
     constant across states.
  5. The reachable-vector count behind the "C3 is not degenerate" claim in
     Extention_Plan.md 4.3 (580 of 1000 under the 3-rule set).
  6. A short end-to-end run of every cell produces a well-formed eval frame.
"""

import itertools
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.agent import calculate_heuristic_init_values
from src.config import Config
from src.environment import PricingEnvironment, get_demand_and_profit_static
from src.strategies import ACT_ABOVE, ACT_MATCH, ACT_UNDERCUT
from src_atomic.config import AtomicConfig
from src_atomic.environment import AtomicEnvironment
from src_calvano.agent import heuristic_init
from src_calvano.config import CalvanoConfig
from src_calvano.environment import CalvanoEnvironment
from src_calvano.simulation import run_simulation

RULES = [ACT_UNDERCUT, ACT_MATCH, ACT_ABOVE]
MU = 0.25


def _cfg(**kw):
    base = dict(num_sellers=3, num_grids=10, mu=MU, active_strategies=list(RULES))
    base.update(kw)
    return CalvanoConfig(**base)


def _env_for(**kw):
    cfg = _cfg(**kw)
    return cfg, CalvanoEnvironment(cfg)


def test_state_encoding_roundtrip():
    cfg = _cfg(state_mode="full_vector", action_mode="price", K=1)
    env = CalvanoEnvironment(cfg)
    seen = set()
    for vec in itertools.product(range(cfg.num_grids), repeat=cfg.num_sellers):
        s = env.encode(list(vec))
        assert 0 <= s < cfg.num_states
        assert env.decode(s) == list(vec), f"roundtrip failed for {vec}"
        seen.add(s)
    assert len(seen) == cfg.num_states, "encoding is not a bijection"

    cfg_min = _cfg(state_mode="market_min", action_mode="rule", K=1)
    env_min = CalvanoEnvironment(cfg_min)
    assert env_min.encode([4, 2, 7]) == 2, "market_min must include own price"
    assert env_min.encode([9, 9, 3]) == 3
    print("  [ok] state encoding round-trips; market_min includes own price")


def test_block_dynamics_match_atomic():
    """C5 physics must equal src_atomic's, for every (vector, rule profile)."""
    cfg = _cfg(state_mode="market_min", action_mode="rule", K=30)
    env = CalvanoEnvironment(cfg)

    atomic_cfg = AtomicConfig(
        num_sellers=3, num_grids=10, active_strategies=list(RULES), mu=MU, K=30,
    )
    atomic_env = AtomicEnvironment(atomic_cfg)

    max_reward_diff = 0.0
    checked = 0
    for vec in itertools.product(range(cfg.num_grids), repeat=cfg.num_sellers):
        for profile in itertools.product(range(len(RULES)), repeat=cfg.num_sellers):
            r_new, next_new = env.run_block(list(vec), list(profile), 30)
            r_ref, next_ref = atomic_env.run_block(list(vec), list(profile), 30)
            assert next_new == next_ref, f"next state differs at {vec}/{profile}"
            max_reward_diff = max(max_reward_diff, float(np.max(np.abs(r_new - r_ref))))
            checked += 1
    assert max_reward_diff < 1e-12, f"reward mismatch {max_reward_diff}"
    print(f"  [ok] block dynamics match src_atomic on all {checked} (vector, profile) pairs "
          f"(max reward diff {max_reward_diff:.2e})")


def _reference_lookup_init(K):
    base_cfg = Config(
        num_sellers=3, num_grids=10, active_strategies=list(RULES), mu=MU, K=K,
    )
    base_env = PricingEnvironment(base_cfg)
    return calculate_heuristic_init_values(base_env, base_cfg)


def test_rule_init_matches_lookup_heuristic():
    """C4 (K=1) and C5 (K=30) priors must equal the exp05/exp07 heuristic."""
    for K in (1, 30):
        cfg = _cfg(state_mode="market_min", action_mode="rule", K=K)
        env = CalvanoEnvironment(cfg)
        q_new = heuristic_init(cfg, env)
        q_ref = _reference_lookup_init(K)
        assert q_new.shape == (cfg.num_sellers,) + q_ref.shape, f"bad shape {q_new.shape}"
        diff = float(np.max(np.abs(q_new - q_ref[None, :, :])))
        assert diff < 1e-9, f"K={K}: init differs from lookup heuristic by {diff}"
        print(f"  [ok] K={K} market_min rule init == calculate_heuristic_init_values "
              f"(max diff {diff:.2e})")


def test_market_min_init_is_position_independent():
    """
    Under state_mode="market_min" the representative vector (o,...,o) is
    symmetric, so per-seller and pooled initialization must coincide exactly.
    This is what keeps C4/C5 identical to the exp05/exp07 prior regardless of
    the init_pooling setting.
    """
    for K in (1, 30):
        per = heuristic_init(*_env_for(state_mode="market_min", action_mode="rule",
                                       K=K, init_pooling="per_seller"))
        pooled = heuristic_init(*_env_for(state_mode="market_min", action_mode="rule",
                                          K=K, init_pooling="pooled"))
        # Agreement is exact in exact arithmetic; the residual is float
        # summation order, which differs across seller accumulators.
        diff = float(np.max(np.abs(per - pooled)))
        spread = float(np.max(np.abs(per - per[0][None])))
        flips = int((per.argmax(axis=2) != pooled.argmax(axis=2)).sum())
        assert diff < 1e-12, f"K={K}: market_min init differs by {diff}"
        assert spread < 1e-12, f"K={K}: seller slices differ by {spread}"
        assert flips == 0, f"K={K}: {flips} argmax flips between per-seller and pooled"
        print(f"  [ok] K={K} market_min: per-seller == pooled "
              f"(max diff {diff:.1e}, 0 argmax flips)")


def test_full_vector_rule_init_is_position_specific():
    """
    Under state_mode="full_vector" with rule actions, sellers at different
    positions face different lowest-competitor prices, so the per-seller prior
    must genuinely differ across sellers -- and must differ from the pooled
    one. Pins the worked example from the appendix discussion, state (0,5,9).
    """
    cfg, env = _env_for(state_mode="full_vector", action_mode="rule", K=1,
                        init_pooling="per_seller")
    per = heuristic_init(cfg, env)
    pooled = heuristic_init(*_env_for(state_mode="full_vector", action_mode="rule",
                                      K=1, init_pooling="pooled"))
    assert np.max(np.abs(per - pooled)) > 1e-3, "per-seller and pooled should differ here"

    s = env.encode([0, 5, 9])
    scale = 1.0 - cfg.gamma
    seller0 = per[0, s] * scale   # seller at index 0, others_min = 5
    seller1 = per[1, s] * scale   # seller at index 5, others_min = 0
    assert np.argmax(seller0) == 0, "seller holding the minimum should prefer Undercut"
    assert np.argmax(seller1) == 2, "seller above the minimum should prefer Above"
    assert np.argmax(pooled[0, s]) == 0, "pooled prior prefers Undercut for everyone"

    # Scope: how many states have at least one seller disagreeing with pooled.
    argmax_per = per.argmax(axis=2)                     # (N, S)
    argmax_pooled = pooled[0].argmax(axis=1)[None, :]   # (1, S)
    states_affected = int((argmax_per != argmax_pooled).any(axis=0).sum())
    print(f"  [ok] full_vector rule init is position-specific; state (0,5,9) gives "
          f"Undercut/Above/Above as expected")
    print(f"       pooled prior would flip the initial argmax for >=1 seller at "
          f"{states_affected}/{cfg.num_states} states")


def test_price_init_is_calvano_eq8():
    """C1/C2 prior == discounted payoff vs uniformly randomizing opponents."""
    cfg = _cfg(state_mode="full_vector", action_mode="price", K=1)
    env = CalvanoEnvironment(cfg)
    q = heuristic_init(cfg, env)

    m, n = cfg.num_grids, cfg.num_sellers
    grid = env.price_grid
    expected = np.zeros(m)
    for own in range(m):
        tot = 0.0
        combos = list(itertools.product(range(m), repeat=n - 1))
        for rivals in combos:
            prices = [grid[own]] + [grid[r] for r in rivals]
            tot += get_demand_and_profit_static(prices, cfg.a_val, cfg.mu, cfg.a0, cfg.c_val)[0]
        expected[own] = tot / len(combos)
    expected = expected / (1.0 - cfg.gamma)

    assert q.shape == (n, cfg.num_states, cfg.num_actions)
    diff = float(np.max(np.abs(q[0, 0] - expected)))
    assert diff < 1e-9, f"Calvano Eq. (8) mismatch: {diff}"
    assert np.allclose(q, q[0, 0]), \
        "price-mode init must be constant across states and sellers"
    print(f"  [ok] price init == Calvano Eq. (8), constant across states "
          f"(max diff {diff:.2e})")


def test_reachable_vector_count():
    cfg = _cfg(state_mode="full_vector", action_mode="rule", K=1)
    env = CalvanoEnvironment(cfg)
    reach = env.reachable_vectors()
    assert len(reach) == 580, f"expected 580 reachable vectors, got {len(reach)}"
    assert len({min(v) for v in reach}) == 10
    print(f"  [ok] {len(reach)}/1000 vectors reachable under the 3-rule map "
          f"(matches Extention_Plan.md 4.3)")


def test_end_to_end_all_cells():
    """Every rung runs and produces a well-formed evaluation frame."""
    cells = {
        "cal_full":     dict(state_mode="full_vector", action_mode="price", K=1),
        "cal_smin":     dict(state_mode="market_min",  action_mode="price", K=1),
        "cal_arule":    dict(state_mode="full_vector", action_mode="rule",  K=1),
        "cal_both":     dict(state_mode="market_min",  action_mode="rule",  K=1),
        "cal_both_k30": dict(state_mode="market_min",  action_mode="rule",  K=30),
    }
    for name, spec in cells.items():
        cfg = _cfg(max_episodes=3000, converge_period=500, eval_H=300, **spec)
        assert cfg.cell_tag == name, f"cell_tag {cfg.cell_tag} != {name}"
        np.random.seed(0)
        df, qdf = run_simulation(cfg, run_id=0, disable_tqdm=True, return_q_snapshot=True)

        expected_rows = max(1, cfg.eval_H // cfg.K) * cfg.K
        assert len(df) == expected_rows, f"{name}: {len(df)} rows, expected {expected_rows}"
        for col in ("delta", "price_min", "price_mean", "a_0", "p_0", "pi_0"):
            assert col in df.columns, f"{name}: missing column {col}"
        assert df["delta"].notna().all(), f"{name}: NaN in delta"
        assert len(qdf) == 2 * 3 * cfg.num_states * cfg.num_actions, f"{name}: bad qtable size"

        act_ids = set(np.unique(df[[f"a_{i}" for i in range(3)]].values))
        if cfg.action_mode == "price":
            assert act_ids <= set(range(cfg.num_grids)), f"{name}: bad price action ids"
        else:
            assert act_ids <= set(RULES), f"{name}: bad rule action ids"

        print(f"  [ok] {name}: {len(df)} eval rows, mean delta {df['delta'].mean():+.3f}, "
              f"|S|={cfg.num_states} |A|={cfg.num_actions}")


if __name__ == "__main__":
    checks = [
        ("state encoding", test_state_encoding_roundtrip),
        ("block dynamics vs src_atomic", test_block_dynamics_match_atomic),
        ("rule Q-init vs lookup heuristic", test_rule_init_matches_lookup_heuristic),
        ("market_min init position-independent", test_market_min_init_is_position_independent),
        ("full_vector init position-specific", test_full_vector_rule_init_is_position_specific),
        ("price Q-init vs Calvano Eq. (8)", test_price_init_is_calvano_eq8),
        ("reachable vectors", test_reachable_vector_count),
        ("end-to-end all cells", test_end_to_end_all_cells),
    ]
    for label, fn in checks:
        print(f"\n== {label} ==")
        fn()
    print("\nAll exp09 engine checks passed.")
