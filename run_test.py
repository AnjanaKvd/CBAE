import numpy as np


def test_dt(dt):
    t = 1.0
    P_base = np.array([0.6], dtype=np.float16)

    dx_forward = 0.01 * np.sin(2 * np.pi * 0.1 * (t + dt))
    P_forward = np.clip(P_base + dx_forward, 0.0, 1.0).astype(np.float16)

    dx_backward = 0.01 * np.sin(2 * np.pi * 0.1 * (t - dt))
    P_backward = np.clip(P_base + dx_backward, 0.0, 1.0).astype(np.float16)

    v_num = (P_forward.astype(np.float32) - P_backward.astype(np.float32)) / (2.0 * dt)

    v_analytic = 0.01 * 2 * np.pi * 0.1 * np.cos(2 * np.pi * 0.1 * t)

    err = abs(v_num[0] - v_analytic)
    print(
        f"dt: {dt}, err: {err}, v_num: {v_num[0]}, v_analytic: {v_analytic}, P_f: {P_forward[0]}, P_b: {P_backward[0]}"
    )


for dt in [0.001, 0.01, 0.05, 0.1, 0.15, 0.2]:
    test_dt(dt)
