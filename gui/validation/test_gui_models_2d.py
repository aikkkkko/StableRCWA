"""Focused interchange and exact geometry checks for the 2D GUI model."""
import json
import unittest

import numpy as np

from gui_models_2d import (build_stack_2d, from_stack_2d, load_rcwa2d,
                           material_slice_2d, normalize_2d, preset_2d, preview_2d)


class TestGUIModels2D(unittest.TestCase):
    def test_preset_roundtrip_and_stack(self):
        cfg = preset_2d()
        self.assertEqual(cfg["mode"], "periodic2d")
        self.assertEqual(cfg["incident"]["M"], 4)
        self.assertEqual(cfg["incident"]["My"], 4)
        self.assertEqual(cfg["sampling"]["slice_at"], .4)
        self.assertEqual(cfg["pml"], {"left": .15, "right": .15, "strength": 3.,
                                       "order": 3, "x0": 0.})
        stack = build_stack_2d(json.loads(json.dumps(cfg)))
        self.assertIsInstance(stack, load_rcwa2d().Stack2D)
        self.assertAlmostEqual(stack.lat.Ly, .8)
        self.assertEqual(len(stack.layers), 3)
        exported = from_stack_2d(stack, name="往返", lam0=1., theta_deg=20.,
                                 phi_deg=30., M=4, My=4, pol="s",
                                 sampling=cfg["sampling"])
        self.assertEqual(exported["structure"], cfg["structure"])
        self.assertEqual(exported["sampling"], cfg["sampling"])

    def test_shifted_boundaries_and_preview_tile(self):
        cfg = preset_2d()
        pillar = cfg["structure"]["layers"][0]
        pillar["y_edges"] = [-.2, .2, .6]
        pillar["x_profiles"] = [
            {"edges": [-.2, .2, .8], "n_values": [1.7 + .02j, 1.]},
            {"edges": [-.2, .8], "n_values": [1.]},
        ]
        cfg = normalize_2d(cfg)
        self.assertEqual(pillar["y_edges"], [-.2, .2, .6])  # input was not mutated
        g = preview_2d(cfg)
        self.assertEqual((g["x0"], g["x1"], g["y0"], g["y1"]), (0., 1., 0., .8))
        self.assertAlmostEqual(g["depth"], .57)
        for layer in g["layers"]:
            rects = layer["rectangles"]
            self.assertAlmostEqual(sum((r["x1"] - r["x0"]) * (r["y1"] - r["y0"])
                                       for r in rects), .8)
            self.assertTrue(all(0 <= r["x0"] < r["x1"] <= 1 and
                                0 <= r["y0"] < r["y1"] <= .8 for r in rects))
            for x in np.linspace(.025, .975, 20):
                for y in np.linspace(.02, .78, 20):
                    self.assertEqual(sum(r["x0"] <= x < r["x1"] and r["y0"] <= y < r["y1"]
                                         for r in rects), 1)
        self.assertEqual(g["layers"][0]["fourier"]["method"], "analytic")
        self.assertEqual(g["layers"][0]["fourier"]["name"], g["layers"][0]["name"])
        self.assertEqual(g["layers"][1]["fourier"]["method"], "uniform")
        expected = (1.7 + .02j) ** 2
        lossy = next(r for r in g["layers"][0]["rectangles"] if r["n"] != 1.)
        self.assertAlmostEqual(lossy["epsilon_real"], expected.real)
        self.assertAlmostEqual(lossy["epsilon_imag"], expected.imag)
        # Offset origins wrap on both axes.
        xz = material_slice_2d(cfg, [0., .3, .9, 1.3], [.1], plane="xz", at=0.)
        np.testing.assert_allclose(xz, [[1.7 + .02j, 1., 1.7 + .02j, 1.]])
        yz = material_slice_2d(cfg, [0., .3, .7], [.1], plane="yz", at=.1)
        np.testing.assert_allclose(yz, [[1.7 + .02j, 1., 1.7 + .02j]])

    def test_material_slice_exteriors_and_uniform(self):
        cfg = preset_2d()
        n = material_slice_2d(cfg, [0., .5, 1.], [-.1, .15, .4, .52, .8],
                              plane="xz", at=.4)
        np.testing.assert_allclose(n, [[1., 1., 1.], [1., 1.7, 1.],
                                       [1.46] * 3, [1.9] * 3, [1.45] * 3])
        n_yz = material_slice_2d(cfg, [0., .4, .7], [.15], plane="yz", at=.5)
        np.testing.assert_allclose(n_yz, [[1., 1.7, 1.]])

    def test_x_only_pattern_and_y_invariance(self):
        cfg = preset_2d()
        cfg["structure"]["layers"][0] = {
            "name": "沿y不变", "kind": "patterned", "thickness": .3,
            "edges": [-.2, .2, .8], "n_values": [1.6, 1.]}
        stack = build_stack_2d(cfg)
        self.assertEqual(stack.layers[0].y_edges, [0., .8])
        np.testing.assert_allclose(material_slice_2d(cfg, [0., .3, .9], [.1], at=-2.),
                                   [[1.6, 1., 1.6]])
        np.testing.assert_allclose(material_slice_2d(cfg, [-.1, .2, .7], [.1], plane="yz", at=.1),
                                   [[1.6] * 3])
        self.assertAlmostEqual(sum((r["x1"] - r["x0"]) * (r["y1"] - r["y0"])
                                   for r in preview_2d(cfg)["layers"][0]["rectangles"]), .8)

    def test_rejects_invalid_configurations(self):
        def rejected(change, match=None):
            cfg = preset_2d()
            change(cfg)
            with self.assertRaisesRegex(ValueError, match or "."):
                normalize_2d(cfg)

        rejected(lambda c: c["structure"].update(period_y=0))
        rejected(lambda c: c["incident"].update(My=31))
        rejected(lambda c: c["incident"].update(M=20, My=20), "625")
        rejected(lambda c: c["structure"].update(n_incident="1+0.1j"), "无损")
        rejected(lambda c: c["structure"]["layers"][0].update(fourier_method="fft"), "FFT")
        rejected(lambda c: c["structure"]["layers"][0].update(fft_samples=256), "FFT")
        rejected(lambda c: c["structure"]["layers"][0].update(y_edges=[0, .2, .5, .7]))
        rejected(lambda c: c["structure"]["layers"][0]["x_profiles"][1].update(edges=[0, .3, .7, .9]))
        rejected(lambda c: c["sampling"].update(slice_plane="xy"))
        rejected(lambda c: c["sampling"].update(slice_at=.81))
        rejected(lambda c: c["sampling"].update(slice_plane="yz", slice_at=1.01))


if __name__ == "__main__":
    unittest.main(verbosity=2)
