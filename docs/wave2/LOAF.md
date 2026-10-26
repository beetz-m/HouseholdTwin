# Sandwich loaf (Toastbrot): research notes

The model is `food/loaf.py`, the hidden-state models are in `food/loaf_models.py`, and the register is `params.LOAF`. Source keys refer to `params.SOURCES`.

## What it is
Industrial German Toastbrot is baked in a lidded tin for about 30 min, which gives a thin, soft crust and a flat top. It cools in a clean room and is sliced and bagged at once. Most packs contain **no heel slices**; organic packs have two (`test25_toast`). The twin therefore has square, cut ends and keeps the crust only on the tin faces and the top.

## Numbers
- **Pack:** 500 g, 20 slices of about 25 g, about 0.25 m long (`toast_pack`, SOURCED-VERIFY). The twin's slice is about 12.5 mm thick.
- **Twin check:** loaf_0 weighs **504 g**.
- **Crumb:**
  - apparent density 0.19 g/cm³ (`nuss90`);
  - specific volume 4.72 ml/g;
  - moisture 37–39 %;
  - L\*a\*b\* 74.7 / 0.4 / 15.5 (`ishida14`).
- **Firmness:** 1.98, 3.62, 4.18 and 5.64 N at days 1, 5, 9 and 13 (`sehn20`, lab loaf).
- **Tensile failure of a slice:** 0.31–0.39 N on a 12 mm neck (`nuss90`). Compressed bread rarely fails, so tearing happens in tension: pulling, or a knife dragging cold butter.
- **Best-before:** 12–20 days (`test25_toast`). The fridge is said to stale bread about 6× faster (`toast_misc`, a weak popular source).
- **Toasting:** 80 s to more than 150 s to the best toast, depending on the toaster. Higher settings mean longer time, not more heat (`toaster26`). Acrylamide benchmark 50 µg/kg; "go for gold".

## Gaps
- Crust thickness.
- Pan-crumb Young's modulus.
- Friction between slices, and slice bending.
- Toast L\* by stage.
- The steps for opening the bag and separating slices.
