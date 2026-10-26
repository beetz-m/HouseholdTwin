# Wheat roll (Weizenbrötchen): research notes, step 2

The model is `food/roll.py`, the hidden-state models are in `food/roll_models.py`, and the register is `params.ROLL`. Every number carries its status in the register; source keys refer to `params.SOURCES`.

## What it is
A roll is a baked foam, not a plant organ. It has a dense, browned, glassy **crust** around a **crumb** of gas cells in a gelatinised starch and gluten matrix.

The legal definition comes from the Leitsätze für Brot und Kleingebäck (`leitsaetze_1993`):
- Kleingebäck weighs no more than 250 g.
- A Weizenbrötchen contains at least 90 % wheat flour.

The ordinary oval Weizenbrötchen has one lengthwise cut, the *Ausbund*. That cut raises the crust share (`testde_2021`). The Kaiser roll instead has five star segments (wiki_kaiser).

## Part tree (USD: /Food/Parts)
| Part | Notes | Key numbers |
|---|---|---|
| Crust/UpperCrust | oven-browned, glossy from steam, fine cooling cracks | 1.0–1.5 mm (≈1.3) cut-off crust, SOURCED (`hirte_2012`); definition 1–3 mm |
| Crust/ScoreCrust | the Ausbund: the cut edge torn open by oven spring, paler and rougher | thinner (ILLUSTRATIVE) |
| Crust/BottomCrust | baked on the hearth or tray | thickness GAP |
| Crumb/TransitionZone | compressed, smaller cells under the crust | colour band 'inner crust' L* 51.6–62.3 (`soleimani_2014`) |
| Crumb/CrumbCore | open foam, cells elongated in the rise direction | porosity 0.73–0.81 (µCT); cell walls 20–61 µm (`hirte_2012`); mean 2D cell area 0.32 mm², 83 cells/cm² (`lagrain_2012`) |
| Crumb/GasCells (points) | cell centres and diameters; cut faces are painted from these | – |
| Dusting (points) | flour, variant 'dusted' | – |

## Size and mass
- 90 × 50 mm crispy roll (`hirte_2012`).
- Round buns 86–92 × 50–55 mm, 197–263 cm³ (`musiienko_2025`).
- Rolls 50–90 g (`testde_2021`).
- RMT standard bake: about 55 g dough per roll (`ireks_rmt`).
- Specific volume 6.4 → 5.9 ml/g over 48 h (`kurek_2016`).
- The twin's roll_0 weighs **48.1 g**.

## Behaviour (hidden state → discrete state)
- **Crust softening.** Water migrates from the crumb (aw 0.97) into the crust.
  - Crust water: 9.0 % at 30 min and 13.9 % at 1 d (40 % RH).
  - Critical water content: 10.1 %, reached after 2.1 h, where half of the crispness is lost (`hirte_2012`).
  - Twin fit: w = 9.73·t^0.112 (CALIBRATED, residual +0.5 points at 2.1 h).
- **Crumb staling.**
  - Crumb E ≈ 20 → 70 → 125 → 157 kPa at 5 h, 2 d, 4 d and 7 d; Poisson's ratio 0.103 (`lagrain_2012`, pan bread).
  - Firmness follows Avrami with A0 6.48 N, A∞ 66.01 N, k 0.186/d (`cardone_2020`).
- **Reheating** ('aufbacken'): 150–170 °C for 5–10 min (`utopia_aufbacken`).
- **Toasting:** 3 min light to 4.5 min 'starting to burn' (`kaur_2026`).
- **Colour:**
  - crust L*a*b* 47.6 / 17.05 / 32.2;
  - crumb 72.1 / 0.9 / 15.4 (`chaple_2023`, pan bread);
  - wheat-roll browning index 50.4 (`kurek_2016`).

## Kitchen actions (designators in `food/wave1_tasks.py`)
- **Slicing** (`wikihow_slice`, `tastingtable_bagel`):
  - Use a serrated knife with the roll flat and a palm flat on top.
  - Saw from the middle of the long side with light pressure.
  - Halfway through, stand the roll on its uncut edge and finish downwards.
  - Stop short to leave a hinge.
- **Robot sawing set-point:** −30 N, 2 cm/s, ±2 cm strokes on a baguette (`luo_2026`).
- **Storage:** a paper bag is recommended; plastic softens the crust; freeze while fresh (`oekotest_storage`).

## Gaps
- Serrated-knife cutting force.
- Crust modulus versus water activity.
- Bottom-crust thickness.
- Roll-specific Avrami constants.
- Dough-to-roll baking loss.
- L*a*b* measured on a German roll.

## Robotics
- `luo_2026`: UR5e sawing.
- Hassouna 2019 (Bremen BSc): PR2 bread slicing in simulation.
- Beetz et al. 2024 (AKR3): actionable knowledge graphs for cutting.
