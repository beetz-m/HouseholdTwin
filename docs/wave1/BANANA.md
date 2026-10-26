# Cavendish banana: research notes, step 2

The model is `food/banana.py`, the hidden-state models are in `food/banana_models.py`, and the register is `params.BANANA`.

## What it is
A banana finger is a parthenocarpic berry from an inferior ovary of **three carpels**. The ovary holds 200–300 ovules on a central placenta; the black dots in the centre are their remnants (`promusa`).

- The **peel** is the exocarp plus the outer mesocarp:
  - a cuticle and a single-layer epidermis;
  - outer ground tissue (chlorenchyma with starch);
  - inner ground tissue with air spaces that grow as the fruit ripens;
  - vascular bundles ringed by latex vessels (laticifers) (`amn12`).
- The **pulp** is the locular, inner pericarp tissue. In the twin it is three carpel segments around the placental axis.
- **Strings** are innermost-mesocarp strands of parenchyma, each with two small vascular bundles. They are not pure phloem (`guapo22`).

## Part tree (USD: /Food/Parts)
- **Peel:** Epidermis, Chlorenchyma, Parenchyma, InnerAerenchyma; Laticifers (points).
- **Stalk** (pedicel): 17.3 × 10.15 mm (`mal22`).
- **CrownCushion.**
- **BlossomTip:** persistent style, blunt tip (`mal22`).
- **Pulp:** Segment_1..3, CentralAxis, Ovules (points), PhloemStrings/String_k.

## Geometry
- **EU minimum:** 14 cm along the convex face from the blossom end to the crown, and 27 mm grade (`eu1333_2011`). The twin's `length_outer` follows this definition.
- **Measured fingers:**
  - Cavendish Valery: outer-curve length 23.3 cm; diameter 34/36/34 mm at 25/50/75 % of the length (`ciro05`).
  - Grand Naine: 19.0 cm; peel 2.56 mm (`mal22`).
  - Mass about 102–148 g; pulp:peel ratio 1.69–2.69 (`chiw23`, `bant24`, `mal22`).
- **Twin check:** banana_0 is 117 g with pulp:peel 2.5.
- **Number of ridges:** GAP. Sources say 'slightly ridged' and 'nearly 4-sided', so the twin uses 4–5 (ILLUSTRATIVE). The peel splits axially at the corners (`khanal23`), so strips follow the ridges.

## Behaviour
- **Ripening:**
  - Seven-stage colour scale (`markon`).
  - At 20 °C with ethylene, stage 1 → 7 takes 7 d (`hou15`). Natural day bands are slower (`ma22`).
  - Chilling below 13 °C (`wflo`).
- **Peel hue** at R2..R7: 98.7 / 97.3 / 92.7 / 89.4 / 87.5 / 82.0° (`ringer21`). Stage 1 is extrapolated (`col20`).
- **Pulp:**
  - E 1.75 → 1.09 MPa in compression and 5.16 → 2.65 MPa in bending; penetration 7.2 → 3.4 N (`ciro05`).
  - Firmness 38 → 14 N (`jaiswal12`).
  - Brix 4.5 → 23.7 (`chiw23`).
- **Peel:**
  - puncture 13.6–15.2 N (`ciro05`);
  - fracture 53 → 2.1 N over ripening (`khanal23`, abstract).
- **Finger drop:** about 49 N on day 1, falling by more than half by day 3. There is no abscission zone; the peel ruptures (`imsabai`, `mbeguie09`).
- **Bruising:** none visible up to 200 mJ on green Grande Naine (`bugaud14`).

## Kitchen actions
- **Peeling from the stem:** snap the neck, then pull the strips down.
- **'Monkey method':** pinch the tip open first.
- **Slicing:** rounds of about 6 mm (3–6 mm).
- **Browning:** lemon juice slows it.
- **Storage:** hang the bunch, keep it apart from ethylene producers, and refrigerate only once ripe (`banana_howto`).
- **Robot peeling** (`kim22`): 40.9 % consecutive success. The neck was too stiff on fresh fruit, so it was pre-cut.

## Gaps
- **Peeling force:** the strip force is ILLUSTRATIVE, so the peel bench inherits it.
- Neck-snap force.
- Ridge, string and strip counts.
- Fresh density.
- Numeric pulp colour.
- Kinetics of cut-surface ΔE.
