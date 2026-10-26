# Sweet orange: research notes

The model is `food/orange.py`, the hidden-state models are in `food/orange_models.py`, and the register is `params.ORANGE`.

## Anatomy
The fruit is a hesperidium (`sadka19`):
- the flavedo, the coloured outer peel with oil glands;
- the albedo, the white spongy layer;
- the endocarp, the membrane around each segment;
- the juice vesicles, on stalks growing from the endocarp;
- 3 vascular bundles per segment;
- the central axis (core).

Segment count is 9.6–11 (`citrus_patents`). Seeds: 1.2–6.6 per Valencia fruit; the Washington navel is essentially seedless (`ucr_citrus`). The navel is a rudimentary secondary fruit at the blossom end (`edis891`).

## Numbers
- **Washington navel** (`ucr_citrus`): 162–335 g, 61–88 mm, peel 4.1–6.4 mm, juice 45–54 %, °Brix 9.3–14.1.
- **Twin check:** orange_0 (a Valencia) weighs **193.6 g**.
- **Peel mechanics:**
  - flavedo modulus about 103 MPa, albedo 0.79 MPa;
  - oil jets at 8.5 m/s, 102 µm wide;
  - glands burst at 30–140 kPa (`smith18`);
  - peel density 746 kg/m³ (`jentzsch22`).
- **Peeling:** navel is medium-easy; Valencia is hard and its strips break; cold fruit is harder (`ritenour13`). No force values were published.
- **Juice:** 60–80 mL per orange by hand; specific gravity 1.045; viscosity about 5 mPa·s (`orange_juice`). Navel juice turns bitter within hours (`edis891`).
- **Twin juice yield:** 63 mL by hand.

## Gaps
- L\*a\*b\* values: the twin's peel colour is chosen to hit the market colour-index target of +10 to +20 (`cci`).
- Size of the juice vesicles.
- Cutting force and segment-membrane strength.
- Peeling force in newtons.
