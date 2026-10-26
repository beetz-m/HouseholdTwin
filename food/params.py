"""Parameter register for the food twins: every number the models use, with unit, value/distribution, status, source.

Status (the D016 rule of the lab twin carried over: no invented physical parameters presented as facts):
  SOURCED         value or range taken from the cited source
  SOURCED-VERIFY  value from a source whose details (species, cultivar, protocol) still need checking in the full text
  STANDARD        a regulation / grading standard (definitional, not a measurement)
  DERIVED         computed from other register entries by a stated physical model
  TYPICAL         common textbook value, no citation held in this repository yet
  CALIBRATED      free model constant fitted to a cited observation (the fit and its residual are stated)
  ILLUSTRATIVE    twin assumption chosen to be plausible; must be calibrated before any quantitative claim

Distributions are what the hidden-world sampler draws per object instance; the robot's SDT starts from the same
distribution as its prior and only learns the instance's values through observations.
"""

SOURCES = {
    "narushin2021": "Narushin V.G., Romanov M.N., Griffin D.K. (2021) Egg and math: introducing a universal formula "
                    "for egg shape. Ann. N.Y. Acad. Sci. 1505:169-177 (Hügelschäffer egg model).",
    "eu589_2008": "Commission Regulation (EC) No 589/2008, marketing standards for eggs: weight grades S<53, "
                  "M 53-63, L 63-73, XL>=73 g; Class A air space height <=6 mm, 'extra' <=4 mm.",
    "icf11_eggshell": "ICF11 conference paper on eggshell fracture, gruppofrattura.it/ocs/index.php/ICF/ICF11/paper/"
                      "download/10244/9618: failure at 34+-6 N (equatorial compression) and 41+-8 N (axial, across "
                      "the poles). Species to verify in the full text.",
    "nedomova2004": "Hen-egg quasi-static compression, Acta Univ. Agric. Silvic. Mendel. Brun. 2004 (actavia.mendelu.cz/"
                    "pdfs/acu/2004/05/07.pdf): model with shell thickness 0.35 mm and Young's modulus 15,000 MPa.",
    "reissner1946": "Reissner E. (1946) Stresses and small displacements of shallow spherical shells. J. Math. Phys. "
                    "25: point-load stiffness of a shallow spherical shell k = 4 E t^2 / (R sqrt(3(1-nu^2))).",
    "apple_E_quasistatic": "Quasi-static equivalent Young's modulus of apple flesh: Jonagold 4.24+-0.96, Joly Red "
                           "5.09+-1.27, Kanzi 7.82+-0.41 MPa (impact study, found via researchportal.be/en/node/4347597; "
                           "citation to verify).",
    "tjaf2005": "Turkish J. Agric. For. 29(6) 2005 (tar-29-6-6-0411-8): apple flesh modulus falls with storage, "
                "1.68->0.51 MPa (Williams), 1.45->0.88 (Starkspur Golden Delicious), 1.51->1.1 (Starking).",
    "fsct2008": "J. Food Sci. Technol. (Iran) 5(18):37-44 (2008): bruise volume at 410 and 1210 mJ impacts; Golden "
                "Delicious about 1 % of fruit volume at 410 mJ; bruise volume rises with impact energy.",
    "hertz": "Hertz contact theory (Johnson, Contact Mechanics, 1985): sphere-on-flat force, pressure, impact.",
    "avo_au_stages": "Avocados Australia, 'Avocado Stages of Ripeness chart for Hass' (2017): stages 1 (hard) .. 5 "
                     "(ripe/eat now); penetrometer (11.1 mm tip, 8 mm into skinless flesh) >10 kgf at stage 1, "
                     "0.4-0.6 kgf at 'eat now'; firmness is a better ripeness index than skin colour.",
    "avo_firmness_12d": "Hass flesh firmness falling from 75.43 to 2.63 N over 12 days of ripening (reported in a "
                        "ripening study surfaced by literature search; citation to verify).",
    "wiki_apple": "https://en.wikipedia.org/wiki/Apple and https://en.wikipedia.org/wiki/Pome: flesh from the "
                  "receptacle/hypanthium; usually 5 carpels with 1-2 seeds; endocarp = leathery core case.",
    "wiki_gala": "https://en.wikipedia.org/wiki/Gala_(apple): 70-80 mm, striped orange-red over yellow, density 0.86.",
    "konarska2006": "Hort. Sci. (Prague) 2006 (agriculturejournals.cz/pdfs/hor/2006/03/04.pdf): apple skin 33.3-73.1 um.",
    "ishs_1099_97": "ISHS Acta Hortic. 1099_97: lenticel density 4.3-12.6 per cm2, mean 7.11.",
    "frontiers2015": "Front. Plant Sci. 2015, 6:679 (fpls.2015.00679): porosity 10.6 % centre .. 26.4 % periphery; "
                     "10 primary bundles (5 sepal + 5 petal), carpellary 10 ventral + 5 dorsal bundles.",
    "uspto": "USPTO apple plant patents PP29089, PP20028, PP21710, PP23862, PP26766: core 24.7-36.7 mm, core "
             "length 41.1 mm, locules 15.2 x 7.9 / 16.6 x 9.9 mm, stems 10.5-28.2 x 2.7-3 mm, stem cavity 8-25 mm deep, "
             "calyx basin 3-17 mm deep.",
    "sciRep2021": "Sci. Rep. 2021 (s41598-021-02143-z): apple seed length 7.69-9.43 mm, width 4.0-4.9 mm.",
    "msu2025": "MSU Extension apple maturity report 2025: Gala ~18 lb Magness-Taylor at harvest.",
    "fmach_porosity": "openpub.fmach.it 10449/23207: apple porosity 17 % (Jazz) .. 29.8 % (Golden Delicious).",
    "jets_skin_tensile": "J. Eng. Technol. Sci. (journals.itb.ac.id/index.php/jets/article/view/1231): skin 2.56 MPa.",
    "pmc7022590": "PMC7022590: Golden Delicious slices browning dE 2.38 / 2.68 / 4.05 at 0 / 30 / 60 min.",
    "atk_cut_apple": "America's Test Kitchen 'How to cut an apple without coring': four cuts beside the core, slices "
                     "6 mm (matchsticks) / 13 mm (dice).",
    "usda_egg": "USDA AMS Egg-Grading Manual (ams.usda.gov/.../Egg Grading Manual.pdf): cuticle (not 'bloom'), "
                "palisade/'spongy' and mammillary layers, two membranes, air cell grades AA <=3.2 mm, A <=4.8 mm; "
                "albumen = outer thin 23 %, thick 57 %, inner thin 17 %, chalaziferous 3 %; yolk ~31 %; check vs leaker.",
    "fr": "Front. Immunol. 2022, 13:838525: cuticle 0.5-12 um, vertical crystal 3-8 um, palisade ~300 um, "
          "mammillary ~100 um, membranes ~70 um; pores ~2x denser at the blunt pole.",
    "ir": "ScienceDirect S2666469024000101: cuticle 5.6-12.5 um, mean 7.85 um (search snippet).",
    "por": "Rev. Bras. Cienc. Avic. (scielo ZBg3hyp6Swr9HgR9XSQSQsN): 10,000-20,000 pores (snippet).",
    "ubc": "UBC open collections 1.0104002: outer shell membrane ~3x thicker than inner.",
    "kop": "J. Process. Energy Agric. (aseestant.ceon.rs 32280): 58.05 x 45.46 mm, SI 78.25, 65.89 g, shell 12.76 %; "
           "compression 32.41 N mean, Hisex Brown 36.43 N short axis / 34.92 N long axis, range 10.03-58.96 N.",
    "alt": "Effect of egg shape index on mechanical properties (unis.gop.edu.tr): SI classes sharp <72, normal "
           "72-76, round >76; highest breaking force along the polar axis at the sharp end.",
    "br": "Redalyc 395940112007: shell 8-11 %, albumen 56-61 %, yolk 27-32 % of egg mass.",
    "sev": "Acta Univ. Agric. Mendel. 2010/01/19: eggshell E 41+-3.9 GPa radial, 43+-4.1 GPa tangential.",
    "sb": "Acta Univ. Agric. Mendel. 2004/05/07: single egg failed ~49 N, linear to failure, shell 0.35 mm, E 15 GPa.",
    "gr": "Grashorn, Lohmann Information 49 (2016) egg storage: air cell < 4 mm after 28 d at 6 C; > 6 mm after "
          "25 d at 22 C; mass loss 1 / 2.58 / 4.54 g at 6 / 15 / 22 C over 28 d.",
    "ro": "Analele Univ. Oradea 2011 (Usturoi): fresh egg specific gravity 1.078-1.097; 1.040-1.059 after >30 d.",
    "pan2026": "Pan Y., Gao S., Fu Y., Obianwuna U.E., Qiu K., Wang J., Zhang H., Qi G., Wu S., Zhou J. (2026) Storage "
               "stability of yolk pigmentation, lipid oxidative stability, and physicochemical quality in lutein-"
               "enriched eggs under refrigerated and ambient conditions. Poultry Science 105(11):107499, "
               "doi:10.1016/j.psj.2026.107499 (PMC13521001). Control eggs, 4 C and 25 C, days 0-30, n = 6: weight "
               "loss, yolk index, albumen height, Haugh unit, albumen pH.",
    "usda_std": "USDA AMS United States Standards, Grades and Weight Classes for Shell Eggs (AMS 56): air cell depth "
                "AA <= 1/8 in (3.2 mm), A <= 3/16 in (4.8 mm), B over 3/16 in (also 7 CFR 56).",
    "eu2023_2465": "Commission Delegated Regulation (EU) 2023/2465 (marketing standards for eggs; repeals 589/2008). "
                   "589/2008: date of minimum durability not more than 28 days after laying (to verify that 2023/2465 "
                   "keeps it).",
    "peel_ph": "The Conversation (2025, UniSQ) on egg peeling, citing a 1960s study: easy peeling needs albumen pH "
               "8.7-8.9; storage before boiling improves peelability; US patent 5053238: fresh albumen ~pH 7.5, rises "
               "to 8.5-9.8 with CO2 loss.",
    "stadelman_ph": "Albumen pH in a newly laid egg 7.6-8.5, rising with CO2 loss through the pores at a temperature-"
                    "dependent rate to ~9.7 (egg-science review via unict / unibo; Sharp & Powell 1931: 9.18 after "
                    "3 d at 3 C; Heath 1977: oiled shells keep pH 8.3 for 7 d at 22 C).",
    "ab": "Alberta Agriculture: infertile blastodisc 2-3 mm, fertile blastoderm 5-9 mm (snippet).",
    "unsw": "A Laboratory Manual of Vertebrate Embryology (1947), UNSW embryology: concentric yolk layers, latebra, "
            "neck of latebra to the nucleus of Pander under the germinal disc; two twisted chalazae.",
    "eng": "Engormix 'Protoporphyrin in shell and cuticle': 13-20 % of brown pigment in the cuticle, ~80 % in shell.",
    "dsm": "DSM YolkFan (formerly Roche): 15 shades since 1965, 16th added 2016.",
    "cohen2025": "Cohen et al. 2025, Communications Physics (MIT; physics.aps.org/articles/v18/100): 180 drops onto a "
                 "hard surface from 8-10 mm: side-on 5 % (8 mm) / ~45 % (10 mm) cracked, vertical 55 % / ~80 %; static "
                 "breaking ~45 N in both orientations.",
    "bps1976": "British Poultry Science 1976 (doi 10.1080/00071667608416318): some well-formed eggs crack from drops "
               "of 3.3 mm onto a stiff flat surface; typically a single hair crack round the equator or to a pole.",
    "alphafoodie_boil": "alphafoodie.com/how-long-to-boil-eggs: hot start, medium eggs: 4.5 min white just set/yolk "
                        "very runny, 6 jammy, 7 barely runny, 8-9 set but creamy, 10-11 hard, >14 overcooked.",
    "kenji_boil": "Lopez-Alt (NYT 2019, via summaries): fridge eggs into boiling water: 6 min liquid yolk, 8.5 fudgy, "
                  "11 just firm; hot start peels cleanly ~90 % vs ~50 % cold start.",
    "egginfo_crack": "British Lion egginfo.co.uk 'How to crack an egg': rounded corner (bowl or pan edge), light tap; "
                     "too hard on a flat surface and you get egg everywhere.",
    "atk_egg_on_egg": "America's Test Kitchen 'The best surface for cracking an egg is another egg': only the weaker "
                      "egg cracks, leaving a clean divot; the membrane holds the shards.",
    "atk_egg": "America's Test Kitchen 'Egg cracking showdown: bowl vs counter': rim tears the membrane and splits "
               "cleanly but leaves shards that can puncture the yolk; counter gives larger pieces, membrane often intact.",
    "wh_separate": "wikiHow 'Separate an Egg': shell-to-shell transfer ~3x or hand; cold eggs; third bowl per egg.",
    "cs42": "Cummings & Schroeder 1942, Calif. Avocado Soc. Yearbook 27:56-64 (avocadosource CAS_1942_PG_056-064): "
            "exocarp (cuticle, epidermis, 1-3 hypodermal cells, parenchyma, stone-cell layer), mesocarp with oil "
            "idioblasts and 6 vascular strands, thin endocarp, two-layer papery testa, two cotyledons, embryonic axis.",
    "uspto_pp14239": "USPTO PP14239 (Hass comparison): skin 1.6 mm, fruit 204 g, seed 28.4 g, seed:skin:flesh 14:13:73.",
    "clark2007": "Clark et al. 2007 Postharvest Biol. Technol. 46:119 (phtnet mb409): fruit density 970-1018 kg/m3.",
    "landahl": "Landahl & Terry (Cranfield): Hass whole-fruit compression 304.1 -> 2.1 N over 6 d at 18 C.",
    "ashton2006": "Ashton et al. 2006 J. Agric. Food Chem. 54:10151: Hass skin L*/C*/h over ripening; flesh in three "
                  "equal bands dark green / pale green / yellow; cyanidin-3-O-glucoside in ripe skin.",
    "farinas2021": "Farinas et al. 2021 J. Food Eng. 289:110266: flesh slice modulus 2.29 -> 0.16 MPa with ripening.",
    "hia2018": "Hort Innovation 2018 'Avocado squeezing is bad': 10 N thumb press bruises firm-ripe fruit; shoppers "
               "3-30 N; bruises show after ~24 h.",
    "farley2019": "Farley et al. 2019 Am. J. Emerg. Med.: ~50,413 US avocado knife injuries 1998-2017, mostly the "
                  "non-dominant hand while pitting.",
    "ucd_avocado": "UC Davis Postharvest Technology Center, avocado produce facts (postharvest.ucdavis.edu/node/7115): "
                   "mature-green 5-13 C by cultivar; ripe fruit 2-4 C; ethylene at 20 C -> ripe in 3-6 d.",
    "marques2011": "Marques et al. (2011) VII World Avocado Congress (avocadosource WAC7 MarquesJose2011b): Hass at the "
                   "softening stage stored up to 10 d at 5-8 C with low risk of quality loss at eating ripe; avoid "
                   "< 5 C once past the rubbery stage.",
    "ripe_window_guides": "Produce guides (axonify produce knowledge; UC Davis-hosted consumer page): a stage-4/5 Hass "
                          "stays ripe ~24-48 h at room temperature before overripening (weak sources).",
    "atk_browning": "America's Test Kitchen 'How to keep avocado from turning brown': vacuum sealing kept halves green "
                    "5 d; plastic wrap + fridge minimal discolouration; lemon / lime juice or ascorbic acid slow browning "
                    "but taste sour; keeping the pit in does not protect the cut face (informal kitchen test).",
    "afm_howto": "Avocados From Mexico / freshfruitportal how-to: cut lengthwise around the seed, twist apart; remove "
                 "seed with a spoon (knife strike discouraged); score to dice; peel from the small end.",
    "usda_handling": "USDA ARS egg safety handling (ars.usda.gov .../eggsafety_handlingeggs.pdf): gently remove a stuck egg; cook cracked eggs thoroughly.",
    "masterclass_crack": "MasterClass 'How to crack an egg' (snippet): hold firm but gentle.",
    "cookscountry_storage": "Cook's Country egg storage (cookscountry.com/how_tos/5900-egg-storage): carton, middle/back shelf 35-40 F; door ~45 F.",
    "eggs_ca": "Egg Farmers of Canada (eggs.ca/?p=15718): crack on a flat surface; crack into a separate bowl first.",
    "atk_crack_separate": "America's Test Kitchen 'How to crack and separate eggs' (articles/1126): cold eggs, flat surface, three bowls, half shell lifts yolk specks.",
    "egginfo_separate": "British Lion egginfo.co.uk 'How to separate egg whites': shell-to-shell, hand, bottle methods.",
    "foodrepublic_separate": "Food Republic 'Ways to separate eggs' (1769095): methods and separator tools.",
    "incredibleegg_boil": "Incredible Egg 'How to make hard-boiled eggs': cold start, stand 9/12/15 min (M/L/XL); refrigerate eggs 7-10 d before; never microwave in shell.",
    "milkstreet_peel": "Milk Street 'How to peel hard-cooked eggs' (2020): steaming 12 min, 5 min ice bath, shake method ~60 % on warm eggs.",
    "atk_scrambled": "America's Test Kitchen 'Perfect scrambled eggs' (recipes/6604) and '6 tips' (articles/3118): fork-beat, half-and-half, medium-high then low.",
    "incredibleegg_scrambled": "Incredible Egg basic scrambled eggs: medium heat, push curds, off while glossy.",
    "pepin_pbs": "Jacques Pepin classic scrambled eggs (PBS American Masters): stop the cooking with reserved egg and cream.",
    "loveandlemons_fried": "Love & Lemons 'Fried egg': 1 tsp fat per egg, lid 2-3 min; over easy 15 s second side.",
    "incredibleegg_fried": "Incredible Egg basic fried eggs: steam-baste with 1 tsp water and a lid, or butter-baste.",
    "kenji_baste": "Lopez-Alt fried eggs via Tasting Table (1346566): baste with shimmering oil ~1 min, crisp lacy edges.",
    "ishs_1119_22": "ISHS Acta Hortic. 1119_22: avocado firmness devices (Sinclair iQ, firmometer, analogue meter, "
                    "hand squeeze) are non-linearly related -- readings are device-specific.",
    # ---------------------------------------------------------------- wave 1: bread roll
    "hirte_2012": "Hirte A. (2012) Extending crispness - controlling water migration in bread. PhD thesis, Wageningen "
                  "(edepot.wur.nl/221072): crispy roll ~90 x 50 mm; crust 1-3 mm (cut-off crust 1.0-1.5, ~1.3 mm); "
                  "crumb porosity 0.73-0.81 (uCT 6 um), cell walls 20-61 um; crumb aw ~0.97; crust water 9.0 % at "
                  "30 min, 13.9 % at 1 d (40 % RH, 22 C); critical crust water 10.1 % reached after 2.1 h; crust "
                  "break >= 3 N (1 x 10 mm blade); bake-off mass loss 2.4-4.1 %, 4 h storage 1.9-2.9 %.",
    "lagrain_2012": "Lagrain et al. (2012) Food Biophysics 7:173 (KU Leuven author manuscript): pan-bread crumb E "
                    "~20 kPa (5 h) -> ~70 (48 h) -> ~125 (96 h) -> ~157 kPa (168 h) (read off figures), Poisson 0.103, "
                    "loss factor 23 -> 10 %; mean 2D cell area 0.32 mm2, 83 cells/cm2; open porosity 0.81-0.89; "
                    "Gibson-Ashby E/Es = (rho/rho_s)^2; crumb moisture 0.46 -> 0.42 g/g.",
    "cardone_2020": "Cardone et al. (2020) Foods 9:260 (PMC7143436): Avrami crumb firming, control wheat bread, n=1: "
                    "A0 6.48 N, Ainf 66.01 N, k 0.186 (per day; unit to verify); crust L*~47, a* 9, b* 25.",
    "chaple_2023": "Chaple et al. (2023) LWT 174:114326 (PMC9883616): pan bread crust L*a*b* 47.6/17.05/32.2, crumb "
                   "72.06/0.94/15.37.",
    "kurek_2016": "Kurek et al. (2016) Food Technol. Biotechnol. 54(1) (PMC5105627): wheat rolls from 60 g dough, "
                  "specific volume 6.4 (3 h) -> 5.9 ml/g (48 h); moisture 47.8 %; crust browning index 50.4.",
    "musiienko_2025": "Musiienko et al. (2025) Front. Nutr. (PMC12497575): round buns 86-92 x 50-55 mm, 197-263 cm3.",
    "testde_2021": "Stiftung Warentest 'Aufbackbroetchen' (test.de 4206027): rolls 50-90 g; target crust "
                   "'hellgoldbraun bis dunkelgoldbraun'; crispness lost after 4 h, crumb unchanged; score raises the "
                   "crust share.",
    "ireks_rmt": "IREKS Kompendium, RMT-Backversuch Broetchen: 1000 g T550 flour, 50 g yeast, 15 g salt, 10 sugar, 10 fat"
                 " -> 30 rolls; proof 25 min 32 C; bake 20 min at 230 C with steam.",
    "leitsaetze_1993": "Leitsaetze fuer Brot und Kleingebaeck (1993): Kleingebaeck <= 250 g; Weizenbroetchen >= 90 % "
                       "wheat flour (2021 revision: verify).",
    "giovanelli_2021": "Giovanelli & Cappa (2021) Foods 10:417 (PMC7918450): browning starts > 105-120 C at aw < 0.4-0.7;"
                       " crust 140-180 C at the end of baking; browning index 35.5 (200 C) / 55.9 (225 C).",
    "soleimani_2014": "Soleimani Pour-Damanab et al. (2014) J Food Sci Technol 51:3439 (PMC4571265): crust thickness "
                      "grows with bake time; colour bands outer crust L* < 51.6, inner crust 51.6-62.3, crumb 62.3-71.2.",
    "kaur_2026": "Kaur et al. (2026) Plant Biotechnol. J. (PMC13278543): home toaster 3 / 3.5 / 4 / 4.5 min, 'starting "
                 "to look burned' at 4.5 min.",
    "utopia_aufbacken": "utopia.de 'Broetchen aufbacken': oven 150-170 C for 5-10 min after dampening; toaster rack "
                        "highest setting 2-3 min, flip, 1-2 min.",
    "oekotest_storage": "oekotest.de: keep rolls in a paper bag at room temperature; plastic softens the crust; freeze "
                        "fresh, do not refreeze.",
    "wikihow_slice": "wikiHow 'Slice Bread' (via kiwix mirror): serrated knife, roll flat, hand flat on top, saw from "
                     "the centre; stop short for a hinge.",
    "tastingtable_bagel": "Tasting Table 'safest way to cut a bagel': flat side down, palm on top, serrated knife "
                          "horizontally; halfway, stand on the uncut edge and finish downwards.",
    "luo_2026": "Luo & Yuan (2026) arXiv 2607.04367, robot food cutting: sawing at -30 N down-force, 2 cm/s, +-2 cm "
                "strokes; carrot = serrated-knife food (chopping failed, sawing succeeded).",
    "zhang_2024": "Zhang et al. (2024) Foods 13:397 (PMC10855756): rolls (80 g dough, 4 C) crumb hardness ~1.5 N (3 d)"
                  " -> ~4.0 N (7 d).",
    # ---------------------------------------------------------------- wave 1: banana
    "eu1333_2011": "Commission Implementing Reg. (EU) 1333/2011 bananas: min length 14 cm along the convex face from "
                   "the blossom end to where the peduncle joins the crown; min grade 27 mm (transverse thickness); "
                   "skin defects Extra <= 1, I <= 2, II <= 4 cm2.",
    "ciro05": "Ciro Velasquez et al. (2005) Rev. Fac. Nal. Agr. Medellin 58:2975 (Cavendish Valery, 24 C, 12 d): outer-"
              "curve length 23.3 +- 1.13 cm; diameter 34/36/34 mm at 25/50/75 %; peel puncture 13.6-15.2 N (2 mm "
              "probe); pulp penetration 7.20 -> 3.41 N; pulp E compression 1.75 -> 1.09 MPa, bending 5.16 -> 2.65 MPa;"
              " whole fruit fracture 129 N (bending, 14 cm span), 121 N (compression).",
    "mal22": "Malikongwa et al. (2022) Plant Science Today 9:364 (Grand Naine): length 19.0 cm; pedicel 17.3 x 10.15 "
             "mm; peel 2.56 mm; pulp 62.3 g / peel 23.2 g; IPGRI: slightly ridged, blunt tip, persistent style, "
             "slightly curved, cream pulp.",
    "chiw23": "Chiwate et al. (2023) Curr Agric Res J 11(1) (Cavendish, 22-27 C): day 1/4/6/8/10 mass 105/101/93/87/82 "
              "g, pulp:peel 1.69..1.78, Brix 4.5/6.8/8/16.7/23.7, firmness 44.8/36.8/34.2/28.1/22.1 N.",
    "bant24": "Bantayehu & Alemayehu (2024) Heliyon 10:e37143: Grand Naine 102 g, pulp:peel 2.18.",
    "amn12": "Amnuaysin et al. (2012) Trop. Nat. Hist. 12:127 (Hom Thong AAA): epidermis + cuticle, outer ground tissue "
             "(chloroplasts, starch), inner ground tissue with air spaces growing on ripening, bundles ringed by "
             "laticifers; peel 1.1-3.8 mm, thinning on ripening.",
    "guapo22": "Guapo & Ferreira (2022) Cadernos UniFOA 17(50): 3 carpels/locules; 'strings' = innermost mesocarp "
               "strands of parenchyma with two small vascular bundles and secretory spaces (not pure phloem).",
    "promusa": "ProMusa 'Banana flowers': tricarpellate inferior ovary, 200-300 ovules on a central placenta; the black "
               "dots in the centre are the ovule remnants.",
    "khanal23": "Khanal et al. (2023) Postharvest Biol Technol: outer peel layers are the mechanical backbone; peel "
                "splits axially in the fruit corners (ridges); peel fracture 53.0 -> 2.1 N on ripening (abstract).",
    "ringer21": "Ringer & Blanke (2021) J Food Meas Charact / Tropentag 2021 (German supply chain): peel hue R2..R7 "
                "98.7/97.3/92.7/89.4/87.5/82.0 deg, b* 56-72; peel water 90.5 % (<= R5) -> 82.9 % (R7); Shore 82 -> 42.",
    "hou15": "Hou et al. (2015) IJABE 8(6):99: ethylene-treated AAA at 20 C, stage 1 -> 7 in 7 days; peel L*, b* peak at "
             "stage 5, a* rises.",
    "ma22": "Ma et al. (2022) Curr Res Food Sci 5:1808: stage day bands RS1 1-4, RS2 5-6, RS3 7-8, RS4 9-10, RS5 11-12, "
            "RS6 13-15, RS7 > 15 d.",
    "markon": "Markon banana colour chart: 1 green, 2 green trace yellow, 3 more green than yellow, 4 more yellow than "
              "green, 5 yellow green tips, 6 all yellow, 7 yellow flecked brown.",
    "wflo": "WFLO Commodity Storage Manual, Bananas (2018): chilling below 13 C (pulp); ripening rooms 14-18 C; specific "
            "heat 3.56 kJ/kg K; moisture 74.3 %.",
    "mbeguie09": "Mbeguie-A-Mbeguie et al. (2009) J Exp Bot 60:2021: pedicel rupture force falls > 2x between day 1 and "
                 "3 (Grande Naine, 20 C); no abscission zone (ketsa25).",
    "imsabai": "Thai finger-drop report (TSRI BGJ4380011): finger-drop pull resistance ~5 kg (~49 N) on day 1 (Hom "
               "Thong / Namwa).",
    "bugaud14": "Bugaud et al. (2014) PBT 87:113: Grande Naine no visible bruise up to 200 mJ; susceptibility rises "
                "with ripening (abstract).",
    "col20": "Colombian Musaceae ripening (PMC7984252), Gros Michel day 0/4/6 L* 39.9/45.0/56.4, a* -26/-12/+13.1, "
             "b* 40.9/62.1/85.4.",
    "kim22": "Kim, Ohmura, Kuniyoshi (2022) arXiv 2203.09749: dual-arm banana peeling, 40.9 % consecutive success "
             "(PeelNeck 95 %, PeelRight 59 %, PeelLeft 64 %); neck too stiff on fresh fruit, pre-cut.",
    "banana_howto": "Tasting Table (1552037), Chiquita blog: peel from the stem (snap the neck) or from the tip "
                    "('monkey method': pinch the tip); rounds ~6 mm (3-6 mm, MyPlate); lemon juice against browning; "
                    "hang, keep apart from apples/avocados; refrigerate only when ripe.",
    # ---------------------------------------------------------------- wave 2: orange
    "sadka19": "Sadka et al. (2019) Front. Plant Sci. 10:1167: hesperidium; flavedo (oil glands), albedo, endocarp "
               "segment membrane, juice vesicles on stalks from the endocarp; 3 bundles per segment (1 dorsal, 2 "
               "septal) ending in the albedo; central axis (columella).",
    "smith18": "Smith, Ebrahimi, Ghosh, Dickerson (2018) PNAS 115:E5887 (PMC6042112): oil jets on bending the peel "
               "8.47 +- 4.03 m/s, 102 um; reservoir ~1 mm3 under an 84 um flavedo cap, burst at 30-140 kPa; flavedo E "
               "~103 MPa (30-300), albedo 0.79 +- 0.58 MPa; albedo 3.9 mm in the FE model.",
    "jentzsch22": "Jentzsch et al. (2022) Front. Mater. 9:979151: 'Lane Late' peel 6.2 mm, density 746 kg/m3, "
                  "compressive E ~0.02 MPa, Poisson -0.05, 0.51 MPa at 50 % strain.",
    "ucr_citrus": "UCR Givaudan Citrus Variety Collection data: Washington navel (ripe) 162-335 g, 61-89 x 61-88 mm, "
                  "rind 4.1-6.4 mm, juice 45-54 %, TSS 9.3-14.1, acid 0.49-0.77 %, 0-6 seeds per 10 fruit; Olinda "
                  "Valencia 113-299 g, L/W 1.0-1.1, rind 3.5-6.2 mm, juice 36.9-56.1 %, 1.2-6.6 seeds per fruit.",
    "edis891": "UF/IFAS HS891 Dooryard Citrus Guide: Valencia and navel 0-6 seeds; navel = rudimentary secondary "
               "fruit at the blossom end; navel juice turns bitter within several hours.",
    "citrus_patents": "USPTO PP21535 / PP27829 (sweet orange): 9.6-10.6 and 10-11 segments; PP27829 208 oil cells/cm2;"
                      " PP21224 albedo 2-5 mm; PP6047 navel peel 5-5.1 mm.",
    "knight01": "Knight, Klieber, Sedgley (2001) Ann. Bot. 88:1039: 8,000-12,000 oil glands on a mature Washington "
                "navel (abstract) -> ~40-60 /cm2 on ~200 cm2.",
    "eu543_citrus": "Reg. (EU) 543/2011 Annex I B 2 (oranges): min diameter 53 mm (size codes 0-13); min juice 33 % "
                    "navel / 35 % others; light green on <= 1/5 of the surface (superseded by 2023/2429: verify).",
    "orange_mech": "Khoshnam et al. 2022 Iran. J. Biosyst. Eng. 53:109 (punch 15.5-79 N, 1.5-7.9 N/mm, abstract); "
                   "Dadvar et al. 2015 (Valencia rupture 140-221 N, abstract); Montero et al. 2012 Sci. Hortic. (oil "
                   "glands rupture under 10-250 N sphere compression; Valencia no oleocellosis).",
    "ritenour13": "Ritenour et al. ASHS 2013: peel-removal rig (16 mm strip, 75 mm): Valencia hard-peel (strips break), "
                  "navel medium-easy; cold fruit (5 C) needs more force and tears more; no N values published.",
    "orange_juice": "BSDA QUID: juice SG 1.045 at 11.2 Brix; US4938985: single-strength viscosity ~4-6 mPa s at 8 C; "
                    "alphafoodie: 60-80 mL per medium orange by hand, roll 10-15 s, halve at the equator, ream.",
    "orange_storage": "UC Davis Produce Facts (Arpaia & Kader): 3-8 C, 90-95 % RH, up to 3 months; chilling (pitting) "
                      "after 3-6 weeks at ~4 C (UF PHNL194).",
    "orange_howto": "instructables 'Easy Orange Peeling' (score 6-8 mm deep round the navel, spiral to the stem); "
                    "kitchprep; The Daily Meal / A Couple Cooks: supreme -- peel and pith off, cut inside each "
                    "membrane; zest the flavedo only (albedo and membranes bitter, limonin).",
    "cci": "Citrus colour index CCI = 1000 a*/(L* b*) (Jimenez-Cuesta 1981); market CI target ~ +10..+20 (snippet).",
    # ---------------------------------------------------------------- wave 2: loaf
    "test25_toast": "Stiftung Warentest 'Toastbrot im Test' (23.04.2025, test.de 4421655): baked ~30 min under a lid "
                    "(thin soft crust), cooled in a clean room, sliced and bagged; most packs have no heels (organic "
                    "packs: 2); best-before 12-20 d; no classic preservatives needed.",
    "toast_pack": "Open Food Facts / REWE / Friedrich Verlag (snippets): 500 g pack, 20 slices of ~25 g, pack ~0.25 m long.",
    "nuss90": "Nussinovitch, Roy, Peleg (1990) Cereal Chem 67:101: commercial white sliced bread, crumb 0.19 g/cm3; "
              "tensile failure 0.31 / 0.39 N (12 mm neck, 10 / 50 mm/min) at 9.2-9.7 mm; compressed bread rarely fails.",
    "sehn20": "Sehn & Steel (2020) J Food Sci Technol 57:557 (PMC7016051): white control pan loaf firmness 1.98 / 3.62 / "
              "4.18 / 5.64 N at day 1 / 5 / 9 / 13 (25 C, PE bag); crumb moisture 39.35 -> 36.01 %.",
    "ishida14": "Ishida & Steel (2014) Food Sci Technol Int: Brazilian commercial white pan bread: 4.72 ml/g, crumb "
                "37.0 % moisture, firmness 276 gf day 1, crumb L*a*b* 74.73 / 0.37 / 15.51 (abstract).",
    "toaster26": "Stiftung Warentest toaster test (2026): 80 s (fastest) to > 2.5 min to the best toast; higher "
                 "settings = longer time, not more heat; browning scored as browned area.",
    "toast_misc": "EU 2017/2158 acrylamide benchmark 50 ug/kg (wheat soft bread); FSA 'Go for Gold'; Daily Meal: "
                  "'perfect toast' 216 s at 5/6, 14 mm; butter ~0.068 g/cm2 (Wedzicha); fridge stales ~6x faster "
                  "(popular, weak); Verbraucherzentrale: no fridge, freeze slices airtight.",
    "tins": "Runex 500 g bread tin: top inside 280 x 100 mm, depth 90 mm; hobby lidded toast tin 36 x 11 x 8 cm (2 loaves).",
    # ---------------------------------------------------------------- wave 2: potato
    "mc20": "Massana-Codina et al. (2020) Front Plant Sci 11:1287 (PMC7468465): periderm = phellem / phellogen / "
            "phelloderm; phellem 46-84 um, 6-10 cell layers (five European cultivars).",
    "mag25": "Magwaza et al. (2025) Front Plant Sci 16:1595828: lenticels <= 12 /cm2 (small tubers); apical buds ~7x "
             "the lateral density, 6-9 apical buds; respiration 1.9-2.75 mL CO2/kg h.",
    "bsl21": "Bundessortenamt Beschreibende Sortenliste Kartoffel 2021: Belana oval yellow-skinned, very flat-flat eyes; "
             "Gala round-oval, flat eyes; Linda long-oval, flat-medium eyes; flesh medium-yellow.",
    "potato_mech": "E 3.12 MPa (Cagnolati 2018; Celik 2019), 3.15 MPa / nu 0.45 / yield 1.18 MPa (Zhao 2023), "
                   "4.6-5.6 MPa, fracture 1.13-1.40 MPa (Polish clones, SV); fracture strain ~0.3; toughness ~212 J/m2 "
                   "(1996, SV); blackspot above ~50 g impact (Hyde 1991, SV).",
    "kk18": "Krishna Kumar et al. (2018) J Food Meas Charact: raw potato 1053 kg/m3, k 0.56 W/m K, alpha 0.15e-6 m2/s, "
            "initial freezing -1.8 C (abstract); ETB: k 0.55, cp 3.43 kJ/kg K.",
    "potato_cook": "Lee (2017) Food Eng Prog 21:303 review: first-order softening, k(100 C) 0.28-0.40 /min, Ea 103-145 "
                   "kJ/mol (Harada 1985); gelatinisation onset 61, peak 67.5, end 76-82 C (Witczak 2013, SV); PME "
                   "firming at 60-70 C (Gonzalez-Martinez 2004).",
    "gritzer": "Serious Eats (Gritzer): cold start in 1-2 % salted water; diced ~15-20 min, large whole up to ~1 h; "
               "done when a skewer passes without resistance; German guides 20-25 min whole from cold.",
    "potato_storage": "<8 C cold-induced sweetening; home 7-13 C dark; ~3 % weight loss in month 1; greening under "
                      "retail light 1-2.5 d at 20 C (Nofima), from day 4 / 2 at 10 / 15 C (JSHS 1990); BfR: remove green"
                      " parts and eyes generously.",
    "potato_peel": "Peeler vs knife study (SV): hand peeler ~1.1 mm, ~9 % waste; peeled slices under light: "
                   "glycoalkaloids 2 -> 74 mg/kg in 48 h (OSU EM 9407).",
    # ---------------------------------------------------------------- wave 2: tomato
    "ince16": "Ince, Cevik, Vursavus (2016) IJABE 9(6):200 (Jadelo F1): 72 x 70 x 53 mm, 165 g; plate firmness initial "
              "G/P/R 15.1 / 6.0 / 4.5 N/mm, average 13.2 / 4.7 / 3.1; 2 mm puncture skin rupture 6.92 / 3.99 / 2.74 N; "
              "a*/b* bands green -0.59..-0.47, breaker ..-0.27, turning ..0.08, pink ..0.60, light red ..0.95, red "
              "..1.21 (after Batu 2004).",
    "fdc_tomato": "USDA FoodData Central 170457: small whole 61 mm 91 g, medium 66 mm 123 g; water 94.5 g/100 g; slices "
                  "1/4 in 20 g, 1/2 in 27 g; wedge (1/4 medium) 31 g.",
    "bn05": "Bargel & Neinhuis (2005) J Exp Bot 56:1049: cuticle 9.9-15.7 um (red); skin E 252-293 MPa, failure "
            "<= 25-28 MPa at ~9-10 % strain (red).",
    "meta19": "Metabolites 9:93 (2019, Ailsa Craig red): pericarp 5.3 mm; pericarp ~50 %, exocarp 9.6 %, columella + "
              "placenta 10.0 %, locular tissue 23.3 % of FW.",
    "ucd_rost": "UC Davis Rost lab tomato anatomy: skin, pericarp, septa, columella, placenta, locular gel; most "
                "cultivated non-cherry types 4-5 locules; small types often 2 (Wikipedia).",
    "cfr_tomato": "7 CFR 51.1860 colour classes: green, breakers (<=10 %), turning (10-30 %), pink (30-60 %), light red "
                  "(60-90 %), red (> 90 %).",
    "wflo_tomato": "WFLO Commodity Storage Manual, Tomatoes: ripen 18.3-20 C; chilling < 12.8 C; ripe 7-10 C 3-5 d; "
                   "freezing -0.5 C; cp 4.08 kJ/kg K; mature-green seeds pushed aside by the knife (gel liquefied).",
    "zhang16": "Zhang et al. (2016) PNAS 113:12580: 8 d at 5 C cuts volatiles by 65 %, not recovered after 3 d at 20 C.",
    "tomato_mech": "Sirisomboon 2012 (whole stiffness 2000-10000 N/m, peel 70-90 % of firmness); Safitri 2022 (Hertz E "
                   "0.58-1.97 MPa, SV); Li 2017 (FEM: damage from ~3 % strain; 6.5 N at 10 %); Kultongkham 2021 "
                   "(bio-yield 2.57 N, SV); Lana 2004 (slice firmness Ea 87.5 / 94.8 kJ/mol).",
    "tomato_howto": "Wikibooks Tomato Concasse: cone-core, X 2.5 cm, blanch 30-45 s, ice water, peel, seed, dice "
                    "0.75 cm; Jessica Gavin: serrated or very sharp knife, slices 6-13 mm, 8 wedges; room temperature.",
    # ---------------------------------------------------------------- wave 2: onion
    "hb66_onion": "USDA AH-66 onion chapter (Adamicki): bulb = enlarged leaf bases + compact stem; 0 C, 65-75 % RH, 6-9 "
                  "months; rest 4-6 weeks; curing 3-5 % loss; respiration 3 (0 C) .. 8 (20 C) mg CO2/kg h; freezing "
                  "-0.8 C; medium 5.1-8.3 cm.",
    "bah04": "Bahnasawy et al. (2004) J Food Eng 62:255 (yellow Giza 20): 51.7 x 57.1 mm, 78.7 g, density 1.04 g/cm3, "
             "crushing 400-780 N, puncture 27.6-45.5 N, friction 0.67-1.34, rolling angle 20-31 deg (base).",
    "kim_heb20": "Hortic Environ Biotechnol (2020), 45 cultivars: 7.6 +- 0.2 fleshy scales, 2.3 growing centres, scale "
                 "thickness 5.5 +- 0.2 mm (abstract).",
    "wu25": "Wu et al. (2025) arXiv 2505.06016 'Droplet outbursts from onion cutting': layer ~3 mm; tissue E 2.2 MPa "
            "(skin on) / 1.4 (off); blunt / fast knives up to ~40x / ~4x more droplets; chilling did not change "
            "droplet speed.",
    "wei01": "Wei, Lintilhac, Tanguay (2001) Plant Physiol 126:1129: adaxial epidermal peel 120 um, E 3.5-8.0 MPa "
             "(turgor 0.14-0.54 MPa), nu 0.18-0.30; strips break at ~40 / ~20 gf (long./trans.).",
    "jafari16": "Jafari Malekabadi et al. (2016) J Agric Machinery 6:114: whole yellow onion E 2.03-5.45 MPa, nu "
                "0.26-0.45.",
    "lokke12": "Lokke et al. (2012) Sensors 12:16060: lachrymatory factor highest at ~1 min after chopping, gone by "
               "~10 min; Imai 2002 Nature 419:685 (LF synthase).",
    "onion_howto": "America's Test Kitchen 'How to cut an onion' (2016): halve through the root, peel, keep the root on,"
                   " horizontal then vertical cuts, claw grip; mince <= 3 mm, fine 3-6, medium 6-13, coarse 13-19 mm; "
                   "rings: trim a flat base; HWQ: 2-3 horizontal cuts, remove tough layers; UNL: chill 30 min.",
    "onion_misc": "Gracie 2012 (6th true leaf forms the outer skin; skins tear easily); Karate 2025 (cured neck "
                  "7.4-10.9 mm); Alamar 2020 (lateral buds up to 1/3 of mass; dormancy break week 8 at 1 C); frying "
                  "colour a* 5.7 -> 17.9, b* 14.2 -> 29.3 (SV); caramelising 30-45 min (typical).",
    # ---------------------------------------------------------------- wave 3: lemon
    "agu20": "Aguilar-Hernandez et al. (2020) Foods 10:62 (PMC7824522), Fino / Verna: 100-184 g, ED 55-67 mm, PD 72.7-91.9"
             " mm, ED/PD 0.6-0.8, peel 4.9-6.8 mm, 8.8-10.2 segments, 1.3-11.3 seeds, juice 31.7-44.3 %, TA 56-75 g/L, "
             "TSS 7.8-10.5, peel L* 64-68, a* -7.5..3.9, b* 42-49, hue 88-99 deg.",
    "jen22_lemon": "Jentzsch et al. (2022) Plants 11:991 (PMC9002614), Primofiore: 91 mm, 166 g, peel 5.26 mm, flavedo ~1 "
                   "mm (20 %), peel 36.4 % of mass, peel density 818 kg/m3, drop impact COR 0.31.",
    "ben26": "Bengisu et al. (2026) Biomimetics 11:451 (PMC13406879), retail lemon: puncture 41.2 N (5 mm), knife cutting "
             "36.3 N, compression 165.8 N at 16 mm; peel 0.68 g/cm3, albedo 0.39 g/cm3; oil glands 1.25 /mm2.",
    "eu2023_2429": "Delegated Reg. (EU) 2023/2429 citrus: lemons min juice 20 %, min diameter 45 mm, size codes 0 (79-90)"
                   " .. 7 (45-52 mm).",
    "lemon_misc": "MAR22 Verna (PMC9407548): 93.7-98.1 mm, peel 6.7-7.1, seeds 1.3-2.3; LIA22 Eureka chilling < 10 C, "
                  "3.5 % loss in 80 d; MIT20 degreening CCI -16.3 -> -1.1; USDA FDC 167747 juice of 1 lemon 48 g, wedge "
                  "5.9 g; Wikipedia citric acid 5-6 %, pH 2.2; zest = flavedo only (albedo bitter).",
    # ---------------------------------------------------------------- wave 3: cucumber
    "dewiki_sg": "de.wikipedia 'Salatgurke': normally three locules; trade classes 300-400 / 400-500 g.",
    "unece_cuc": "UNECE FFV-15 (2017): Extra / I max 10 mm arc height per 10 cm, II 20 mm; weight spread 150 g (>= 400 g);"
                 " EU 1677/88 (expired 2009): >= 250 g, >= 25 cm (250-500 g), >= 30 cm (>= 500 g); seeds soft.",
    "sakata11": "Sakata et al. (2011) J Japan Soc Hort Sci 80:420: greenhouse type puncture (3 mm) flesh 8.7 N, placenta "
                "2.8 N, skin 9.6-9.8 N; 25-30 cm, ~260 g.",
    "jah16": "Jahangiri et al. (2016) CIGR J 18:323: apparent modulus 0.85-1.39 MPa, failure stress 0.53-0.85 MPa, strain "
             "0.45-0.53; firmness -49 % in 9 d; stem end 7.53 vs blossom end 5.01 N.",
    "disect_cuc": "Heiden et al. (2021) DiSECt (arXiv 2105.12244) Table A1: cucumber E 2.5 MPa, nu 0.37, 950 kg/m3; SliceIt!"
                  " (2404.02569): 5 mm slices, stiffness jump cucumber -> board.",
    "ucd_cuc": "UC Davis Produce Facts, Cucumber: 10-12.5 C, 95 % RH, < 14 d; chilling below 10 C after 1-3 d (7.2 C: "
               "2-3 d); ethylene 1-5 ppm yellows; freezing -0.5 C; respiration 12-15 mL CO2/kg h at 10 C.",
    "cuc_misc": "rett24 cuticle 4.0 um (1.1-10.7); ando15 peel layer 1-2 mm; wang22 skin a* -20.6, b* 27.5 (yellowing to"
                " b* 35 in 20 d); dhall10 wrapped 0.66 vs unwrapped 11.1 % loss in 15 d at 12 C; shoji22 typical 350 g;"
                " wikibooks: fork scrape for wax, zebra peel, spoon out seeds, salt 30 min.",
    # ---------------------------------------------------------------- wave 3: croissant, baguette
    "leitsaetze_fb": "Leitsaetze fuer Feine Backwaren (2025) 2.2.3.4: Plunder >= 30 kg butter (or equivalent fat) per 100"
                     " kg flour, flaky structure; croissant = rolled Plunder, often crescent-shaped; 2.1.1: butter in the"
                     " name -> no other fats.",
    "ste25": "Steinkellner, Kroll, Franke (2025) Curr Res Food Sci 10:101042 (PMC11995129): butter croissant ~50 % butter "
             "on flour, 3 single folds = 27 fat layers, specific volume 3.13 +- 0.66 cm3/g, pore surface 6.62 /cm; TPA "
             "firmness 5.16 N (1 h) -> 19.2 N (24 h); bake 200/220 C 20 min; lamination 13-15 C.",
    "croissant_misc": "Wikipedia en/de/fr Croissant: rolled 3.5-4 turns; optimum 16-50 fat layers (Ooms 2016); butter melts "
                      "32-34 C; final sheet 3.5 mm; bake 210-220 C. USDA FDC 174987: 28 / 42 / 57 / 67 g; water 23.2, fat "
                      "21.0 g/100 g. Weekend Bakery: egg wash 2 coats; reheat from frozen 180 C 8 min.",
    "wp_baguette": "Wikipedia fr/en 'Baguette': ~65 cm, 4-6 cm wide, 3-5 cm high, ~250 g; tradition = flour, water, "
                   "yeast/levain, salt (decret 93-1074); scores at 45 deg with a razor; stales faster than a boule; Grand "
                   "Prix de Paris: 55-65 cm, 250-300 g, salt 18 g/kg.",
    "wp_grigne": "Wikipedia fr 'Grigne': coupe parisienne = 5 medium overlapping cuts along the axis, ~1 cm deep; the "
                 "grignes open into raised lips ('oreilles').",
    "dv2012": "Della Valle et al. (2012) J Cereal Sci 56:440: French breads (64), mean crust 1.7 mm; crust ends ~160 C, "
              "~5 % moisture.",
    "zou2024": "Zou et al. (2024) npj Sci Food (PMC11390739): baguette crust L*a*b* 57.5 / 10.5 / 34.9, crumb 66.4 / -1.7 /"
               " 15.7 (powders, SV); bake 230 C 30 min with steam.",
    "baguette_misc": "King Arthur baguettes (232 C, 24-28 min, steam; paper bag overnight, freeze); van Vliet 2006 crispness"
                     " lost within 2 h at 80 % RH; Wikipedia Staling: fastest at ~4 C, partly reversed at 60 C.",
    # ---------------------------------------------------------------- wave 3: bell pepper
    "he22": "He et al. (2022) Plants 11:985 (PMC9003265): blocky red cv. Gina 237 +- 37 g, 77.6 x 81.3 mm, wall 7.5 +- "
            "0.7 mm (orange 6.9); cuticle 10.4 / 8.7 um; bite firmness 11.6 / 6.9 N, -81 % after 15 d at 20 C.",
    "alt20": "Althaus & Blanke (2020) J Imaging 6:122: German retail peppers 230-245 g; no stomata; at 17 C / 55 % RH "
             "2.9-4.8 g/d loss, shrivel from 6-8 % (4-5 d); roughness Sa 2 / 7 / 24 um; gloss 450-500 -> 80-90; "
             "non-climacteric.",
    "bar20": "Barbosa et al. (2020) Molecules (PMC7287789): pericarp L*a*b* green 33.7 / -8.6 / 16.9, red 38.8 / 31.4 / "
             "27.8, yellow 61.4 / 5.0 / 50.0; whole-fruit shelf life 12-28 d.",
    "sha26": "Sharma et al. (2026) Sci Rep (DUS): locules 2-4, mostly 3; 1000-seed weight 4 to > 6 g.",
    "pepper_misc": "Mensah 2026: 38-142 seeds per fruit; Afolabi 2023: whole-fruit 8 mm probe 43.5 N; Darre 2024 / "
                   "Hanaei 2022: chilling below 7 C (7-10); EU 2023/2429 sweet peppers (classes, sizing); USDA ring "
                   "(3 in x 1/4 in) 10 g; Harvey / Sweeper harvesting robots (36.9 s, 24 s cycles); Wikibooks: remove "
                   "stem, core, seeds and white veins; char to loosen the skin.",
    # ---------------------------------------------------------------- wave 1: carrot
    "eu730_1999": "Commission Reg. (EC) 730/1999 carrot marketing standard (mirrors UNECE FFV-10): main crop >= 20 mm or"
                  " 50 g, Extra <= 45 mm / 200 g; green/violet tops: Extra none, Class I <= 1 cm (<= 10 cm long) / "
                  "<= 2 cm (longer), Class II <= 2 / 3 cm; firm, not woody, not forked.",
    "con22": "Conversa et al. (2022) Foods (PMC8750513): Nantes 'Presto' 135.4 g, 210.8 mm, 31.4 mm; 'CPT' 110.4 g, "
             "182.7 mm, 30.3 mm; hand peeler removes ~2 mm = 12.1 / 14.6 % of fresh mass.",
    "jah18": "Jahanbakhshi et al. (2018) Food Sci Nutr 6:1898 (PMC6189625): 154.6 mm, 72.7 g; density 1.04 g/cm3; "
             "moisture 87.2 %; 3-point bending Fmax 48.6 N at 21.9 mm, 0.50 J; blade shear 41.1 N; compression 71.9 N.",
    "vob24": "Vo-Bui, Rogers, Pegg (2024) R Soc Open Sci 11:230420 (PMC10805594): Nantes orthotropic constants after "
             "Stopa et al. 2017: xylem E 4.62-4.88 MPa, phloem 4.33-4.85 MPa, nu 0.47-0.48, G 1.47-1.66 MPa; E falls "
             "1.32x in 7 d (fridge, zip bag); core/root diameter ratio falls crown -> tip.",
    "peng14": "Peng, Tang, Barrett, Sablani, Powers (2014) J Food Eng 122:84: carrot softening, 2nd-order fractional "
              "conversion, k 0.035 (80 C) -> 1.453 1/min (110 C), Ea 138.9 kJ/mol, Finf/F0 0.037-0.008; raw dice "
              "(12.7x12.7x6 mm, 70 % strain) 274 N; xylem core 4-7 mm in dice.",
    "singh17": "Singh et al. (2017) Indian J Hort 74:56: 40 genotypes, root 30.55 mm, core 11.74 mm (ratio ~0.38).",
    "hb66": "USDA Agric. Handbook 66 (2016), carrot chapter: 0-1 C / 98-100 % RH 7-9 months; bagged 2-3 weeks at 3-5 C; "
            "freezing point -1.2 C; respiration 46-95 mg CO2/kg h at 20 C; Nantes prone to cracking and tip breakage; "
            "parenchyma cells ~50 um; softening = turgor loss then pectin beta-elimination.",
    "herp99_calbo00": "Herppich et al. (1999) PBT; Calbo (2000) Bragantia: fresh turgor ~0.7 MPa; turgor firmness ~0 "
                      "below relative water content 0.95 (~5 % water loss) -- the limp carrot.",
    "storage_loss": "Snippets: unwrapped 20 C / 50 % RH unacceptable after ~3 d (US4919948); > 5 % loss in 4 d bulk "
                    "unsaleable (Semina 2010); 5-8 % loss unsaleable.",
    "sim09_cz95": "Simoes et al. (2009) PBT; Cisneros-Zevallos et al. (1995) J Food Sci 60:320: white blush on cut carrot "
                  "= surface dehydration (reversible within ~1-1.5 h) then lignification (irreversible after ~15 h).",
    "pac16_dry": "Paciulli et al. (2016) J Food Sci Technol; drying study IJFP 10.1080/10942912.2010.489209: raw carrot "
                 "L*a*b* 50-55 / 25-29 / 30-40 (hue ~50-54 deg).",
    "alv00": "Alvarez et al. (2000) J Texture Stud: SENB on carrot/celery/apple, K_IC 10-50 kPa m^0.5, fracture energy "
             "40-400 J/m2.",
    "morpheus_2024": "Ye et al. ICRA 2024 MORPHeus (arXiv 2404.06570): instrumented peeler, carrot 90.1 % area peeled; "
                     "skin and flesh colours hard to tell apart -> force/vibration sensing.",
    "carrot_cuts": "ATK 'How to cut carrots' (articles/3368), Leiths, Wikipedia culinary cuts: plank first so it does "
                   "not roll; 5-7.5 cm lengths; julienne 3x3x40-50 mm, brunoise 3 mm, batonnet 6x6x50-60 mm, dice "
                   "6/12/19 mm; peel from the thick end, rotate after each stroke.",
    "carrot_boil": "Evolving Table / Food & Home snippets: 1/4 in rounds 4-5 min crisp-tender (+1-2 min soft); whole "
                   "10-15 min; not much more than 10 min for slices.",
    "diarch_rows": "Botany: lateral roots arise beside each xylem pole -> a diarch root has 4 rows of lateral roots "
                   "(general text; carrot-specific confirmation in Esau 1940 Hilgardia 13:175 to obtain).",
}

# name: (unit, value or distribution, status, source key or model note)
#   distribution forms: ("normal", mean, sd, lo, hi) truncated; ("uniform", lo, hi); ("lognormal", median, gsd)
EGG = {
    "length_L": ("mm", ("normal", 57.0, 1.8, 52, 63), "SOURCED", "kop: 58.05 mm; turjaf: 56.12 mm"),
    "shape_index_B_over_L": ("1", ("normal", 0.765, 0.02, 0.70, 0.81), "SOURCED", "kop: 78.25 %; alt: normal 72-76"),
    "asymmetry_w": ("mm", ("uniform", 1.6, 3.4), "ILLUSTRATIVE", "narushin2021 parameter w; range assumed"),
    "shell_thickness": ("mm", ("normal", 0.35, 0.03, 0.28, 0.42), "SOURCED", "kop: 0.28-0.41 mm; sb: 0.35 mm"),
    "cuticle_thickness": ("um", ("normal", 7.85, 1.5, 5.0, 12.5), "SOURCED", "ir: 5.6-12.5 um, mean 7.85"),
    "mammillary_thickness": ("um", 100.0, "SOURCED", "fr: ~100 um (palisade ~300, vertical crystal 3-8)"),
    "membranes_thickness": ("um", ("normal", 70.0, 8.0, 50, 90), "SOURCED", "fr/mem: ~70 um combined"),
    "outer_membrane_share": ("1", 0.75, "SOURCED", "ubc: outer membrane ~3x thicker than inner"),
    "pore_count": ("1", ("int_uniform", 10000, 20000), "SOURCED", "por: 10,000-20,000; ~2x denser at blunt pole"),
    "shell_E": ("GPa", ("uniform", 15.0, 43.0), "SOURCED", "sev: 41-43 GPa nanoindentation; sb model 15 GPa"),
    "shell_poisson": ("1", 0.30, "TYPICAL", ""),
    "shell_density": ("kg/m3", 2300.0, "ILLUSTRATIVE", "checked: shell 8-11 % of mass (br)"),
    "albumen_density": ("kg/m3", 1035.0, "SOURCED-VERIFY", "cs: liquid white ~1.03 g/ml (weak source)"),
    "yolk_density": ("kg/m3", 1030.0, "TYPICAL", ""),
    "whole_specific_gravity": ("1", "1.078-1.097 fresh, 1.040-1.059 after >30 d", "SOURCED", "ro"),
    "yolk_mass_fraction": ("1", ("normal", 0.30, 0.015, 0.27, 0.32), "SOURCED", "br: 27-32 %; usda 31 %"),
    "albumen_layer_shares": ("1", "outer thin 0.23, thick 0.57, inner thin 0.17, chalaziferous 0.03", "SOURCED",
                             "usda egg grading manual"),
    "germinal_disc_diameter": ("mm", ("uniform", 2.0, 3.0), "SOURCED", "ab: infertile blastodisc 2-3 mm"),
    "yolk_colour_fan": ("DSM", ("uniform", 9.0, 14.0), "SOURCED", "dsm YolkFan 1-16; consumers prefer >= 12"),
    "storage_temp_C": ("C", ("choice", 6.0, 22.0), "ILLUSTRATIVE", "fridge or counter"),
    "age_days": ("d", ("uniform", 1, 21), "ILLUSTRATIVE", "days since lay"),
    "air_cell_height": ("mm", "1.5 + k(T) * age; k(6 C) = 0.08, k(22 C) = 0.19 mm/d", "SOURCED-VERIFY",
                        "gr: < 4 mm after 28 d at 6 C; > 6 mm after 25 d at 22 C; graded vs eu589/usda"),
    "F_break_equatorial": ("N", ("normal", 35.0, 8.0, 12, 59), "SOURCED", "kop: Hisex 36.43 N short axis, 10-59 N; "
                           "icf11 34+-6 N"),
    "F_break_axial": ("N", ("normal", 37.0, 9.0, 12, 62), "SOURCED", "kop 34.92 N long axis; icf11 41+-8 N; "
                      "alt: highest along the polar axis at the sharp end"),
    "F_break_thickness_scaling": ("1", "F ~ (t / 0.35 mm)^2", "ILLUSTRATIVE", "thin-shell scaling assumed"),
    "shell_contact_stiffness": ("N/mm", "Reissner k = 4 E t^2 / (R_local sqrt(3(1-nu^2)))", "DERIVED",
                                "reissner1946 + shell_E + shell_thickness"),
    "microcrack_band": ("1", 0.85, "ILLUSTRATIVE", "F/F_break in [0.85, 1) may leave an invisible micro-crack"),
    "microcrack_strength_loss": ("1", 0.40, "ILLUSTRATIVE", "F_break of a micro-cracked shell x (1 - 0.40)"),
    "membrane_tear_ratio": ("1", 1.6, "ILLUSTRATIVE", "check -> leaker when the load exceeds 1.6 x crack force"),
    "edge_concentration": ("1", "F_crack_edge = F_break * sqrt(r_edge / 5 mm), r_edge >= 0.2 mm", "ILLUSTRATIVE",
                           "sharp tool parts crack at lower force (Morgenstern crackforce depends on parts)"),
    "friction_shell_rubber": ("1", ("uniform", 0.45, 0.75), "ILLUSTRATIVE", "dry shell vs gripper pad"),
    "crack_strike_model": ("-", "food/egg_crack.py P: coupled mass 0.05 kg; membrane tears at F/Fc 1.8 (flat), "
                           "1.05 (rim), 3.0 (egg-on-egg); shatter at 2.6 / 2.2 / 2.4; shards 0.15 / 0.6 / 0 per F/Fc; "
                           "yolk puncture 0.06 per shard", "ILLUSTRATIVE",
                           "structure from atk_egg, egginfo_crack, atk_egg_on_egg; numbers to calibrate"),
    "cook_thermal_props": ("W/mK, kg/m3, J/kgK", "albumen 0.55/1035/3700, yolk 0.34/1030/2800, shell 0.45/2300/900",
                           "TYPICAL", "egg thermophysical properties, to cite (e.g. Denys et al. 2003)"),
    "cook_yolk_thresholds": ("1", "f65 < 0.15 runny, < 0.45 jammy, < 0.85 creamy, else firm", "CALIBRATED",
                             "alphafoodie timing table (hot start, medium eggs) + Lopez-Alt fridge-egg timings"),
    "impact_effective_mass_fraction": ("1", 0.53, "CALIBRATED",
                                       "c = sqrt(m_eff)/rate fitted to cohen2025: side-on drops onto a hard surface "
                                       "crack ~45 % at 10 mm (model 45 %), vertical 55 % at 8 mm (model ~57 %); the "
                                       "model population is more heterogeneous than one MIT batch (8 mm side-on: "
                                       "model ~28 % vs 5 %)"),
    "impact_rate_factor": ("1", 1.30, "ILLUSTRATIVE", "dynamic/static crack force; strength rises with loading rate "
                           "(only sqrt(m_eff)/rate is identified by the calibration)"),
    # ---- freshness and ageing (egg SDT stack L0/L7, docs/EGG_SDT.md; per-egg draws in the EXT_DRAW stream)
    "lay_SG": ("1", ("normal", 1.087, 0.005, 1.078, 1.097), "SOURCED-VERIFY",
               "ro: fresh specific gravity 1.078-1.097; sd assumed"),
    "hu_fresh": ("HU", ("normal", 81.3, 3.0, 72.0, 92.0), "SOURCED-VERIFY",
                 "pan2026 day 0: 81.36 / 81.13 (means of 6 replicates); egg-to-egg sd assumed"),
    "yi_fresh": ("1", ("normal", 0.43, 0.02, 0.38, 0.47), "SOURCED-VERIFY", "pan2026 day 0: 0.43; sd assumed"),
    "ph_fresh": ("1", ("normal", 7.9, 0.15, 7.6, 8.5), "SOURCED-VERIFY",
                 "pan2026 day 0: 7.90 / 7.91; range 7.6-8.5 (stadelman_ph); sd assumed"),
    "ageing_rate_factor": ("1", ("lognormal", 1.0, 1.25), "ILLUSTRATIVE",
                           "egg-to-egg factor on every ageing rate (shell porosity, cuticle)"),
    "ageing_table": ("-", {"days": [0, 5, 10, 15, 20, 25, 30],
                           4.0: {"weight_loss_pct": [0.0, 0.39, 1.00, 1.64, 2.01, 2.36, 2.76],
                                 "yolk_index": [0.43, 0.43, 0.43, 0.43, 0.42, 0.39, 0.37],
                                 "albumen_height_mm": [6.26, 5.76, 5.36, 5.11, 5.05, 4.76, 4.76],
                                 "haugh_unit": [81.36, 76.44, 72.40, 70.80, 69.51, 68.93, 67.63],
                                 "albumen_pH": [7.90, 8.36, 8.40, 8.45, 8.54, 8.61, 8.64]},
                           25.0: {"weight_loss_pct": [0.0, 0.67, 1.55, 2.27, 2.80, 3.38, 4.05],
                                  "yolk_index": [0.43, 0.41, 0.35, 0.35, 0.33, 0.28, 0.27],
                                  "albumen_height_mm": [6.25, 4.63, 4.45, 4.12, 3.60, 2.90, 2.14],
                                  "haugh_unit": [81.13, 72.52, 68.65, 63.50, 58.54, 50.25, 38.32],
                                  "albumen_pH": [7.91, 8.73, 8.74, 8.76, 8.78, 8.80, 8.82]}},
                     "SOURCED", "pan2026 Table (control eggs, means of n = 6)"),
    "ageing_law": ("-", "state-based: each quantity's change per day at 4 C and 25 C is read off the pan2026 curve at "
                   "the egg's current value; ln(rate) is interpolated linearly in T between, extrapolated beyond; "
                   "an egg's own rate factor scales all rates", "DERIVED", "pan2026 + ageing_rate_factor"),
    "air_cell_from_mass_loss": ("-", "air-cell volume = the lens at lay (1.5 mm) + the lost water volume (mass loss / "
                                "1000 kg/m3), as a lens at the blunt end of the egg's own shell", "DERIVED",
                                "pan2026 weight loss; cross-check against gr (air_cell_height)"),
    "float_test_model": ("-", "posture in water from the egg's geometry: the net weight (W - buoyancy) acts at "
                         "(m c_m - rho_w V c_b)/(m - rho_w V); the egg rolls until that point is above the contact: "
                         "lies flat / tilted / stands on the pointed end; floats when SG < 1", "DERIVED",
                         "statics on the twin's part graph"),
    "air_cell_grades_usda": ("mm", "AA <= 3.2, A <= 4.8, B > 4.8", "SOURCED", "usda_std"),
    "best_before_days": ("d", 28.0, "SOURCED-VERIFY", "eu2023_2465 (589/2008: <= 28 d after laying)"),
    "peel_ph_threshold": ("1", 8.8, "SOURCED-VERIFY", "peel_ph: easy peeling at albumen pH 8.7-8.9"),
    "peel_ph_width": ("1", 0.15, "ILLUSTRATIVE", "logistic width of P(clean peel) around the threshold"),
    "sensor_scale_sd_g": ("g", 0.3, "ILLUSTRATIVE", "kitchen scale"),
    "sensor_calliper_sd_mm": ("mm", 0.3, "ILLUSTRATIVE", ""),
    "sensor_photo_size_rel_sd": ("1", 0.03, "ILLUSTRATIVE", "with a scale reference in the image; without: PRIOR"),
    "sensor_photo_si_sd": ("1", 0.025, "ILLUSTRATIVE", "shape index from the silhouette (food/egg_vision.segment_egg): "
                           "RMS error 0.023 on 8 twin renders (scripts/eval_egg_photos.py); no real-photo check yet"),
    "sensor_candling_air_sd_mm": ("mm", 0.6, "ILLUSTRATIVE", "air-cell depth read against the light"),
    "sensor_candling_crack_detect": ("1", 0.6, "ILLUSTRATIVE", "P(a micro-crack is seen when candling)"),
    "sensor_float_tilt_sd_deg": ("deg", 8.0, "ILLUSTRATIVE", "tilt of the long axis read from an image"),
    "sensor_albumen_height_sd_mm": ("mm", 0.4, "ILLUSTRATIVE", "thick-albumen height of an opened egg from depth"),
    "sensor_yolk_index_sd": ("1", 0.02, "ILLUSTRATIVE", "yolk height / diameter from depth"),
    "sensor_force_rel_sd": ("1", 0.10, "ILLUSTRATIVE", "gripper force reading"),
}

APPLE = {
    "cultivar": ("-", "Gala-like", "ILLUSTRATIVE", "shape/colour archetype (striped orange-red over yellow)"),
    "diameter_D": ("mm", ("normal", 75.0, 3.5, 66, 86), "SOURCED", "wiki_gala: 78 % of Gala fruit 70-80 mm"),
    "height_over_D": ("1", ("normal", 0.88, 0.03, 0.80, 0.96), "SOURCED", "uspto_pp20028: 71.3 x 81.4 mm"),
    "lobe_amplitude": ("1", ("uniform", 0.008, 0.03), "ILLUSTRATIVE", "five-fold relief at the calyx end"),
    "lopsidedness": ("1", ("uniform", 0.0, 0.03), "ILLUSTRATIVE", ""),
    "skin_thickness": ("um", ("normal", 55.0, 10.0, 33, 73), "SOURCED", "konarska2006: 33.3-73.1, mean 55.2 um"),
    "skin_tensile_strength": ("MPa", 2.56, "SOURCED-VERIFY", "jets_skin_tensile"),
    "lenticel_density": ("1/cm2", ("uniform", 4.3, 12.6), "SOURCED", "ishs_1099_97: mean 7.11 /cm2"),
    "flesh_density": ("kg/m3", ("normal", 860.0, 40.0, 760, 970), "SOURCED",
                      "wiki_gala 0.86 g/cm3; fuji 955-968 kg/m3 (degruyter)"),
    "cortex_porosity": ("1", ("uniform", 0.17, 0.30), "SOURCED", "fmach_porosity: Jazz 17 % .. Golden Del. 29.8 %"),
    "core_density_excess": ("kg/m3", 60.0, "ILLUSTRATIVE", "core porosity 14.5 % vs cortex 26 % (frontiers2015)"),
    "flesh_E": ("MPa", ("lognormal", 3.0, 1.7), "SOURCED",
                "fuji 5.0 MPa (consensus f4110d16); quasi-static 4.2-7.8; stored 0.5-1.7 (tjaf2005)"),
    "flesh_poisson": ("1", 0.30, "TYPICAL", ""),
    "storage_days": ("d", ("uniform", 0, 120), "ILLUSTRATIVE", "E multiplied by exp(-storage_days / 180)"),
    "mt_firmness": ("N", ("normal", 75.0, 8.0, 50, 95), "SOURCED", "msu2025: Gala ~18 lb (80 N) at harvest"),
    "flesh_shear_strength": ("MPa", ("normal", 0.40, 0.05, 0.28, 0.5), "SOURCED",
                             "fuji shear 0.45 MPa; failure stress 0.32-0.37 MPa (consensus)"),
    "bruise_energy_threshold": ("J", ("uniform", 0.004, 0.012), "SOURCED-VERIFY",
                                "no bruise below 0.25 m/s impact; bruises from 0.011 J pendulum (consensus)"),
    "bruise_volume_per_J": ("cm3/J", "0.01 * V_fruit / 0.41 J", "SOURCED-VERIFY",
                            "fsct2008 (about 1 % of fruit volume at 410 mJ); linear in energy"),
    "cutting_force_peak": ("N", ("normal", 25.0, 5.0, 12, 40), "ILLUSTRATIVE", "no apple cutting force found"),
    "friction_skin_rubber": ("1", ("uniform", 0.70, 0.95), "SOURCED-VERIFY", "fuji vs TPU rubber 0.88 (consensus)"),
    "core_line_radius": ("mm", ("normal", 14.0, 1.5, 11, 18), "SOURCED", "uspto: core diameter 24.7-36.7 mm"),
    "core_length": ("mm", ("normal", 41.0, 3.0, 34, 48), "SOURCED", "uspto_pp29089: 41.1 mm"),
    "locule_length": ("mm", ("normal", 16.0, 1.0, 13, 19), "SOURCED", "uspto: 15.2 x 7.9, 16.6 x 9.9 mm"),
    "locule_width": ("mm", ("normal", 8.5, 0.8, 7, 10.5), "SOURCED", "uspto (radial width in the long. section)"),
    "endocarp_thickness": ("mm", 0.35, "ILLUSTRATIVE", "leathery/papery carpel wall"),
    "seeds_per_carpel": ("1", "P(0,1,2) = 0.1, 0.4, 0.5", "SOURCED", "wiki_apple: 1-2 seeds per carpel"),
    "seed_length": ("mm", ("normal", 8.5, 0.6, 7.5, 9.6), "SOURCED", "sciRep2021: 7.69-9.43 mm"),
    "seed_width": ("mm", ("normal", 4.5, 0.3, 3.9, 5.0), "SOURCED", "sciRep2021: 4.0-4.9 mm"),
    "seed_thickness": ("mm", ("normal", 2.7, 0.2, 2.2, 3.2), "TYPICAL", ""),
    "stem_length": ("mm", ("uniform", 10.0, 28.0), "SOURCED", "uspto: 10.5, 12-14, 28.2 mm"),
    "stem_diameter": ("mm", ("normal", 2.8, 0.15, 2.5, 3.1), "SOURCED", "uspto: 2.7-3.0 mm"),
    "vascular_bundle_radius": ("mm", 0.45, "ILLUSTRATIVE", "10 primary bundles (frontiers2015)"),
    "browning_dE": ("dE", "2.38 (0 min), 2.68 (30 min), 4.05 (60 min)", "SOURCED", "pmc7022590 (Golden Delicious)"),
}

AVOCADO = {
    "cultivar": ("-", "Hass", "SOURCED", "avo_au_stages; uspto_pp14239"),
    "length_L": ("mm", ("normal", 98.0, 6.0, 80, 120), "SOURCED-VERIFY", "specialty: 7-15 cm; no Hass mm data held"),
    "diameter_D": ("mm", ("normal", 66.0, 3.5, 56, 76), "TYPICAL", "chosen so mass ~ 200 g (uspto mean 204 g)"),
    "fruit_mass_check": ("g", "142-340 (CAC), 200-300 (wiki), mean 204 (uspto_pp14239)", "SOURCED", ""),
    "neck": ("1", ("uniform", 0.08, 0.20), "ILLUSTRATIVE", "obovate/pyriform taper of the stem end"),
    "skin_thickness": ("mm", ("normal", 1.6, 0.15, 1.2, 2.0), "SOURCED", "uspto_pp14239: Hass skin 1.6 mm"),
    "skin_layers": ("-", "cuticle, epidermis (1 layer), hypodermis (1-3 cells), parenchyma, stone-cell layer",
                    "SOURCED", "cs42"),
    "pebble_scale": ("mm", ("uniform", 1.2, 2.0), "ILLUSTRATIVE", "no source for tubercle size"),
    "seed_length": ("mm", ("normal", 40.0, 2.5, 34, 47), "DERIVED", "seed mass 28.4-30.6 g (uspto) at 1.1 g/cm3"),
    "seed_width_ratio": ("1", ("normal", 0.86, 0.04, 0.78, 0.95), "SOURCED", "uspto: 'variable, mostly ellipsoid'"),
    "seed_offset": ("mm", ("uniform", 0.5, 2.5), "SOURCED-VERIFY", "cs42: seed slightly off-centre"),
    "seed_coat_thickness": ("mm", 0.25, "ILLUSTRATIVE", "two papery testa layers (cs42); 'paper-thin' (ufl)"),
    "endocarp_thickness": ("mm", 0.15, "ILLUSTRATIVE", "a few rows of flattened cells (cs42)"),
    "testa_adheres_to": ("-", ("choice_str", "seed", "flesh"), "SOURCED-VERIFY",
                         "cs42: endocarp may adhere to the seed coat; side not settled"),
    "flesh_density": ("kg/m3", 990.0, "ILLUSTRATIVE", "whole fruit 970-1018 (clark2007)"),
    "seed_density": ("kg/m3", 1100.0, "ILLUSTRATIVE", ""),
    "skin_density": ("kg/m3", 1050.0, "ILLUSTRATIVE", ""),
    "mass_ratio_seed_skin_flesh": ("%", "16:12:72 (avocadosource) / 14:13:73 (uspto)", "SOURCED", "check"),
    "ripeness_stage": ("1..5", ("uniform", 1.0, 4.6), "STANDARD", "avo_au_stages; continuous hidden value"),
    "ripening_rate": ("stage/d", ("normal", 0.55, 0.10, 0.3, 0.8), "SOURCED-VERIFY",
                      "ashton2006: firm-ripe ~day 8, soft-ripe ~9.5 at 20 C; landahl: 304 -> 2 N in 6 d at 18 C"),
    "penetrometer_force": ("N", "stage 1..5 = 118, 46, 15, 8.3, 4.9 N (geometric band centres), log-linear",
                           "SOURCED", "avo_au_stages: hard >10, rubbery 2.2-10, breaking 1.1-2.1, firm-ripe 0.7-1.0, "
                                      "ripe 0.4-0.6 kgf (11.1 mm tip, 8 mm)"),
    "flesh_E": ("MPa", "E = 2.29 (F_pen / 118 N)^0.837 -> 2.29 .. 0.16 MPa", "SOURCED",
                "farinas2021: 2.29+-0.75 -> 0.16+-0.08 MPa over ripening (endpoints); mapping ILLUSTRATIVE"),
    "flesh_poisson": ("1", 0.45, "ILLUSTRATIVE", "near-incompressible soft tissue"),
    "bruise_force_ratio": ("1", 0.6, "SOURCED-VERIFY",
                           "hia2018: a 10 N thumb press bruises firm-ripe fruit (visible within 48 h); ratio of thumb "
                           "pressure to penetrometer pressure at stage 4"),
    "bruise_visible_after": ("h", 24.0, "SOURCED", "hia2018: bruises take ~24 h to show"),
    "friction_skin_rubber": ("1", ("uniform", 0.6, 0.9), "ILLUSTRATIVE", "pebbled skin"),
    "skin_colour_Lab": ("L*,C*,h", "L 40->27, C 15->3, h 125->250 over ripening (whole fruit)", "SOURCED", "ashton2006"),
    "flesh_bands_Lab": ("L*,h", "three equal bands: dark green L 74->64 h 113; pale green L 83->75 h 105; yellow "
                        "L 83->75 h 101; C 43->35", "SOURCED", "ashton2006"),
    "vascular_strands": ("1", 6, "SOURCED", "cs42: pedicel cylinder splits into 6 strands, join near the apex and "
                                            "enter the seed coat as one"),
    "squeeze_stiffness_model": ("N/mm", "Isaac FEM table (test_reports/avocado_squeeze.json): k(4 N) 7.0 / 5.4 / 3.3 / "
                                "2.4 / 2.1 / 1.8 for stages 1 / 2 / 3 / 4 / 4.5 / 5", "DERIVED",
                                "convergence (1000 Hz/120 it vs 2000 Hz/200 it): stages >= 3 within 10 %; stages 1-2 "
                                "under-resolved (stage 1: 7.0 vs 11.9 N/mm) -> unripe fruit is stiffer than the table"),
    "stem_button_length": ("mm", ("uniform", 2.0, 4.0), "SOURCED-VERIFY", "alder: ~3 mm clipped button"),
    # ---- processing: cutting round the seed, twisting apart, removing the seed (food/avocado_process.py)
    "process_protocol": ("-", "cut lengthwise round the seed, twist the halves apart, remove the seed with a spoon "
                         "(knife strike discouraged)", "SOURCED", "afm_howto; farley2019 (knife injuries when pitting)"),
    "skin_puncture_force": ("N", ("uniform", 6.0, 15.0), "ILLUSTRATIVE",
                            "GAP: no knife-puncture data for Hass skin held; sharp chef's knife at the stem end"),
    "skin_cut_resistance": ("N/mm", ("uniform", 1.5, 3.0), "ILLUSTRATIVE",
                            "force per mm of skin thickness while slicing along the perimeter; GAP"),
    "flesh_cut_toughness_unripe": ("J/m2", 300.0, "ILLUSTRATIVE",
                                   "analogue: fruit-flesh cutting toughness 0.1-0.4 kJ/m2 (apple, potato); scaled with "
                                   "sqrt(E / E_unripe) over ripening"),
    "seed_adhesion_unripe": ("kPa", 20.0, "ILLUSTRATIVE",
                             "seed-flesh interface shear strength, scaled with (E / E_unripe)^0.7; GAP"),
    "clean_twist_stage": ("1..5", 2.8, "ILLUSTRATIVE",
                          "centre of the clean-separation transition: unripe flesh clings to the seed (twin "
                          "AvocadoTruth.halve: clean from ~3; afm_howto: ripe fruit twists apart)"),
    "testa_remnant_fraction": ("1", ("uniform", 0.4, 1.0), "ILLUSTRATIVE",
                               "share of the seed coat left in the cavity when the testa adheres to the flesh (cs42)"),
    "knife_blade_thickness": ("mm", 1.6, "TYPICAL", "chef's knife spine 2-2.5 mm, edge region ~1-2 mm"),
    # ---- lifetime: storage, ripe window, decline, browning of cut flesh (food/avocado_lifetime.py)
    "storage_temperatures": ("C", "counter 20, fridge 4 (ripe fruit 2-4 C; softening fruit 5-8 C up to 10 d)",
                             "SOURCED", "ucd_avocado; marques2011"),
    "ripening_q10": ("1", 2.5, "ILLUSTRATIVE", "temperature factor of the ripening rate per 10 C (respiration-type "
                     "Q10); no avocado-specific value held"),
    "ripe_window_counter": ("h", ("uniform", 24.0, 48.0), "SOURCED-VERIFY",
                            "ripe_window_guides: stage 5 lasts ~1-2 d at room temperature"),
    "decline_per_day_counter": ("1/d", 0.6, "ILLUSTRATIVE", "eating-quality loss per day past the ripe window "
                                "(overripe: soft, stringy, brown spots), slowed by the fridge with ripening_q10"),
    "spoil_days_past_window": ("d", ("uniform", 1.0, 3.0), "ILLUSTRATIVE",
                               "days past the ripe window at 20 C until the fruit is unusable (rot, flesh browning)"),
    "browning_rate_cut": ("dE/sqrt(h)", 3.0, "ILLUSTRATIVE", "enzymatic browning of exposed flesh at 20 C, dE ~ k sqrt(t) "
                          "(PPO, oxygen-limited); apple slices dE 2.4-4 within 1 h (pmc7022590) as an analogue"),
    "browning_visible_dE": ("dE", 5.0, "ILLUSTRATIVE", "colour difference at which the cut face looks brown"),
    "browning_treatment_factors": ("1", "none 1.0, lemon juice 0.45, plastic wrap 0.35, wrap + fridge 0.12, "
                                   "vacuum + fridge 0.03; seed left in: no effect on the cut face", "ILLUSTRATIVE",
                                   "ordering from atk_browning; factors are assumptions"),
    "peel_coat_success": ("1", 0.85, "ILLUSTRATIVE", "chance that one pass with the spoon tip lifts the seed-coat "
                          "sheets left in the cavity"),
}


ROLL = {
    "product": ("-", "Weizenbroetchen, oval, one lengthwise score ('Schnittbroetchen')", "STANDARD",
                "leitsaetze_1993: >= 90 % wheat flour; testde_2021: the score raises the crust share"),
    "length": ("mm", ("normal", 102.0, 5.0, 90, 115), "ILLUSTRATIVE",
               "oval single-score roll; round crispy roll ~90 mm (hirte_2012), round buns 86-92 mm (musiienko_2025)"),
    "width": ("mm", ("normal", 80.0, 4.0, 70, 90), "ILLUSTRATIVE", "chosen with length/height for ~50 g at ~0.2 g/cm3"),
    "height": ("mm", ("normal", 52.0, 3.0, 44, 60), "SOURCED", "hirte_2012 ~50 mm; musiienko_2025 50-55 mm"),
    "mass_check": ("g", "50-90 (bake-at-home rolls, testde_2021); RMT ~55 g dough per roll (ireks_rmt)", "SOURCED", ""),
    "base_edge_height": ("mm", ("uniform", 6.0, 10.0), "ILLUSTRATIVE", "rounded base edge"),
    "lumpiness": ("1", ("uniform", 0.01, 0.03), "ILLUSTRATIVE", "hand-moulded irregularity"),
    "score_length_frac": ("1", ("uniform", 0.70, 0.85), "ILLUSTRATIVE", "share of the long axis"),
    "score_width": ("mm", ("uniform", 14.0, 22.0), "ILLUSTRATIVE", "opened 'Ausbund' width"),
    "score_depth": ("mm", ("uniform", 4.0, 8.0), "ILLUSTRATIVE", ""),
    "crust_thickness_top": ("mm", ("normal", 1.3, 0.2, 0.9, 2.0), "SOURCED", "hirte_2012: cut-off crust 1.0-1.5 mm"),
    "crust_thickness_score": ("mm", ("uniform", 0.6, 0.9), "ILLUSTRATIVE", "thinner, paler Ausbund crust"),
    "crust_thickness_bottom": ("mm", ("uniform", 1.2, 1.8), "ILLUSTRATIVE", "GAP: no bottom-crust measurement held"),
    "transition_thickness": ("mm", ("uniform", 1.0, 2.0), "ILLUSTRATIVE",
                             "soleimani_2014 distinguishes an inner-crust colour band; thickness GAP"),
    "crumb_density": ("kg/m3", ("normal", 180.0, 15.0, 150, 220), "DERIVED",
                      "roll specific volume 5.9-6.4 ml/g (kurek_2016) -> whole ~0.16-0.17 g/cm3, minus the crust share;"
                      " crumb 0.088-0.252 (zghal_1999, SOURCED-VERIFY)"),
    "crust_density": ("kg/m3", ("normal", 320.0, 30.0, 260, 400), "TYPICAL", "crust 50-100 % denser than crumb"),
    "transition_density": ("kg/m3", 240.0, "ILLUSTRATIVE", "between crust and crumb"),
    "crumb_porosity": ("1", ("uniform", 0.73, 0.81), "SOURCED", "hirte_2012 uCT"),
    "cell_diameter_median": ("mm", ("normal", 0.70, 0.08, 0.5, 0.9), "DERIVED",
                             "lagrain_2012 mean 2D cell area 0.32 mm2 -> equivalent diameter 0.64 mm; 2D under-"
                             "estimates (x1.5 volume, rathnayake_2018)"),
    "cell_diameter_gsd_log": ("1", 0.55, "ILLUSTRATIVE", "lognormal spread of cell size (2-2189e3 um3 range, hirte)"),
    "cell_number_density": ("1/cm3", 1200.0, "DERIVED", "83 cells/cm2 (lagrain_2012) / ~0.07 cm mean diameter"),
    "cell_elongation": ("1", ("uniform", 1.2, 1.5), "ILLUSTRATIVE", "cells stretched in the rise direction"),
    "crust_L": ("L*", ("normal", 52.0, 4.0, 42, 62), "SOURCED-VERIFY",
                "pan-bread crust 47.6 (chaple_2023); roll browning index 50.4 -> L* ~50 (kurek_2016)"),
    "crust_C": ("C*", 36.0, "SOURCED-VERIFY", "chaple_2023 a* 17.05, b* 32.2 -> C* 36.4"),
    "crust_h": ("deg", 62.0, "SOURCED-VERIFY", "chaple_2023 -> h 62"),
    "crumb_L": ("L*", 74.0, "SOURCED-VERIFY", "pan-bread crumb 72.06 (chaple_2023); white roll slightly lighter"),
    "crumb_C": ("C*", 15.0, "SOURCED-VERIFY", "chaple_2023 a* 0.94, b* 15.37"),
    "crumb_h": ("deg", 86.0, "SOURCED-VERIFY", "chaple_2023"),
    "variant": ("-", ("choice_str", "plain", "plain", "dusted"), "ILLUSTRATIVE", "flour-dusted or plain"),
    "slice_height_frac": ("1", 0.45, "TYPICAL", "wikihow_slice, tastingtable_bagel: horizontally through the middle"),
    "friction_crust_rubber": ("1", ("uniform", 0.5, 0.8), "ILLUSTRATIVE", ""),
    # ---- hidden state and dynamics
    "age_h": ("h", ("choice", 0.5, 2.0, 6.0, 20.0, 30.0), "ILLUSTRATIVE", "time since baking (fresh .. 'vom Vortag')"),
    "storage": ("-", ("choice_str", "paper_bag", "paper_bag", "plastic_bag", "open"), "ILLUSTRATIVE",
                "oekotest_storage: paper bag recommended; plastic softens the crust"),
    "crumb_E_kPa_by_age": ("kPa @ h", "20 @ 5, 70 @ 48, 125 @ 96, 157 @ 168", "SOURCED-VERIFY",
                           "lagrain_2012 pan bread, read off figures; log-time interpolation"),
    "crumb_poisson": ("1", 0.103, "SOURCED", "lagrain_2012"),
    "crust_water_model": ("%", "w(t) = 9.73 t_h^0.112 (40 % RH, 22 C)", "CALIBRATED",
                          "power law through hirte_2012 9.0 % @ 0.5 h and 13.9 % @ 24 h; predicts 10.6 % @ 2.1 h "
                          "(measured: 10.1 % critical reached at 2.1 h) -> residual +0.5 points"),
    "crust_water_critical": ("%", 10.1, "SOURCED", "hirte_2012: half the sensory crispness lost"),
    "plastic_bag_factor": ("1", 4.0, "ILLUSTRATIVE", "crust softening clock runs x4 in a closed bag (oekotest: "
                                                      "'plastic softens the crust'; no rate held)"),
    "crust_break_force": ("N", 3.0, "SOURCED", "hirte_2012: >= 3 N with a 1 x 10 mm blade"),
    "avrami_firmness": ("N", "F = 66.01 - 59.53 exp(-0.186 t_d)", "SOURCED", "cardone_2020 loaf (n=1, k per day)"),
    "reheat_oven": ("C, min", "150-170, 5-10", "TYPICAL", "utopia_aufbacken"),
    "toaster_rack": ("min", "2-3 + flip + 1-2", "SOURCED-VERIFY", "utopia_aufbacken"),
    "toast_times": ("min", "3 light .. 4.5 'starting to burn' (sliced bread, home toaster)", "SOURCED", "kaur_2026"),
    "saw_setpoint": ("N, m/s, m", "-30 N, 0.02 m/s, +-0.02 m strokes (baguette)", "SOURCED", "luo_2026 (robot)"),
    "crumb_squash_strain": ("1", 0.30, "ILLUSTRATIVE", "permanent crumb compression when pressing instead of sawing"),
    "squeeze_k_model": ("N/mm", "k = 0.29 E / 20 kPa (30 mm pads across the width, 1 -> 3 N)", "DERIVED",
                        "Isaac FEM (test_reports/roll_squeeze_summary.json): 20 kPa lane converged (0.276 -> 0.29 N/mm, "
                        "1000 -> 2000 Hz); stiffer lanes under-converged -> linear-elastic scaling; crust excluded (GAP)"),
}

CARROT = {
    "type": ("-", "Nantes (cylindrical, blunt tip, small core), loose, topped", "STANDARD", "eu730_1999; con22"),
    "length": ("mm", ("normal", 190.0, 15.0, 150, 230), "SOURCED", "con22: 182.7 / 210.8 mm (Nantes)"),
    "shoulder_diameter": ("mm", ("normal", 31.0, 2.0, 25, 38), "SOURCED", "con22: 30.3 / 31.4 mm"),
    "mass_check": ("g", "110-135 (con22 Nantes); >= 50 g main crop (eu730_1999)", "SOURCED", ""),
    "tip_bluntness": ("1", ("uniform", 0.3, 0.9), "ILLUSTRATIVE", "Nantes 'blunt tip' (catalogues)"),
    "bend_radius": ("mm", ("uniform", 600.0, 3000.0), "ILLUSTRATIVE", "slight curvature"),
    "periderm_thickness": ("um", 150.0, "ILLUSTRATIVE", "GAP (knowles & flore 1983 to obtain); ~3 cells of ~50 um"),
    "xylem_radius_fraction": ("1", ("normal", 0.33, 0.04, 0.25, 0.42), "SOURCED-VERIFY",
                              "singh17 ~0.38 over mixed genotypes; Nantes bred for a small core; vob24 trend "
                              "crown -> tip; peng14 4-7 mm core"),
    "primary_xylem_diameter": ("mm", 1.0, "ILLUSTRATIVE", ""),
    "petiole_stubs": ("1", ("int_uniform", 4, 9), "ILLUSTRATIVE", ""),
    "petiole_stub_length": ("mm", ("uniform", 2.0, 6.0), "ILLUSTRATIVE", "topped carrots"),
    "green_shoulder_length": ("mm", ("choice", 0.0, 0.0, 0.0, 6.0, 12.0), "STANDARD",
                              "eu730_1999 Class I <= 20 mm on long roots (instances drawn inside Class I)"),
    "lateral_root_rows": ("1", 4, "SOURCED-VERIFY", "diarch_rows"),
    "lateral_root_scar_spacing": ("mm", ("uniform", 8.0, 15.0), "ILLUSTRATIVE", ""),
    "lenticel_band_spacing": ("mm", ("uniform", 2.0, 4.0), "ILLUSTRATIVE", "GAP: no spacing found"),
    "lenticel_band_relief": ("mm", ("uniform", 0.05, 0.12), "ILLUSTRATIVE", ""),
    "phloem_LCh": ("L*,C*,h", (55.0, 49.0, 54.0), "SOURCED-VERIFY", "pac16_dry raw flesh 55.35/28.9/40.05"),
    "xylem_LCh": ("L*,C*,h", (63.0, 45.0, 64.0), "ILLUSTRATIVE", "paler, yellower core (lutein; aub22) -- no L*a*b*"),
    "skin_LCh": ("L*,C*,h", (50.0, 44.0, 52.0), "ILLUSTRATIVE", "duller than the flesh"),
    "soil_residue": ("1", ("uniform", 0.0, 0.25), "ILLUSTRATIVE", "washed supermarket carrots"),
    "friction_skin_rubber": ("1", ("uniform", 0.5, 0.8), "ILLUSTRATIVE", "moist skin"),
    "density": ("kg/m3", 1040.0, "SOURCED-VERIFY", "jah18 1.04 g/cm3 (1.00-1.17)"),
    "E_phloem": ("MPa", ("normal", 4.6, 0.15, 4.33, 4.85), "SOURCED-VERIFY", "vob24 after Stopa 2017"),
    "E_xylem": ("MPa", ("normal", 4.75, 0.08, 4.62, 4.88), "SOURCED-VERIFY", "vob24 after Stopa 2017"),
    "poisson": ("1", 0.475, "SOURCED-VERIFY", "vob24"),
    "E_decay_7d": ("1", 1.32, "SOURCED", "vob24: E / 1.32 after 7 d in the fridge (zip bag)"),
    "bend_Fmax": ("N", 48.6, "SOURCED-VERIFY", "jah18 three-point bending (span: see source)"),
    "bend_deflection": ("mm", 21.9, "SOURCED-VERIFY", "jah18"),
    "blade_shear": ("N", 41.1, "SOURCED-VERIFY", "jah18 straight blade 1.4 mm, 30 deg"),
    "impact_damage_force": ("N", "43-45", "SOURCED-VERIFY", "xia21"),
    "fracture_energy": ("J/m2", "40-400 (mixed produce)", "SOURCED-VERIFY", "alv00"),
    "turgor_fresh": ("MPa", 0.7, "SOURCED-VERIFY", "herp99_calbo00"),
    "limp_rwc": ("1", 0.95, "SOURCED-VERIFY", "herp99_calbo00: firmness from turgor ~0 below RWC 0.95"),
    "softening_kinetics": ("1/min", "2nd-order fractional conversion; k(80 C) 0.035, k(110 C) 1.453; Ea 138.9 kJ/mol;"
                           " Finf/F0 0.02", "SOURCED", "peng14"),
    "peel_depth": ("mm", 2.0, "SOURCED", "con22 hand peeler; mass 12.1-14.6 %"),
    "water_loss_rate": ("%/d", "fridge in bag 0.3, room open 2.0", "ILLUSTRATIVE",
                        "storage_loss: unwrapped 20 C unacceptable after ~3 d at 5-8 % loss -> ~2 %/d; fridge GAP"),
    "age_d": ("d", ("uniform", 1.0, 14.0), "ILLUSTRATIVE", "since purchase; hidden"),
    "storage": ("-", ("choice_str", "fridge_bag", "fridge_bag", "fridge_open", "room_open"), "ILLUSTRATIVE", ""),
    "blush_reversible_h": ("h", 1.25, "SOURCED-VERIFY", "sim09_cz95"),
    "blush_irreversible_h": ("h", 15.0, "SOURCED-VERIFY", "sim09_cz95"),
    "boil_rounds_6mm": ("min", "4-5 crisp-tender, +1-2 soft", "SOURCED-VERIFY", "carrot_boil"),
    "cut_dims": ("mm", "rounds 5-6; julienne 3x3x45; brunoise 3; batonnet 6x6x55; dice 6/12/19", "TYPICAL",
                 "carrot_cuts"),
}

BANANA = {
    "cultivar": ("-", "Cavendish (Musa acuminata AAA; Grand Naine / Valery)", "SOURCED", "ciro05; mal22"),
    "length_outer": ("mm", ("normal", 210.0, 15.0, 170, 250), "SOURCED",
                     "convex-face length incl. pedicel (eu1333_2011 definition): 190 (mal22), 233 (ciro05)"),
    "calibre": ("mm", ("normal", 35.0, 1.5, 30, 40), "SOURCED", "ciro05 34/36/34 mm; eu1333 min 27 mm"),
    "mass_check": ("g", "102 (bant24), 105 (chiw23), 126-148 (polat19)", "SOURCED", ""),
    "curvature_radius": ("mm", ("uniform", 150.0, 260.0), "ILLUSTRATIVE", "'slightly curved' (mal22); GAP"),
    "ridges": ("1", ("choice", 4, 5, 5), "ILLUSTRATIVE", "'slightly ridged' (mal22), 'nearly 4-sided'; count GAP"),
    "angularity": ("1", ("uniform", 0.04, 0.08), "ILLUSTRATIVE", "ridges round out as the finger matures (fao)"),
    "peel_thickness": ("mm", ("normal", 2.6, 0.25, 2.0, 3.2), "SOURCED",
                       "mal22 2.56 mm (Grand Naine); amn12 1.1-3.8 mm, thinning on ripening"),
    "peel_thinning": ("1", 0.20, "ILLUSTRATIVE", "fractional thinning stage 1 -> 7 (amn12: thinning, no % held)"),
    "peel_layer_fractions": ("1", (0.15, 0.60, 0.25), "ILLUSTRATIVE",
                             "chlorenchyma / parenchyma / inner aerenchyma share of the peel (amn12 layer order)"),
    "pulp_peel_ratio_check": ("1", "1.69-2.69 (chiw23, bant24, mal22); rises with ripening", "SOURCED", ""),
    "central_axis_diameter": ("mm", 3.0, "ILLUSTRATIVE", "placental axis"),
    "stalk_length": ("mm", ("normal", 17.3, 1.5, 13, 22), "SOURCED", "mal22 pedicel"),
    "stalk_width": ("mm", ("normal", 10.15, 0.8, 8, 12.5), "SOURCED", "mal22 pedicel"),
    "phloem_strings": ("1", ("int_uniform", 3, 6), "ILLUSTRATIVE", "'strings' (guapo22); count GAP"),
    "ovules_total": ("1", ("int_uniform", 200, 300), "SOURCED", "promusa: 200-300 ovules"),
    "ovule_diameter": ("mm", 0.6, "ILLUSTRATIVE", "tiny black specks"),
    "ripeness_stage": ("1..7", ("uniform", 2.0, 7.0), "STANDARD", "markon / ma22 seven-stage colour scale"),
    "peel_hue_by_stage": ("deg", [118.0, 98.7, 97.3, 92.7, 89.4, 87.5, 82.0], "SOURCED-VERIFY",
                          "ringer21 R2..R7; stage 1 from col20 Gros Michel day 0 (h ~122) -> 118"),
    "peel_L_by_stage": ("L*", [48.0, 55.0, 62.0, 68.0, 72.0, 72.0, 64.0], "SOURCED-VERIFY",
                        "shape: hou15 L* peaks at stage 5 then falls; values: hega26 45-57 -> 55-73 -> 52-66, "
                        "shahir14 max 74"),
    "peel_C_by_stage": ("C*", [42.0, 50.0, 56.0, 60.0, 62.0, 62.0, 58.0], "SOURCED-VERIFY",
                        "ringer21 b* 56-72 (a* ~0 at R5 -> C* ~ b*); col20 green C* ~48"),
    "pulp_LCh": ("L*,C*,h", (86.0, 24.0, 92.0), "ILLUSTRATIVE", "'cream' (mal22); numeric GAP"),
    "sugar_spot_density": ("1", 1.0, "ILLUSTRATIVE", "stage 7 = brown flecks (markon); kinetics GAP"),
    "friction_peel_rubber": ("1", ("uniform", 0.5, 0.8), "ILLUSTRATIVE", "waxy peel"),
    "peel_density": ("kg/m3", 950.0, "ILLUSTRATIVE", "inner aerenchyma; whole fruit SG 1.023 (arcc, SV)"),
    "pulp_density": ("kg/m3", 1050.0, "ILLUSTRATIVE", "whole fruit SG 1.023 (arcc, SV)"),
    # ---- mechanics and ripening
    "pulp_E_compression": ("MPa", "1.75 (day 1) -> 1.09 (day 12)", "SOURCED", "ciro05 cylinders 20 x 40 mm"),
    "pulp_E_bending": ("MPa", "5.16 -> 2.65", "SOURCED", "ciro05"),
    "pulp_firmness": ("N", "38.0 -> 14.0 over 10 d (2 mm probe)", "SOURCED", "jaiswal12; chiw23 44.8 -> 22.1"),
    "peel_puncture": ("N", "13.6-15.2 (2 mm probe)", "SOURCED", "ciro05"),
    "peel_fracture": ("N", "53.0 -> 2.1 on ripening", "SOURCED-VERIFY", "khanal23 (test geometry not in abstract)"),
    "whole_fracture": ("N", "129 bending (14 cm span), 121 compression", "SOURCED", "ciro05"),
    "finger_drop_force": ("N", "~49 on day 1, > 2x lower by day 3", "SOURCED-VERIFY", "imsabai; mbeguie09"),
    "bruise_threshold": ("mJ", 200.0, "SOURCED-VERIFY", "bugaud14: no visible bruise up to 200 mJ (Grande Naine)"),
    "peel_bond_ratios": ("1", (0.45, 0.30, 1.5), "ILLUSTRATIVE",
                         "peel-to-pulp adhesion / ridge seam / strip tear strength as multiples of the peel fracture "
                         "force (khanal23 53 -> 2.1 N, SOURCED-VERIFY); adhesion > seam so the peel splits along the "
                         "ridges (khanal23) instead of sliding off as a sleeve; GAP: no peeling-force study"),
    "stage_rate_20C": ("stage/d", ("normal", 0.6, 0.15, 0.3, 1.0), "SOURCED-VERIFY",
                       "hou15: 1 -> 7 in 7 d with ethylene (0.86/d); ma22 day bands ~0.4/d; 18 C RS6 at 9 d"),
    "q10_ripening": ("1", 2.0, "ILLUSTRATIVE", "temperature dependence of the stage rate"),
    "chilling_threshold": ("C", 13.0, "SOURCED", "wflo"),
    "brix_by_day": ("Bx", "4.5 / 6.8 / 8 / 16.7 / 23.7 at day 1 / 4 / 6 / 8 / 10", "SOURCED", "chiw23"),
    "mass_loss_rate": ("%/d", 1.0, "SOURCED-VERIFY", "chiw23 ~2.2 %/d (22-27 C, outlier-looking); kojima96 ~0"),
    "rounds_thickness": ("mm", 6.0, "TYPICAL", "banana_howto"),
}

ORANGE = {
    "cultivar": ("-", ("choice_str", "navel", "navel", "valencia"), "ILLUSTRATIVE",
                 "German supermarket: navel (winter) / Valencia (summer)"),
    "diameter": ("mm", ("normal", 75.0, 6.0, 61, 88), "SOURCED", "ucr_citrus Washington navel 61-88 mm"),
    "shape_index": ("1", ("uniform", 1.0, 1.08), "SOURCED", "ucr_citrus L/W 1.0-1.1"),
    "mass_check": ("g", "162-335 (navel), 113-299 (Valencia)", "SOURCED", "ucr_citrus"),
    "peel_thickness": ("mm", ("normal", 5.0, 0.6, 3.5, 6.4), "SOURCED", "ucr_citrus rind 4.1-6.4 / 3.5-6.2 mm"),
    "flavedo_thickness": ("mm", ("uniform", 1.0, 1.6), "ILLUSTRATIVE",
                          "peel minus albedo (2-5 mm, citrus_patents; 3.9 mm smith18)"),
    "core_diameter": ("mm", ("uniform", 5.0, 9.0), "ILLUSTRATIVE", "central axis; GAP"),
    "segments": ("1", ("choice", 9, 10, 10, 10, 11, 12), "SOURCED-VERIFY", "citrus_patents 9.6-11"),
    "membrane_thickness": ("mm", 0.12, "ILLUSTRATIVE", "endocarp film; GAP"),
    "seeds": ("1", ("int_uniform", 1, 6), "SOURCED", "ucr_citrus Valencia 1.2-6.6; navel ~0 (set in the model)"),
    "seed_length": ("mm", 9.0, "TYPICAL", ""),
    "oil_gland_density": ("1/cm2", 50.0, "SOURCED-VERIFY", "knight01 8,000-12,000 per fruit -> ~40-60 /cm2 (derived);"
                                                             " one cultivar 208 /cm2 (citrus_patents)"),
    "peel_density": ("kg/m3", 746.0, "SOURCED", "jentzsch22"),
    "albedo_density": ("kg/m3", 700.0, "ILLUSTRATIVE", "spongy with air spaces (sadka19)"),
    "pulp_density": ("kg/m3", 1045.0, "DERIVED", "juice SG 1.045 (orange_juice)"),
    "peel_LCh": ("L*,C*,h", (60.0, 74.0, 57.0), "ILLUSTRATIVE",
                 "chosen for CCI ~ +11 (cci market target +10..+20); measured L*a*b* GAP"),
    "albedo_LCh": ("L*,C*,h", (92.0, 10.0, 90.0), "ILLUSTRATIVE", "white; GAP"),
    "pulp_LCh": ("L*,C*,h", (66.0, 62.0, 66.0), "ILLUSTRATIVE", "'medium orange' (ucr_citrus); GAP"),
    "membrane_LCh": ("L*,C*,h", (84.0, 28.0, 80.0), "ILLUSTRATIVE", "translucent pale; GAP"),
    "friction_peel_rubber": ("1", ("uniform", 0.5, 0.8), "ILLUSTRATIVE", "waxy, pitted peel"),
    "flavedo_E": ("MPa", 103.0, "SOURCED", "smith18 (30-300 across species)"),
    "albedo_E": ("MPa", 0.79, "SOURCED", "smith18 (+-0.58)"),
    "peel_compressive_E": ("MPa", 0.020, "SOURCED-VERIFY", "jentzsch22 whole peel, compression"),
    "peel_poisson": ("1", -0.05, "SOURCED", "jentzsch22"),
    "punch_force": ("N", "15.5-79 (4.3-14.3 mm punches)", "SOURCED-VERIFY", "orange_mech"),
    "rupture_force": ("N", "140-221 (Valencia)", "SOURCED-VERIFY", "orange_mech"),
    "gland_damage_force": ("N", "10-250 (sphere; oil release)", "SOURCED-VERIFY", "orange_mech"),
    "oil_jet": ("m/s, um", "8.5 +- 4.0, 102 +- 20", "SOURCED", "smith18"),
    "gland_burst_pressure": ("kPa", "30-140", "SOURCED", "smith18"),
    "juice_fraction": ("%", ("uniform", 45.0, 54.0), "SOURCED", "ucr_citrus navel 45-54 (Valencia 37-56)"),
    "juice_by_hand": ("mL", "60-80 per medium orange", "TYPICAL", "orange_juice"),
    "brix": ("Bx", ("uniform", 9.3, 14.1), "SOURCED", "ucr_citrus"),
    "juice_viscosity": ("mPa s", 5.0, "SOURCED-VERIFY", "orange_juice 4-6 at 8 C"),
    "peelability": ("-", "navel medium-easy; Valencia hard (strips break); cold fruit harder", "SOURCED",
                    "ritenour13 (no N values)"),
    "peel_strip_force": ("N", "navel 4, Valencia 8 per 16 mm strip", "ILLUSTRATIVE", "ranking from ritenour13; GAP"),
    "storage": ("C, %RH", "3-8, 90-95, <= 3 months; chilling after 3-6 weeks at ~4 C", "SOURCED-VERIFY",
                "orange_storage"),
    "bitter_juice_h": ("h", 3.0, "SOURCED-VERIFY", "edis891 'within several hours' (navel)"),
    "age_d": ("d", ("uniform", 2.0, 20.0), "ILLUSTRATIVE", "since purchase"),
}

LOAF = {
    "product": ("-", "Toastbrot (Buttertoast), lidded tin, pre-sliced, 500 g", "SOURCED", "test25_toast; toast_pack"),
    "length": ("mm", ("normal", 250.0, 6.0, 235, 265), "SOURCED-VERIFY", "toast_pack ~0.25 m"),
    "width": ("mm", ("normal", 100.0, 3.0, 92, 110), "SOURCED-VERIFY", "tins 100 / 110 mm"),
    "tin_height": ("mm", ("normal", 98.0, 3.0, 88, 105), "SOURCED-VERIFY",
                   "square slices: 25 g x 0.19 g/cm3 / 12.5 mm -> ~100 x 100 mm (derived; tins 80-90 mm deep)"),
    "dome_height": ("mm", 0.0, "SOURCED", "test25_toast: baked under a lid (flat top)"),
    "edge_radius": ("mm", ("uniform", 8.0, 12.0), "ILLUSTRATIVE", "rounded tin corners"),
    "slices": ("1", 20, "SOURCED-VERIFY", "toast_pack"),
    "heels": ("1", ("choice", 0, 0, 0, 2), "SOURCED", "test25_toast: most packs no heels; organic 2"),
    "crust_thickness_top": ("mm", ("uniform", 0.8, 1.3), "ILLUSTRATIVE", "thin soft lidded crust; GAP"),
    "crust_thickness_tin": ("mm", ("uniform", 0.5, 0.9), "ILLUSTRATIVE", "GAP"),
    "crumb_density": ("kg/m3", ("normal", 190.0, 10.0, 170, 215), "SOURCED", "nuss90 0.19 g/cm3"),
    "crust_density": ("kg/m3", 300.0, "ILLUSTRATIVE", ""),
    "cell_diameter_median": ("mm", 0.5, "ILLUSTRATIVE", "finer, more even than a roll; C-Cell values GAP"),
    "cell_number_density": ("1/cm3", 2500.0, "ILLUSTRATIVE", ""),
    "cell_elongation": ("1", 1.15, "ILLUSTRATIVE", ""),
    "crumb_LCh": ("L*,C*,h", (74.7, 15.5, 88.6), "SOURCED-VERIFY", "ishida14 74.73 / 0.37 / 15.51"),
    "crust_LCh": ("L*,C*,h", (62.0, 34.0, 70.0), "ILLUSTRATIVE",
                  "pale lidded crust; tin loaves bottom out at L* ~64.5 at 200 C (giovanelli_2021, derived)"),
    "friction_bread": ("1", ("uniform", 0.6, 0.9), "ILLUSTRATIVE", ""),
    "firmness_by_day": ("N", "1.98 / 3.62 / 4.18 / 5.64 at day 1 / 5 / 9 / 13 (25 C, bagged)", "SOURCED",
                        "sehn20 lab white pan loaf"),
    "tensile_failure": ("N", "0.31-0.39 (12 mm neck) at 9.2-9.7 mm", "SOURCED", "nuss90"),
    "best_before_d": ("d", "12-20", "SOURCED", "test25_toast"),
    "fridge_staling_factor": ("1", 6.0, "TYPICAL", "toast_misc (popular, weak)"),
    "toast_time": ("s", "80 .. > 150 to 'best' (per toaster)", "SOURCED", "toaster26"),
    "butter_load": ("g/cm2", 0.068, "SOURCED-VERIFY", "toast_misc"),
    "age_d": ("d", ("uniform", 0.5, 8.0), "ILLUSTRATIVE", "since baking"),
    "storage": ("-", ("choice_str", "bag_closed", "bag_closed", "bag_open", "fridge"), "ILLUSTRATIVE", ""),
}

POTATO = {
    "cultivar": ("-", ("choice_str", "Linda", "Belana", "Gala"), "STANDARD", "bsl21"),
    "cooking_type": ("-", ("choice_str", "festkochend", "festkochend", "vorwiegend festkochend"), "SOURCED-VERIFY",
                     "Belana / Linda festkochend, Gala vorwiegend festkochend"),
    "length": ("mm", ("normal", 85.0, 10.0, 60, 110), "SOURCED-VERIFY", "75 / 65 / 99 mm (Astrix / Diamont / Santana)"),
    "width": ("mm", ("normal", 55.0, 4.0, 45, 65), "SOURCED-VERIFY", "48 / 45 / 58 mm"),
    "thickness_ratio": ("1", ("uniform", 0.80, 0.92), "SOURCED-VERIFY", "thickness / width 0.81-0.89"),
    "mass_check": ("g", "64-167 (examples); min square mesh 35 / 30 mm (repealed HKV)", "SOURCED-VERIFY", ""),
    "lumpiness": ("1", ("uniform", 0.02, 0.05), "ILLUSTRATIVE", ""),
    "eyes": ("1", ("int_uniform", 8, 14), "SOURCED-VERIFY", "Norgold 10-12 buds; 6-9 apical (mag25)"),
    "eye_crowding": ("1", 1.8, "ILLUSTRATIVE", "apical bud density ~7x lateral (mag25)"),
    "phyllotaxis": ("turn", 0.4, "ILLUSTRATIVE", "spiral arrangement sourced; the 2/5 fraction is a GAP"),
    "eye_depth": ("mm", ("uniform", 0.6, 1.4), "ILLUSTRATIVE", "bsl21 'flach' (ordinal)"),
    "eye_width": ("mm", ("uniform", 3.0, 4.0), "ILLUSTRATIVE", ""),
    "periderm_thickness": ("um", ("normal", 65.0, 10.0, 46, 84), "SOURCED", "mc20 phellem"),
    "lenticel_density": ("1/cm2", ("uniform", 5.0, 12.0), "SOURCED", "mag25"),
    "vascular_ring_depth": ("mm", ("uniform", 3.0, 6.0), "ILLUSTRATIVE", "GAP (Reeve 1969 paywalled)"),
    "pith_fraction": ("1", ("uniform", 0.35, 0.5), "ILLUSTRATIVE", "GAP"),
    "pith_branch_width": ("mm", ("uniform", 1.5, 2.5), "ILLUSTRATIVE", "star of the pith; GAP"),
    "density": ("kg/m3", ("normal", 1070.0, 12.0, 1050, 1100), "SOURCED-VERIFY", "1053 (kk18); SG 1.070-1.103"),
    "skin_LCh": ("L*,C*,h", (62.0, 30.0, 75.0), "ILLUSTRATIVE", "yellow skin, L* > 55 typical (SV); GAP"),
    "flesh_LCh": ("L*,C*,h", (80.0, 30.0, 92.0), "ILLUSTRATIVE", "medium-yellow (bsl21), L* > 75 (SV)"),
    "soil_residue": ("1", ("uniform", 0.0, 0.3), "ILLUSTRATIVE", ""),
    "friction_skin_rubber": ("1", ("uniform", 0.5, 0.8), "ILLUSTRATIVE", ""),
    "E": ("MPa", ("uniform", 3.1, 5.6), "SOURCED-VERIFY", "potato_mech"),
    "poisson": ("1", 0.45, "SOURCED-VERIFY", "potato_mech (Zhao 2023)"),
    "fracture_stress": ("MPa", "1.13-1.40", "SOURCED-VERIFY", "potato_mech"),
    "thermal": ("W/mK, kJ/kgK, m2/s", "k 0.55, cp 3.43, alpha 0.15e-6", "SOURCED", "kk18"),
    "gelatinisation": ("C", "onset 61, peak 67.5, end 76-82", "SOURCED-VERIFY", "potato_cook"),
    "softening": ("1/min, kJ/mol", "first order, k(100 C) 0.28-0.40, Ea 103-145", "SOURCED", "potato_cook (Harada)"),
    "k100": ("1/min", ("uniform", 0.28, 0.40), "SOURCED", "potato_cook"),
    "Ea": ("kJ/mol", 133.0, "SOURCED", "potato_cook (Harada 133-145)"),
    "boil_times": ("min", "diced 15-20, large whole up to 60, whole 20-25 (cold start)", "SOURCED", "gritzer"),
    "peel_depth": ("mm", 1.1, "SOURCED-VERIFY", "potato_peel (~9 % waste)"),
    "storage_temp": ("C", "7-13 dark; < 8 sweetens", "SOURCED-VERIFY", "potato_storage"),
    "greening_days_20C": ("d", 1.5, "SOURCED-VERIFY", "potato_storage 1-2.5 d under retail light"),
    "age_d": ("d", ("uniform", 2.0, 30.0), "ILLUSTRATIVE", ""),
    "light_exposure_h": ("h", ("choice", 0.0, 0.0, 0.0, 24.0, 48.0), "ILLUSTRATIVE", "hours under light"),
}

TOMATO = {
    "type": ("-", "round / truss tomato, 80-120 g", "STANDARD", "fdc_tomato"),
    "diameter": ("mm", ("normal", 63.0, 4.0, 52, 76), "SOURCED", "fdc_tomato 61-66 mm for 91-123 g"),
    "height_ratio": ("1", ("uniform", 0.75, 0.90), "SOURCED-VERIFY", "ince16 0.75; GRIN 'slightly flattened'"),
    "mass_check": ("g", "91 / 123 (small / medium)", "SOURCED", "fdc_tomato"),
    "locules": ("1", ("choice", 2, 3, 3, 4), "SOURCED", "ucd_rost: small types 2, globe 3-5"),
    "skin_thickness": ("um", ("uniform", 120.0, 250.0), "ILLUSTRATIVE",
                       "cuticle 9.9-15.7 um (bn05) + epidermis + collenchymatous hypodermis; exocarp 9.6 % FW (meta19)"),
    "pericarp_thickness": ("mm", ("normal", 6.5, 0.8, 5.0, 9.0), "SOURCED-VERIFY", "meta19 5.3 mm (small); 7-12 germplasm"),
    "septum_thickness": ("mm", ("uniform", 2.5, 3.5), "ILLUSTRATIVE", ""),
    "columella_diameter": ("mm", ("uniform", 10.0, 16.0), "ILLUSTRATIVE", "columella + placenta 10 % FW (meta19)"),
    "placenta_width": ("mm", ("uniform", 3.0, 5.0), "ILLUSTRATIVE", ""),
    "seeds_per_fruit": ("1", ("int_uniform", 50, 120), "SOURCED-VERIFY", "68.8 (one cultivar)"),
    "seed_length": ("mm", 3.5, "ILLUSTRATIVE", "GAP"),
    "rib_depth": ("1", ("uniform", 0.01, 0.03), "ILLUSTRATIVE", ""),
    "sepals": ("1", ("choice", 5, 6), "ILLUSTRATIVE", "GAP"),
    "sepal_length": ("mm", ("uniform", 12.0, 20.0), "ILLUSTRATIVE", ""),
    "pedicel_length": ("mm", ("choice", 0.0, 12.0, 15.0), "ILLUSTRATIVE", "truss tomatoes keep calyx and pedicel"),
    "ripeness_stage": ("1..6", ("uniform", 3.5, 6.0), "STANDARD", "cfr_tomato colour classes"),
    "skin_hue_by_stage": ("deg", [118.0, 110.0, 96.0, 71.0, 52.0, 43.0], "DERIVED",
                          "h = atan2(1, a*/b*) at the band centres of ince16; ripe ~40 deg (Radzevicius 2009, SV)"),
    "skin_L_by_stage": ("L*", [56.0, 56.0, 53.0, 49.0, 45.0, 42.0], "ILLUSTRATIVE", "L* falls on ripening (rad08); GAP"),
    "skin_C_by_stage": ("C*", [38.0, 36.0, 40.0, 50.0, 60.0, 66.0], "ILLUSTRATIVE", "GAP"),
    "flesh_LCh_green": ("L*,C*,h", (70.0, 30.0, 110.0), "ILLUSTRATIVE", "GAP"),
    "flesh_LCh_breaker": ("L*,C*,h", (80.0, 22.0, 85.0), "ILLUSTRATIVE", "pale interior at breaker; GAP"),
    "flesh_LCh_red": ("L*,C*,h", (48.0, 58.0, 36.0), "ILLUSTRATIVE", "GAP"),
    "flesh_density": ("kg/m3", 1000.0, "ILLUSTRATIVE", "GAP (cherry 1194, suspect)"),
    "gel_density": ("kg/m3", 1010.0, "ILLUSTRATIVE", ""),
    "friction_skin_rubber": ("1", ("uniform", 0.4, 0.7), "ILLUSTRATIVE", "smooth waxy skin"),
    "plate_firmness": ("N/mm", "initial G/P/R 15.1 / 6.0 / 4.5; average 13.2 / 4.7 / 3.1", "SOURCED-VERIFY", "ince16"),
    "skin_rupture_2mm": ("N", "G/P/R 6.92 / 3.99 / 2.74", "SOURCED-VERIFY", "ince16"),
    "skin_E": ("MPa", "252-293 (red); failure <= 25-28 MPa at 9-10 %", "SOURCED", "bn05"),
    "damage_strain": ("1", 0.03, "SOURCED-VERIFY", "tomato_mech (Li 2017 FEM)"),
    "bio_yield_force": ("N", 2.57, "SOURCED-VERIFY", "tomato_mech"),
    "stage_rate_20C": ("stage/d", ("uniform", 0.35, 0.7), "SOURCED-VERIFY", "breaker -> full colour 7-14 d (patent)"),
    "chilling_threshold": ("C", 12.5, "SOURCED", "wflo_tomato 12.5-12.8"),
    "fridge_volatile_loss": ("%", "65 after 8 d at 5 C", "SOURCED", "zhang16"),
    "water": ("%", 94.5, "SOURCED", "fdc_tomato"),
    "slice_thickness": ("mm", "6-13", "SOURCED", "tomato_howto"),
    "blanch_s": ("s", "30-45 (20-60)", "SOURCED", "tomato_howto"),
}

ONION = {
    "type": ("-", "yellow storage onion (Haushaltszwiebel), cured", "STANDARD", "hb66_onion"),
    "diameter": ("mm", ("normal", 65.0, 7.0, 50, 85), "SOURCED", "hb66_onion medium 51-83 mm; bah04 51.7"),
    "shape_index": ("1", ("uniform", 0.85, 1.10), "SOURCED-VERIFY", "height / diameter: bah04 57.1 / 51.7; flat-round "
                                                                     "German types"),
    "mass_check": ("g", "78.7 +- 24 (51.7 mm)", "SOURCED-VERIFY", "bah04"),
    "n_fleshy": ("1", ("choice", 7, 8, 8), "SOURCED-VERIFY", "kim_heb20 7.6 +- 0.2"),
    "scale_thickness_outer": ("mm", ("uniform", 3.5, 4.5), "ILLUSTRATIVE",
                              "kim_heb20 5.5 mm (bulk mean), wu25 ~3 mm; constrained by count x radius"),
    "scale_thickness_inner": ("mm", ("uniform", 2.0, 2.8), "ILLUSTRATIVE", "inner scales thinner; GAP"),
    "scale_gap": ("mm", 0.05, "ILLUSTRATIVE", "epidermis interface between scales"),
    "n_dry": ("1", ("choice", 2, 3), "ILLUSTRATIVE", "GAP (ordinal skin-thickness classes only)"),
    "dry_thickness": ("mm", ("uniform", 0.08, 0.15), "ILLUSTRATIVE", "GAP"),
    "plate_diameter": ("mm", ("uniform", 15.0, 22.0), "ILLUSTRATIVE", "GAP"),
    "plate_thickness": ("mm", ("uniform", 4.0, 6.0), "ILLUSTRATIVE", "GAP"),
    "n_roots": ("1", ("int_uniform", 15, 30), "ILLUSTRATIVE", ""),
    "root_length": ("mm", ("uniform", 3.0, 10.0), "ILLUSTRATIVE", "trimmed root crown"),
    "neck_length": ("mm", ("uniform", 5.0, 20.0), "ILLUSTRATIVE", ""),
    "neck_diameter": ("mm", ("uniform", 7.4, 10.9), "SOURCED-VERIFY", "onion_misc (cured neck, Indian cultivars)"),
    "flesh_density": ("kg/m3", 1040.0, "SOURCED-VERIFY", "bah04 bulb 1.04 g/cm3"),
    "dry_density": ("kg/m3", 400.0, "ILLUSTRATIVE", "papery skin"),
    "plate_density": ("kg/m3", 1050.0, "ILLUSTRATIVE", ""),
    "skin_LCh": ("L*,C*,h", (52.0, 45.0, 65.0), "ILLUSTRATIVE", "golden brown; GAP"),
    "flesh_LCh": ("L*,C*,h", (88.0, 12.0, 95.0), "ILLUSTRATIVE", "white to pale yellow; GAP"),
    "friction_skin_rubber": ("1", ("uniform", 0.6, 1.0), "SOURCED-VERIFY", "bah04 0.67-1.34"),
    "bulb_E": ("MPa", "2.03-5.45, nu 0.26-0.45", "SOURCED-VERIFY", "jafari16"),
    "tissue_E": ("MPa", "2.2 skin on / 1.4 off", "SOURCED", "wu25"),
    "epidermis_E": ("MPa", "3.5-8.0", "SOURCED", "wei01"),
    "crushing_load": ("N", "400-780", "SOURCED-VERIFY", "bah04"),
    "rolling_angle": ("deg", "20-31 on base, 14-23 on side", "SOURCED-VERIFY", "bah04"),
    "lf_window": ("min", "peak ~1, gone ~10", "SOURCED", "lokke12"),
    "storage": ("C, %RH", "0, 65-75, 6-9 months; household 3-4 weeks", "SOURCED", "hb66_onion"),
    "dice_spacing": ("mm", "mince <=3, fine 3-6, medium 6-13, coarse 13-19", "SOURCED", "onion_howto"),
    "caramelise_min": ("min", "30-45", "TYPICAL", "onion_misc"),
    "age_d": ("d", ("uniform", 3.0, 40.0), "ILLUSTRATIVE", ""),
}

LEMON = {
    "cultivar": ("-", ("choice_str", "Primofiori", "Primofiori", "Verna"), "SOURCED", "agu20"),
    "diameter": ("mm", ("normal", 61.0, 3.0, 55, 67), "SOURCED", "agu20 55-67 mm"),
    "shape_index": ("1", ("uniform", 1.3, 1.55), "SOURCED", "length / diameter from agu20 ED/PD 0.6-0.8 (body without"
                                                           " nipple)"),
    "nipple_height": ("mm", ("uniform", 3.0, 7.0), "ILLUSTRATIVE", "stylar mammilla; GAP"),
    "neck_height": ("mm", ("uniform", 1.5, 4.0), "ILLUSTRATIVE", "stem-end neck; GAP"),
    "mass_check": ("g", "100-184 (agu20); 166 (jen22_lemon)", "SOURCED", ""),
    "peel_thickness": ("mm", ("normal", 5.6, 0.5, 4.9, 7.1), "SOURCED", "agu20, jen22_lemon"),
    "flavedo_thickness": ("mm", ("uniform", 1.0, 1.4), "SOURCED-VERIFY", "jen22_lemon ~1 mm (ben26: ~46 % of peel)"),
    "core_diameter": ("mm", ("uniform", 4.0, 7.0), "ILLUSTRATIVE", "GAP"),
    "segments": ("1", ("choice", 9, 10, 10), "SOURCED", "agu20 8.8-10.2"),
    "membrane_thickness": ("mm", 0.10, "ILLUSTRATIVE", "GAP"),
    "seeds": ("1", ("int_uniform", 1, 8), "SOURCED", "agu20 1.3-11.3, lemon_misc 1.3-2.3"),
    "seed_length": ("mm", 8.0, "TYPICAL", ""),
    "oil_gland_density": ("1/cm2", 125.0, "SOURCED", "ben26 1.25 /mm2"),
    "peel_density": ("kg/m3", 818.0, "SOURCED", "jen22_lemon"),
    "albedo_density": ("kg/m3", 390.0, "SOURCED", "ben26"),
    "pulp_density": ("kg/m3", 1030.0, "DERIVED", "juice ~1.03 g/mL from USDA portion weights (lemon_misc)"),
    "peel_LCh": ("L*,C*,h", (66.0, 46.0, 93.0), "SOURCED", "agu20 L* 64-68, b* 42-49, hue 88-99"),
    "albedo_LCh": ("L*,C*,h", (92.0, 12.0, 95.0), "ILLUSTRATIVE", "GAP"),
    "pulp_LCh": ("L*,C*,h", (82.0, 35.0, 98.0), "ILLUSTRATIVE", "pale yellow; GAP"),
    "membrane_LCh": ("L*,C*,h", (88.0, 18.0, 95.0), "ILLUSTRATIVE", "GAP"),
    "friction_peel_rubber": ("1", ("uniform", 0.5, 0.8), "ILLUSTRATIVE", ""),
    "juice_fraction": ("%", ("uniform", 30.0, 45.0), "SOURCED", "agu20 31.7-44.3 (EU min 20)"),
    "juice_of_one_lemon": ("g", 48.0, "SOURCED", "lemon_misc (USDA)"),
    "acidity": ("g/L citric", "56-75", "SOURCED", "agu20"),
    "puncture_force": ("N", 41.2, "SOURCED", "ben26 5 mm probe"),
    "cutting_force": ("N", 36.3, "SOURCED", "ben26 craft knife"),
    "compression_force": ("N", "165.8 at 16 mm", "SOURCED", "ben26"),
    "chilling_threshold": ("C", 10.0, "SOURCED", "lemon_misc (Eureka)"),
    "wedge_mass": ("g", 5.9, "SOURCED", "lemon_misc (USDA)"),
    "bitter_juice_h": ("h", 1.0e6, "ILLUSTRATIVE", "lemon juice does not turn bitter like navel juice"),
}

CUCUMBER = {
    "type": ("-", "greenhouse long cucumber (Schlangengurke), parthenocarpic", "STANDARD", "dewiki_sg"),
    "length": ("mm", ("normal", 320.0, 25.0, 250, 380), "SOURCED", "unece_cuc 25-30 cm minimum; sakata11"),
    "diameter": ("mm", ("normal", 42.0, 3.0, 36, 50), "TYPICAL", "derived: 400 g / 32 cm / 950 kg/m3 -> ~41 mm"),
    "mass_check": ("g", "300-500 (dewiki_sg), typical 350 (cuc_misc)", "SOURCED", ""),
    "neck_ratio": ("1", ("uniform", 0.80, 0.90), "ILLUSTRATIVE", "stem end narrower"),
    "ribs": ("1", 3, "ILLUSTRATIVE", "faint ribs over the three carpels"),
    "rib_depth": ("1", ("uniform", 0.005, 0.012), "ILLUSTRATIVE", ""),
    "bend_radius": ("mm", ("uniform", 1000.0, 4000.0), "ILLUSTRATIVE", "within Class I (>= ~130 mm, unece_cuc)"),
    "skin_thickness": ("mm", ("uniform", 1.0, 1.6), "SOURCED-VERIFY", "cuc_misc peelable exocarp 1-2 mm"),
    "cavity_fraction": ("1", ("uniform", 0.45, 0.55), "ILLUSTRATIVE", "seed cavity / fruit radius; GAP"),
    "core_diameter": ("mm", ("uniform", 3.0, 5.0), "ILLUSTRATIVE", ""),
    "seeds_per_row": ("1", ("int_uniform", 25, 40), "ILLUSTRATIVE", "parthenocarpic: soft immature ovules"),
    "seed_length": ("mm", ("uniform", 4.0, 6.0), "ILLUSTRATIVE", "< mature 6.4-9.1 mm"),
    "density": ("kg/m3", 950.0, "SOURCED-VERIFY", "disect_cuc; water 95.2 % (USDA)"),
    "skin_LCh": ("L*,C*,h", (38.0, 34.0, 127.0), "SOURCED-VERIFY", "cuc_misc a* -20.6, b* 27.5 -> C 34, h 127; L* GAP"),
    "flesh_LCh": ("L*,C*,h", (85.0, 20.0, 112.0), "ILLUSTRATIVE", "pale whitish green; GAP"),
    "friction_skin_rubber": ("1", ("uniform", 0.4, 0.7), "ILLUSTRATIVE", "glossy waxy skin"),
    "round_thickness": ("mm", 5.0, "SOURCED", "disect_cuc (SliceIt! 5 mm slices)"),
    "puncture_3mm": ("N", "flesh 8.7, placenta 2.8, skin 9.7", "SOURCED", "sakata11"),
    "E_apparent": ("MPa", "0.85-1.39 (slice) | 2.5 (DiSECt)", "SOURCED-VERIFY", "jah16; disect_cuc"),
    "failure": ("MPa, 1", "0.53-0.85 at 0.45-0.53 strain", "SOURCED-VERIFY", "jah16"),
    "firmness_loss_9d": ("%", 49.0, "SOURCED-VERIFY", "jah16"),
    "storage_optimum": ("C, %RH", "10-12.5, 95, < 14 d", "SOURCED", "ucd_cuc"),
    "chilling": ("C, d", "< 10 C after 1-3 d", "SOURCED", "ucd_cuc"),
    "weight_loss_15d": ("%", "wrapped 0.66, unwrapped 11.1 (12 C)", "SOURCED-VERIFY", "cuc_misc"),
    "salting_min": ("min", 30.0, "SOURCED", "cuc_misc (wikibooks)"),
    "age_d": ("d", ("uniform", 1.0, 10.0), "ILLUSTRATIVE", ""),
    "storage": ("-", ("choice_str", "fridge_wrapped", "fridge_wrapped", "counter_wrapped", "fridge_unwrapped"),
                "ILLUSTRATIVE", "home fridge 4 C is in the chilling range"),
}

CROISSANT = {
    "product": ("-", "Buttercroissant (butter the only fat)", "STANDARD", "leitsaetze_fb"),
    "length": ("mm", ("normal", 135.0, 8.0, 115, 160), "ILLUSTRATIVE",
               "finished dimensions GAP; chosen for ~180 cm3 = 57 g x 3.13 cm3/g (ste25, USDA medium)"),
    "width": ("mm", ("normal", 68.0, 5.0, 55, 80), "ILLUSTRATIVE", "see length"),
    "height": ("mm", ("normal", 42.0, 4.0, 34, 50), "ILLUSTRATIVE", "see length"),
    "mass_check": ("g", "42 / 57 / 67 (small / medium / large)", "SOURCED", "croissant_misc (USDA)"),
    "curvature_radius": ("mm", ("choice", 70.0, 90.0, 1.0e5), "ILLUSTRATIVE", "crescent or straight (fr: butter = straight, SV)"),
    "steps": ("1", ("int_uniform", 5, 7), "ILLUSTRATIVE", "visible rolls; rolled 3.5-4 turns (croissant_misc)"),
    "step_relief": ("1", ("uniform", 0.05, 0.10), "ILLUSTRATIVE", ""),
    "rolling_turns": ("1", ("uniform", 3.5, 4.0), "SOURCED", "croissant_misc"),
    "butter_layers": ("1", 27, "SOURCED", "ste25: 3 single folds"),
    "butter_on_flour": ("%", ("uniform", 50.0, 64.0), "SOURCED", "ste25, recipes; legal min 30 (leitsaetze_fb)"),
    "crust_thickness": ("mm", ("uniform", 0.8, 1.4), "ILLUSTRATIVE", "GAP"),
    "crust_density": ("kg/m3", 450.0, "ILLUSTRATIVE", ""),
    "crumb_density": ("kg/m3", 300.0, "DERIVED", "1 / 3.13 cm3/g (ste25) = 0.32 g/cm3 whole"),
    "crust_LCh": ("L*,C*,h", (52.0, 48.0, 66.0), "ILLUSTRATIVE", "deep golden, egg-washed; GAP"),
    "crumb_LCh": ("L*,C*,h", (84.0, 26.0, 88.0), "ILLUSTRATIVE", "pale yellow (butter); GAP"),
    "flakes": ("1", ("int_uniform", 40, 120), "ILLUSTRATIVE", "loose flakes; GAP"),
    "friction_crust": ("1", ("uniform", 0.4, 0.7), "ILLUSTRATIVE", ""),
    "firmness": ("N", "5.16 (1 h) -> 19.2 (24 h), TPA 25 mm 50 %", "SOURCED", "ste25"),
    "reheat": ("C, min", "180, 8 from frozen; refresh 160-180, 3-5 (typical)", "SOURCED", "croissant_misc"),
    "age_h": ("h", ("choice", 1.0, 3.0, 8.0, 24.0), "ILLUSTRATIVE", ""),
}

BAGUETTE = {
    "product": ("-", "baguette de tradition francaise (flour, water, yeast/levain, salt)", "STANDARD", "wp_baguette"),
    "length": ("mm", ("normal", 600.0, 30.0, 540, 660), "SOURCED", "wp_baguette 55-65 cm"),
    "width": ("mm", ("normal", 55.0, 4.0, 45, 62), "SOURCED", "wp_baguette 4-6 cm"),
    "height": ("mm", ("normal", 42.0, 4.0, 32, 50), "SOURCED", "wp_baguette 3-5 cm"),
    "mass_check": ("g", "~250 (250-300 tradition)", "SOURCED", "wp_baguette"),
    "base_edge_height": ("mm", ("uniform", 4.0, 7.0), "ILLUSTRATIVE", ""),
    "lumpiness": ("1", ("uniform", 0.01, 0.025), "ILLUSTRATIVE", ""),
    "n_scores": ("1", ("choice", 5, 5, 6, 7), "SOURCED", "wp_grigne: 5 classic; 5-7 typical for 55-65 cm"),
    "score_length_mm": ("mm", ("uniform", 100.0, 125.0), "TYPICAL", "~10-12 cm, overlapping ~1/3"),
    "score_axis_angle": ("deg", ("uniform", 10.0, 20.0), "TYPICAL", ""),
    "score_length_frac": ("1", 1.0, "ILLUSTRATIVE", "per score: set by score_length_mm"),
    "score_width": ("mm", ("uniform", 9.0, 14.0), "ILLUSTRATIVE", "opened grigne"),
    "score_depth": ("mm", ("uniform", 5.0, 9.0), "SOURCED-VERIFY", "wp_grigne ~1 cm cut"),
    "crust_thickness_top": ("mm", ("normal", 1.9, 0.3, 1.4, 3.5), "SOURCED", "dv2012 1.7 mean (zou2024: 3-4, SV)"),
    "crust_thickness_score": ("mm", ("uniform", 0.8, 1.2), "ILLUSTRATIVE", ""),
    "crust_thickness_bottom": ("mm", ("uniform", 1.6, 2.4), "ILLUSTRATIVE", ""),
    "transition_thickness": ("mm", ("uniform", 1.0, 2.0), "ILLUSTRATIVE", ""),
    "crumb_density": ("kg/m3", ("normal", 200.0, 20.0, 160, 250), "DERIVED",
                      "~250 g / ~0.9 L bulk (dimensions, wp_baguette) minus a denser crust"),
    "crust_density": ("kg/m3", 380.0, "ILLUSTRATIVE", ""),
    "transition_density": ("kg/m3", 260.0, "ILLUSTRATIVE", ""),
    "crumb_porosity": ("1", ("uniform", 0.75, 0.85), "ILLUSTRATIVE", "open irregular crumb; tomography GAP"),
    "cell_diameter_median": ("mm", ("uniform", 1.3, 1.9), "ILLUSTRATIVE", "'irregular voids of various sizes'"),
    "cell_diameter_gsd_log": ("1", 0.75, "ILLUSTRATIVE", "wide spread: open crumb"),
    "cell_number_density": ("1/cm3", 260.0, "ILLUSTRATIVE", ""),
    "cell_elongation": ("1", ("uniform", 1.1, 1.3), "ILLUSTRATIVE", ""),
    "crust_L": ("L*", ("normal", 52.0, 3.0, 44, 58), "SOURCED-VERIFY", "zou2024 powder 57.5; intact darker"),
    "crust_C": ("C*", 36.4, "SOURCED-VERIFY", "zou2024 a* 10.5, b* 34.9"),
    "crust_h": ("deg", 73.3, "SOURCED-VERIFY", "zou2024"),
    "crumb_L": ("L*", 70.0, "SOURCED-VERIFY", "zou2024 66.4 (powder)"),
    "crumb_C": ("C*", 15.8, "SOURCED-VERIFY", "zou2024"),
    "crumb_h": ("deg", 96.0, "SOURCED-VERIFY", "zou2024"),
    "variant": ("-", ("choice_str", "plain"), "STANDARD", ""),
    "slice_height_frac": ("1", 0.42, "TYPICAL", "split lengthwise for sandwiches"),
    "friction_crust_rubber": ("1", ("uniform", 0.5, 0.8), "ILLUSTRATIVE", ""),
    "age_h": ("h", ("choice", 0.5, 2.0, 6.0, 20.0), "ILLUSTRATIVE", ""),
    "storage": ("-", ("choice_str", "paper_bag", "open"), "ILLUSTRATIVE", "baguette_misc"),
    "crumb_poisson": ("1", 0.103, "SOURCED", "lagrain_2012 (bread)"),
    "crust_water_critical": ("%", 10.1, "SOURCED", "hirte_2012"),
    "plastic_bag_factor": ("1", 4.0, "ILLUSTRATIVE", ""),
    "crust_break_force": ("N", 3.0, "SOURCED", "hirte_2012"),
    "crumb_squash_strain": ("1", 0.30, "ILLUSTRATIVE", ""),
    "bias_slice_mm": ("mm", 12.0, "TYPICAL", "crostini 1-1.5 cm at 30-45 deg"),
}

PEPPER = {
    "type": ("-", "blocky sweet pepper (Paprika)", "STANDARD", "pepper_misc (EU 2023/2429)"),
    "colour": ("-", ("choice_str", "red", "red", "yellow", "green"), "ILLUSTRATIVE", "retail mix"),
    "diameter": ("mm", ("normal", 78.0, 4.0, 68, 90), "SOURCED", "he22 77.6 mm"),
    "length": ("mm", ("normal", 83.0, 6.0, 70, 100), "SOURCED", "he22 81.3-85.3 mm"),
    "mass_check": ("g", "237 +- 37 (he22), 230-245 German retail (alt20)", "SOURCED", ""),
    "lobes": ("1", ("choice", 3, 3, 4), "SOURCED", "sha26: 2-4, mostly 3"),
    "lobe_depth": ("1", ("uniform", 0.04, 0.08), "ILLUSTRATIVE", ""),
    "wall_thickness": ("mm", ("normal", 7.2, 0.6, 5.5, 8.5), "SOURCED", "he22 7.5 red / 6.9 orange"),
    "wall_density": ("kg/m3", 1000.0, "TYPICAL", "92-94 % water"),
    "rib_depth": ("mm", ("uniform", 6.0, 12.0), "ILLUSTRATIVE", "GAP"),
    "rib_thickness": ("mm", ("uniform", 1.2, 2.0), "ILLUSTRATIVE", "GAP"),
    "placenta_diameter": ("mm", ("uniform", 20.0, 28.0), "ILLUSTRATIVE", "GAP"),
    "placenta_height": ("mm", ("uniform", 22.0, 32.0), "ILLUSTRATIVE", "GAP"),
    "seeds": ("1", ("int_uniform", 40, 140), "SOURCED-VERIFY", "pepper_misc 38-142"),
    "seed_diameter": ("mm", 3.5, "ILLUSTRATIVE", "1000-seed weight 4-6 g (sha26); size GAP"),
    "stem_length": ("mm", ("uniform", 10.0, 25.0), "ILLUSTRATIVE", "cut peduncle (EU: neatly cut)"),
    "red_LCh": ("L*,C*,h", (38.8, 41.9, 41.6), "SOURCED", "bar20"),
    "yellow_LCh": ("L*,C*,h", (61.4, 50.2, 84.3), "SOURCED", "bar20"),
    "green_LCh": ("L*,C*,h", (33.7, 19.0, 116.9), "SOURCED", "bar20"),
    "friction_skin_rubber": ("1", ("uniform", 0.35, 0.6), "ILLUSTRATIVE", "glossy waxy skin; robots report slip"),
    "firmness_whole": ("N", 43.5, "SOURCED", "pepper_misc (8 mm probe)"),
    "bite_firmness": ("N", "red 11.6, orange 6.9; -81 % in 15 d at 20 C", "SOURCED", "he22"),
    "chilling_threshold": ("C", 7.0, "SOURCED", "pepper_misc"),
    "weight_loss": ("g/d", "2.9-4.8 at 17 C / 55 % RH; shrivel from 6-8 % loss", "SOURCED", "alt20"),
    "shelf_life_d": ("d", "12-28", "SOURCED", "bar20"),
    "ring_mass": ("g", 10.0, "SOURCED", "pepper_misc (USDA)"),
    "age_d": ("d", ("uniform", 1.0, 10.0), "ILLUSTRATIVE", ""),
}

REGISTERS = {"egg": EGG, "apple": APPLE, "avocado": AVOCADO, "roll": ROLL, "carrot": CARROT, "banana": BANANA, "orange": ORANGE, "loaf": LOAF, "potato": POTATO, "tomato": TOMATO, "onion": ONION, "lemon": LEMON, "cucumber": CUCUMBER, "croissant": CROISSANT, "baguette": BAGUETTE, "pepper": PEPPER}


def draw(spec, rng):
    """Draw one value from a register entry's value field (numbers pass through; strings are model notes)."""
    if isinstance(spec, (int, float)):
        return float(spec)
    if not isinstance(spec, tuple):
        return None
    kind = spec[0]
    if not isinstance(kind, str):                         # a fixed vector, e.g. an L*C*h triple
        return tuple(float(v) for v in spec)
    if kind == "normal":
        _, m, s, lo, hi = spec
        while True:
            x = rng.normal(m, s)
            if lo <= x <= hi:
                return float(x)
    if kind == "uniform":
        return float(rng.uniform(spec[1], spec[2]))
    if kind == "choice_str":
        return spec[1 + int(rng.integers(len(spec) - 1))]
    if kind == "choice":
        return float(spec[1 + int(rng.integers(len(spec) - 1))])
    if kind == "int_uniform":
        return int(rng.integers(spec[1], spec[2] + 1))
    if kind == "lognormal":
        return float(spec[1] * spec[2] ** rng.normal())
    raise ValueError(spec)


# entries added after instances were built: drawn from a second random stream per instance (sample), so the
# earlier draws -- and the built assets -- stay the same
EXT_DRAW = {"egg": ("lay_SG", "hu_fresh", "yi_fresh", "ph_fresh", "ageing_rate_factor")}


def sample(kind, seed):
    """Hidden instance parameters for one object (only the entries with a value or a distribution)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    out = {}
    ext = EXT_DRAW.get(kind, ())
    for name, (unit, val, status, src) in REGISTERS[kind].items():
        if name in ext:
            continue
        v = draw(val, rng)
        if v is not None:
            out[name] = v
    out["_rng_state"] = int(rng.integers(1 << 31))
    rng2 = np.random.default_rng([seed, 1])              # entries added later: their own stream, so older instances
    for name in ext:                                     # keep every earlier draw (EXT_DRAW)
        v = draw(REGISTERS[kind][name][1], rng2)
        if v is not None:
            out[name] = v
    return out


def status_counts(kind):
    from collections import Counter
    return Counter(v[2] for v in REGISTERS[kind].values())
