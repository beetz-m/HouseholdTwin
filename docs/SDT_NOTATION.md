# SDT tree notation (s-expressions with RPL tags)

The twin is one compositional tree, written with RPL's `:tag` construct. The full proposal, with grammar,
designators, schemas and knowledge assertions, is in §4b of the concept document "A Conceptual CRAM Layer"
(https://claude.ai/artifact/DaTuYZ6ftPF8YSJ1kuCmHE).

## Rules

1. **Tags name nodes; paths are implicit.**
   - `(:tag egg-3 (node … (parts (:tag yolk (node …)))))`.
   - The path `(egg-3 yolk)` is computed from the node's position and never written.
2. **References go through the tag table.**
   - A reference is a tag, scoped where ambiguous: `(tagged yolk egg-3)`.
   - It resolves to a node handle that stays bound when the tree is restructured: a subtree moves, is replaced, or
     gets a new parent. Only the computed path changes. This is RPL: tags give stable references that survive
     transformation.
3. **Location is a relation, not part of the name:** `(in (tagged yolk egg-3) bowl-1)`.
4. **Transformation creates new tagged nodes with `(derived-from …)`.**
5. **Actions are tagged nodes too:** `(:tag crack-1 (perform (an action …)))`.
   - Episode trees have action tokens as parts.
   - Roles refer to world nodes by tag.
6. **The node is the grounding of its reference:** `(grounding (usd FILE PRIM) (iri IRI))`.

## Tooling

    E=~/Documents/LabAutomationTwin/env_cad/bin/python
    $E tools/sdt_sexpr.py assets/food/egg_3 assets/food/avocado_4      # -> docs/sexpr/<instance>.lisp
    $E tools/sdt_sexpr.py --episode assets/food/egg_3/process/rim/log.json
    $E tools/sdt_sexpr.py --ground "germinal-disc egg-3"              # implicit path, USD prim, IRI

API:
- `parse()` reads the trees.
- `TagTable(forms)`:
  - `.find(tag, within)` resolves a reference to a handle;
  - `.path(handle)` gives the implicit path;
  - `.move(handle, new_parent)` restructures the tree.
- Tests are in `tests/test_sdt_sexpr.py`.

## Not yet

- **Flat IRIs.** `sdt:represents` IRIs are flat (`…/egg_3/yolk_body`). They should be minted once per node and kept
  through restructuring, like the tag.
- **No tagged export for kitchen, robot and tools** (SDT world bodies and annotations).
- **Made objects** carry no `derived-from` yet, and the emulator does not re-parent subtrees in the tag table.
