"""Tag references survive restructuring (RPL :TAG): the node path is implicit and recomputed, the tag stays bound."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import sdt_sexpr as S  # noqa: E402

TREE = """
(:tag kitchen (node (is-a sdt:kitchen) (parts
  (:tag bowl-1 (node (is-a sdt:bowl) (parts (:tag rim (node (is-a sdt:rim))))))
  (:tag egg-3 (node (is-a food:egg) (parts
     (:tag shell (node (is-a food:eggshell)))
     (:tag yolk (node (is-a food:yolk) (parts (:tag germinal-disc (node (is-a food:germinal-disc))))))))))))
"""


def test_paths_are_implicit():
    tt = S.TagTable(S.parse(TREE))
    h = tt.find("germinal-disc", "egg-3")
    assert tt.path(h) == ["kitchen", "egg-3", "yolk", "germinal-disc"]


def test_reference_survives_restructuring():
    tt = S.TagTable(S.parse(TREE))
    yolk = tt.find("yolk", "egg-3")
    node_before = tt.node[yolk]
    tt.move(yolk, tt.find("bowl-1"))                       # open-up: the yolk is now in the bowl
    assert tt.find("yolk") == yolk and tt.node[yolk] is node_before
    assert tt.path(yolk) == ["kitchen", "bowl-1", "yolk"]
    assert tt.path(tt.find("germinal-disc")) == ["kitchen", "bowl-1", "yolk", "germinal-disc"]


def test_ambiguous_tag_needs_scope():
    two = TREE.replace("(:tag rim", "(:tag yolk (node (is-a food:yolk))) (:tag rim")
    tt = S.TagTable(S.parse(two))
    try:
        tt.find("yolk")
        raise AssertionError("expected an ambiguity error")
    except KeyError:
        pass
    assert tt.path(tt.find("yolk", "egg-3"))[-2] == "egg-3"
