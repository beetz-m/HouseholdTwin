;;; Rules of the egg activity's knowledge base: state-dependent facts from the observed atoms (def-rule).
;;; Float test classes after food/egg_lifetime.py float_test_model (lies flat / tilts / stands / floats); the tilt
;;; thresholds are ILLUSTRATIVE (the model's classes are hidden; the robot sees floats yes/no and a tilt angle).

(def-rule lies-flat
  (:if (float-observed ?egg sinks ?tilt) (< ?tilt 15))
  (:then (float-class ?egg flat) (fresh ?egg)))

(def-rule tilts
  (:if (float-observed ?egg sinks ?tilt) (>= ?tilt 15) (< ?tilt 60))
  (:then (float-class ?egg tilts) (fresh ?egg)))

(def-rule stands
  (:if (float-observed ?egg sinks ?tilt) (>= ?tilt 60))
  (:then (float-class ?egg stands) (use-soon ?egg)))

(def-rule floats
  "An egg that floats has lost too much moisture: not for the batch (usda / egg_lifetime float test)."
  (:if (float-observed ?egg floats ?tilt))
  (:then (float-class ?egg floats) (stale ?egg)))

(def-rule unfit-contents
  "A cup holding the contents of a stale egg, or with shell left in it, must not go into the batch."
  (:if (contents-in ?egg ?cup) (stale ?egg))
  (:then (unfit ?cup)))

(def-rule shell-in-cup-unfit
  (:if (shell-seen ?cup ?n) (> ?n 0))
  (:then (unfit ?cup)))
