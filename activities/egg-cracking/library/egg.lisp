;;; Cracking eggs into a batch -- the activity's schema library (conceptual CRAM, D1-D10) on the HouseholdTwin egg
;;; SDT (food/egg_emulator.py). Every keyword is an ontology term (D9; ontology/egg.ttl + the ccram core).
;;; The robot sees only measurements: the shell state from the strike outcome, freshness from the float test, shell
;;; fragments and the yolk from inspecting the cup (bowl monitor); hidden truth stays in the emulator.

;;; ================================================================================================================
;;; 1. The egg-shell microtheory (egg_crack.py: strike / open_up / fish_out)
;;; ================================================================================================================
(def-microtheory egg-shell
  (:regime contact-transmission)
  (:quantities (strike-speed m/s) (shards count))
  (:modes
    (intact        :when (shell-state ?egg intact)       :class safe)
    (microcracked  :when (shell-state ?egg microcracked) :class safe)        ; invisible: reads as intact
    (cracked       :when (shell-state ?egg cracked)      :class safe)
    (opened        :when (shell-state ?egg opened)       :class safe)
    (shattered     :when (shell-state ?egg shattered)    :class unrecoverable :signals egg-shattered))
  (:closing-events (open-into-batch-bowl :prevented-by (gate opening))))

;;; ================================================================================================================
;;; 2. Failures (Q2)
;;; ================================================================================================================
(def-failure no-crack
  (:specializes manipulation-failure)
  (:signalled-by (action cracking))
  (:describes (egg ?e) (speed ?v))
  (:suggests (seq retry)))

(def-failure egg-shattered
  (:specializes manipulation-failure)
  (:signalled-by (action cracking) (microtheory egg-shell))
  (:describes (egg ?e)))

(def-failure egg-crushed
  (:specializes manipulation-failure)
  (:signalled-by (action picking-up))
  (:describes (egg ?e) (force ?f)))

(def-failure egg-slipped
  (:specializes manipulation-failure)
  (:signalled-by (action picking-up))
  (:describes (egg ?e) (force ?f)))

(def-failure yolk-broken
  (:specializes manipulation-failure)
  (:signalled-by (action opening))
  (:describes (egg ?e) (vessel ?v))
  (:violates (yolk-whole ?e)))                          ; a resuming handler accepts exactly this goal part

(def-failure shell-left-in-contents
  (:specializes manipulation-failure)
  (:signalled-by (action fishing-out))
  (:describes (vessel ?v) (seen ?n)))

(def-failure batch-spoiled
  (:specializes closing-event-violation))              ; never handled: the gate prevents it

;;; ================================================================================================================
;;; 3. Action schemas
;;; ================================================================================================================
(def-action-schema picking-up
  (:roles (object (an egg)) (force (some grip-force)))
  (:telos (:over ?i) (achieve (held ?object) :at (end ?i))                 ; "egg held, shell unchanged"
                     (prevent (shell-state ?object cracked) :during ?i)    ;   (egg_tasks PickUpEgg)
                     (prevent (shell-state ?object shattered) :during ?i))
  (:failures (egg-slipped EggSlipped) (egg-crushed EggCrushed))
  (:phases (phase approach (goal (pre-grasp ?object))) (phase close (goal (held ?object)))))

(def-action-schema cracking
  (:roles (object (an egg)) (tool (a strike-surface)) (speed (some strike-speed)))
  (:pre   (held ?object))
  (:telos (:over ?i) (achieve (shell-state ?object cracked) :at (end ?i))
                     (prevent (shell-state ?object shattered) :during ?i))
  (:failures (no-crack NoCrack) (egg-shattered EggShattered))
  (:phases (phase swing (goal (above ?object ?tool 0.05))) (phase strike (terminal (contact ?object ?tool)))))

(def-action-schema opening
  (:roles (object (an egg)) (destination (a cup)))
  (:pre   (shell-state ?object cracked))
  (:telos (:over ?i) (achieve (contents-in ?object ?destination) :at (end ?i))
                     (achieve (yolk-whole ?object) :at (end ?i))
                     (prevent (opened-into ?object ?destination) :while (batch-bowl ?destination)))
  (:failures (yolk-broken YolkBroken))
  (:phases (phase hold-over (goal (above ?object ?destination 0.03))) (phase pull-apart (terminal released))))

(def-action-schema inspecting
  (:roles (container (a cup)))
  (:telos (:over ?i) (achieve (inspected ?container) :at (end ?i)))
  (:phases (phase look (goal (inspected ?container)))))

(def-action-schema fishing-out
  (:roles (container (a cup)))
  (:telos (:over ?i) (achieve (shell-free ?container) :at (end ?i)))
  (:failures (shell-left-in-contents ShellLeftInContents))
  (:phases (phase scoop (goal (shell-free ?container))) (phase look (goal (inspected ?container)))))

(def-action-schema float-testing
  (:roles (object (an egg)))
  (:telos (:over ?i) (achieve (float-tested ?object) :at (end ?i)))
  (:phases (phase lower (goal (in-water ?object))) (phase observe (goal (float-tested ?object)))))

(def-action-schema weighing
  (:roles (object (an egg)))
  (:telos (:over ?i) (achieve (weighed ?object) :at (end ?i))))

(def-action-schema emptying
  (:roles (object (a cup)) (destination (a batch-bowl)))
  (:pre   (shell-free ?object))
  (:telos (:over ?i) (achieve (empty ?object) :at (end ?i))
                     (prevent (emptied-into ?object ?destination) :while (unfit ?object))
                     (prevent (spilled ?object) :during ?i))
  (:phases (phase tip (goal (tilted ?object))) (phase return (goal (upright ?object)))))

;;; ================================================================================================================
;;; 4. Plan schemas
;;; ================================================================================================================
(def-plan-schema crack-egg-cleanly ((egg egg) (cup cup))
  (:body (seq (:tag grip (perform (an action (type picking-up) (object ?egg) (force (some grip-force)))))
              (with-failure-handling ((no-crack (f) (retry)))               ; strike again, a little faster
                (:tag strike (perform (an action (type cracking) (object ?egg) (tool (a strike-surface))
                                                (speed (some strike-speed))))))
              (:tag open (perform (an action (type opening) (object ?egg) (destination ?cup))))
              (:tag look (perform (an action (type inspecting) (container ?cup))))
              (:tag clean (achieve (shell-free ?cup)))))
  (:escapes manipulation-failure)
  (:succ (and (contents-in ?egg ?cup) (shell-free ?cup))))

(def-plan-schema add-egg-to-batch ((egg egg) (cup cup) (bowl batch-bowl))
  (:body (seq (:tag check (perform (an action (type float-testing) (object ?egg))))
              (with-failure-handling ((yolk-broken (f) (continue)))            ; a batter takes a broken yolk
                (perform-plan crack-egg-cleanly ?egg ?cup))
              (:tag add (perform (an action (type emptying) (object ?cup) (destination ?bowl))))))
  (:escapes manipulation-failure)
  (:succ (contents-in ?egg ?bowl)))

(def-plan-schema two-eggs-into-batch ((bowl batch-bowl))
  (:body (seq (:tag egg-1 (perform-plan add-egg-to-batch (the egg (tag egg-1)) (a cup) ?bowl))
              (:tag egg-2 (perform-plan add-egg-to-batch (the egg (tag egg-2)) (a cup) ?bowl))))
  (:escapes manipulation-failure))

(def-plan-schema open-into-the-batch-bowl ((egg egg) (bowl batch-bowl))   ; the closing event, for the gate test
  (:body (seq (:tag grip (perform (an action (type picking-up) (object ?egg) (force (some grip-force)))))
              (with-failure-handling ((no-crack (f) (retry)))
                (:tag strike (perform (an action (type cracking) (object ?egg) (tool (a strike-surface))
                                                (speed (some strike-speed))))))
              (:tag open (perform (an action (type opening) (object ?egg) (destination ?bowl))))))
  (:escapes manipulation-failure))
