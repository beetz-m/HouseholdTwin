;;; Belonging norms as rules (BE1: contextual, defeasible, functional): from the observed kind and state of an object
;;; and the household context to (belongs-at o place relation confidence); prohibitions as (prohibited-at o place).
;;; The adapter picks the applicable place: the most confident norm whose place is not prohibited (BelongingStore).
;;; Confidences are ILLUSTRATIVE (a household's conventions; in a real deployment learned or told).

(def-rule opened-milk-after-breakfast
  "Opened milk goes back into the fridge after breakfast (it stays on the table while breakfast is on)."
  (:if (is-a ?o milk-pack) (opened ?o) (context after-breakfast) (not (empty ?o)))
  (:then (belongs-at ?o fridge inside 0.9)))

(def-rule empty-milk-to-the-bin
  "An empty carton: the original relation no longer applies (successor-state resolution)."
  (:if (is-a ?o milk-pack) (empty ?o))
  (:then (belongs-at ?o bin inside 0.95) (prohibited-at ?o fridge)))

(def-rule dirty-dishes-to-the-dishwasher
  "Used plates, cups and knives go into the dishwasher, never back into the cupboard (state-mediated return)."
  (:if (is-a ?o ?k) (member ?k (plate cup knife)) (dirty ?o))
  (:then (belongs-at ?o dishwasher inside 0.9) (prohibited-at ?o cupboard)))

(def-rule clean-dishes-to-the-cupboard
  (:if (is-a ?o ?k) (member ?k (plate cup)) (not (dirty ?o)))
  (:then (belongs-at ?o cupboard on 0.8)))

(def-rule coffee-to-the-cupboard
  (:if (is-a ?o coffee-jar))
  (:then (belongs-at ?o cupboard on 0.9)))

(def-rule sugar-on-the-counter
  (:if (is-a ?o sugar-jar))
  (:then (belongs-at ?o counter on 0.9)))

;;; Requests (RQ1-RQ*): who may overrule a standing norm, and what compliance costs.
(def-rule resident-request-licenses
  "A resident's episodic request defeats the standing norm for its scope: compliance is a licensed disturbance and
  opens a restoration obligation against the place the norm established (RQ4 = RE1)."
  (:if (request ?r resident ?o ?place episodic) (not (request-ended ?r)) (belongs-at ?o ?home ?rel ?c) (/= ?home ?place))
  (:then (licensed ?o) (obligation-open ?o ?home ?r)))

(def-rule request-scope-ended
  "When the request's scope ends the debt comes due (ObligationStatus.DISCHARGEABLE)."
  (:if (request ?r resident ?o ?place episodic) (request-ended ?r) (belongs-at ?o ?home ?rel ?c) (/= ?home ?place)
       (not (at ?o ?home)))
  (:then (obligation-due ?o ?home ?r)))

(def-rule guest-request-conflict
  "A guest may ask but not overrule: a conflicting request is asked about, not decided either way (RQ3)."
  (:if (request ?r guest ?o ?place ?scope) (not (request-ended ?r)) (belongs-at ?o ?home ?rel ?c) (/= ?home ?place))
  (:then (request-conflict ?o ?r)))
