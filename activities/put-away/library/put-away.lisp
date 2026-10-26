;;; Put things where they belong -- the activity's schema library (conceptual CRAM, D1-D10; competence A, M3).
;;; Belonging is context-indexed and defeasible (D7): the norms are rules (rules/belonging.lisp) that conclude
;;; (belongs-at o place relation confidence) from the observed kind and state of o and the household context; the
;;; adapter selects the applicable place (most confident, not prohibited) and the placement verdict exactly as
;;; coraplex.household_norms.assess_placement does. "its destination" is that place, resolved by the climb.
;;;
;;; The four guard behaviours (TD, RQ*):  stray -> restore;  in place -> leave (no over-tidying);  no home -> ask
;;; (no guessing);  a request -> comply with a debt (resident) or abstain and ask (guest, or nothing admissible).

;;; ================================================================================================================
;;; 1. Failures (Q2)
;;; ================================================================================================================
(def-failure no-belonging-place
  (:specializes designator-failure)
  (:signalled-by (action putting-away))
  (:describes (object ?o))
  (:suggests (seq asking)))

(def-failure place-not-admissible
  (:specializes manipulation-failure)
  (:signalled-by (action putting-away))
  (:describes (object ?o) (place ?p) (fails ?constraint))
  (:violates (at ?o ?p))                           ; a resuming handler accepts that it is not placed (it asked)
  (:suggests (seq asking)))

(def-failure prohibited-placement
  (:specializes closing-event-violation))                  ; never handled: the gate prevents it

;;; ================================================================================================================
;;; 2. Action schemas
;;; ================================================================================================================
(def-action-schema surveying
  (:roles (area (a countertop)))
  (:telos (:over ?i) (achieve (surveyed ?area) :at (end ?i)))
  (:phases (phase look (goal (surveyed ?area)))))

(def-action-schema putting-away
  (:roles (object (a household-object)) (destination (its destination)))
  (:telos (:over ?i) (achieve (at ?object ?destination) :at (end ?i))
                     (prevent (at ?object ?destination) :while (prohibited-at ?object ?destination))
                     (prevent (dropped ?object) :during ?i))
  (:failures (no-belonging-place NoBelongingPlace) (place-not-admissible PlaceNotAdmissible))
  (:phases (phase take (goal (held ?object))) (phase carry (goal (above ?object ?destination 0.05)))
           (phase place (goal (at ?object ?destination)) (terminal (released ?object)))))

(def-action-schema asking
  (:roles (recipient (a person)) (query (a household-object)))      ; SOMA QueryingTask: the query is about it
  (:telos (:over ?i) (achieve (asked ?recipient ?query) :at (end ?i)))
  (:phases (phase say (goal (asked ?recipient ?query)))))

;;; ================================================================================================================
;;; 3. Plan schemas
;;; ================================================================================================================
(def-plan-schema tidy-after-breakfast ((area countertop))
  (:body (seq (:tag look (perform (an action (type surveying) (area ?area))))
              (with-failure-handling ((no-belonging-place (f) (continue))       ; asked about it: leave it
                                      (place-not-admissible (f) (continue)))   ; asked: never "somewhere admissible"
                (:tag tidy (achieve (at (each household-object) (its destination)))))))
  (:escapes manipulation-failure))

(def-plan-schema put-one-away ((object household-object))
  (:body (seq (:tag put (perform (an action (type putting-away) (object ?object) (destination (its destination)))))))
  (:escapes designator-failure manipulation-failure))
