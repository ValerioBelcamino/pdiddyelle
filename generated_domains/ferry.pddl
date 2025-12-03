(define (domain ferry)
(:requirements :typing :strips :negative-preconditions :action-costs)
(:types
car - object
location - object )

(:predicates
(at-ferry ?l - location)
(at ?c - car ?l - location)
(empty-ferry)
(on ?c - car))

(:functions
(total-cost) - number
)

(:action sail
:parameters  (?from - location ?to - location)
:precondition (and (at-ferry ?from) (not (at-ferry ?to)))
:effect (and  (at-ferry ?to) (not (at-ferry ?from))
(increase (total-cost) 1)
))


(:action board
:parameters (?car - car ?loc - location)
:precondition  (and  (at ?car ?loc) (at-ferry ?loc) (empty-ferry))
:effect (and
(on ?car)
(not (at ?car ?loc))
(not (empty-ferry))
(increase (total-cost) 1)
))

(:action debark
:parameters  (?car - car  ?loc - location)
:precondition  (and (on ?car) (at-ferry ?loc))
:effect (and
(at ?car ?loc)
(empty-ferry)
(not (on ?car))
(increase (total-cost) 1)
))
)
