(define (domain miconic)
(:requirements :strips :typing :action-costs)
(:types passenger - object
floor - object
)

(:predicates
(origin ?person - passenger ?floor - floor)

(destin ?person - passenger ?floor - floor)

(above ?floor1 - floor  ?floor2 - floor)

(boarded ?person - passenger)

(served ?person - passenger)

(lift-at ?floor - floor)
)

(:functions
(total-cost) - number
)

(:action board
:parameters (?f - floor ?p - passenger)
:precondition (and (lift-at ?f) (origin ?p ?f))
:effect (and (boarded ?p) (not (origin ?p ?f))
(increase (total-cost) 1)
))

(:action depart
:parameters (?f - floor ?p - passenger)
:precondition (and (lift-at ?f) (destin ?p ?f)
(boarded ?p))
:effect (and (not (boarded ?p))
(served ?p)
(increase (total-cost) 1)
))

(:action up
:parameters (?f1 - floor ?f2 - floor)
:precondition (and (lift-at ?f1) (above ?f1 ?f2))
:effect (and (lift-at ?f2) (not (lift-at ?f1))
(increase (total-cost) 1)
))

(:action down
:parameters (?f1 - floor ?f2 - floor)
:precondition (and (lift-at ?f1) (above ?f2 ?f1))
:effect (and (lift-at ?f2) (not (lift-at ?f1))
(increase (total-cost) 1)
))
)
