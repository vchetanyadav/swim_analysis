# Label codebook - AI Flotation Analyser

The scientific target is **passive flotation**: whether the swimmer stays at the surface with
little or no propulsion. Orientation (horizontal vs vertical) describes HOW someone floats, not
WHETHER. A calm vertical swimmer holding the surface is floating. A swimmer sculling hard and
sinking is not. Rate orientation, effort and stability **separately** - never fold them into the
float decision.

## float_label - the primary target
| value | meaning |
|---|---|
| 0 | not floating - losing the surface / sinking, face/airway going under, cannot sustain position |
| 1 | almost floating - borderline; holding only with effort, drifting down slowly, or intermittent |
| 2 | floating - surface sustained, airway clear, position maintainable (in ANY orientation) |

## passive_active - how much work it takes
| value | meaning |
|---|---|
| 0 | passive - still/relaxed, surface held with minimal movement |
| 1 | mixed - some propulsion but not constant |
| 2 | actively held - continuous sculling/kicking needed to stay up |

## posture - orientation only (NOT a float judgement)
| value | meaning |
|---|---|
| 0 | horizontal - body flat near surface |
| 1 | inclined - diagonal trunk |
| 2 | vertical - upright trunk (can still be floating) |

## stability
| value | meaning |
|---|---|
| 0 | stable - little wobble, steady position |
| 1 | moderate - some sway/correction |
| 2 | unstable - large movement, fighting to hold position |

## label_confidence - the annotator's certainty in THIS label
| value | meaning |
|---|---|
| 0 | low - hard to tell (poor skeleton, ambiguous behaviour) |
| 1 | medium |
| 2 | high - clear |

## exclude
1 = do not use (pose too poor, not a float attempt, wrong segment). 0 = keep.

## Provenance columns (round 2 onward)
- `annotator_id` - who labelled (e.g. instructor_01)
- `annotation_round` - 1 = original set, 2 = this targeted round
- `notes` - free text, especially for vertical-but-floating calls (e.g. "vertical but stable passive float")

## Decision reminders
- Vertical + calm + surface held = **float_label 2**, posture 2, passive_active 0. This is the case
  the earlier round got wrong; it is the whole point of the re-review.
- Sinking slowly but still fighting = float_label 1 (almost), not 2.
- If the skeleton is too broken to judge: set exclude = 1 rather than guessing.
