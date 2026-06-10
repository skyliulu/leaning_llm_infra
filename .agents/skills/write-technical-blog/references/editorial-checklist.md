# Technical Article Editorial Checklist

## Source Analysis

- Identify the source's conceptual order before compressing it.
- Record definitions, assumptions, theorem dependencies, canonical examples, and key figures.
- Preserve key derivations even when reducing repetition.
- Verify modern or unstable claims against primary sources.
- Replace local source paths with public repository or paper URLs.

## Reader Onboarding

- State the problem in ordinary language.
- Declare prerequisites and provide only the missing mathematical background.
- Define specialized terms before first use.
- Introduce a complete running example before the first abstraction-heavy section.
- Explain what success and failure look like in that example.

## Narrative Continuity

- Begin each major section with a limitation or question inherited from the previous section.
- Make the conceptual dependency visible in transitions.
- Reuse the running example instead of inventing unrelated examples for every formula.
- Explain algorithms as changes to learning signals, data, objectives, or estimators.
- End each major section by stating what it solves and what remains unresolved.

## Mathematical Exposition

- Define the domain and meaning of every symbol.
- Show intermediate algebra when changing distributions, differentiating objectives, introducing a baseline, applying recursion, or normalizing probabilities.
- Label approximations and sample estimators honestly.
- Explain the role of each term after the equation.
- State edge cases and assumptions.
- Pair long derivations with an interpretation or worked example.

## Figures

- Give every figure one dominant teaching goal.
- Ensure prose refers to the figure before it appears.
- Ensure the caption tells readers what conclusion to draw.
- Avoid figures that only decorate a section or repeat nearby prose.
- Compare methods using aligned dimensions such as data, objective, model components, online sampling, and cost.

## Style

- State knowledge directly instead of repeatedly writing "the source says."
- Prefer concrete verbs over vague transitions.
- Use English technical terms when they improve precision, but explain them in Chinese on first use.
- Avoid unexplained acronym clusters.
- Avoid a sequence of headings containing only equations.
- Keep table cells compact; move nuanced explanations into prose.

## Final Review Questions

1. Can a reader explain the running example before seeing the first formal model?
2. Can the reader say why each new concept is introduced?
3. Can the reader reconstruct the main derivation without guessing a missing step?
4. Can the reader distinguish methods that look similar but use different data or objectives?
5. Does every figure make a specific relationship easier to understand?
6. Are all references public, authoritative, and relevant?
