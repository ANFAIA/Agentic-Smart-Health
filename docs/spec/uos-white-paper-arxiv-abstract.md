# Abstract for the arXiv submission form (provisional)

**Status:** provisional draft of 20 November 2026, **not approved by the author**. It must not
be submitted until the registration accuracy of both clinical cases (landmark TRE) and the
results of the pending registration and segmentation work are in the paper; the sentences
on registration will change then.

**Rules for this field:** plain text, no LaTeX, at most 1,920 characters (this draft:
1912). The PDF keeps its own executive summary. Every figure here must appear in
`uos-white-paper.tex` with the same scope.

Source of each figure in the paper: mesh regeneration on Teeth3DS+ (section "What has been
measured"); repeatability and source audit of case A (same section and "A case, end to
end"); seed sensitivity of case B ("Every relation carries its error and its author") and the
count of 7 of 20 seeds ("Precision without an annotator", in "What has been measured").

---

Dental cases arrive in pieces: intraoral scans, CBCT series, photographs and reports, each in its own format, with the spatial, temporal and semantic relations between them left inside the applications that produced them. We present UOS (Unified Oral Scene), an open container that composes a multimodal case without re-encoding its sources. Acquired originals are referenced by hash rather than copied. Converted geometry, derived representations, registrations with their residuals, views and a per-asset record of losses and provenance travel inside, and measured, computed and model-inferred content are kept in separate layers that a recipient can inspect or remove. We release a specification, JSON Schemas, a validator, synthetic conformance fixtures and a reference implementation, and report each result at the scope at which it was measured, on two clinical cases. Regenerating public intraoral meshes through the pipeline preserves coordinates to within 5 x 10^-6 mm, a numerical result rather than scanner accuracy. In the first case, three runs on each of its two CBCT series produced byte-identical binary payloads in one environment, and a source audit showed that the historical scan-to-CBCT registration had been computed on the two series mixed together; registered separately, neither reaches the overlap the review gate requires. In the second case, the production registration moved by 1-3 mm with its random seed and could converge rotated by about 180 degrees with an unchanged residual; in a later measurement, 7 of 20 seeds converged more than 5 mm from the reference pose. Experimental tooth labels, photograph-based regional colour and reconstructed roots remain prototypes. No registration has yet been measured against anatomical landmarks, no result is validated against clinical tasks, and UOS remains a proposal until a second implementation opens containers written by the first.
