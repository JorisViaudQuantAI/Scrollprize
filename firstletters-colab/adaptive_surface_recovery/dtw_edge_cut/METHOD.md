# Exact graph encoding of monotone waveform correspondence costs

This note documents the implemented and tested Adaptive 062 solver, not a new efficacy claim. [Protocol](protocol.md), [solver](edge_solver.py), [receipt](result.json), [handoff](../../../collab/chatgpt-adaptive-062.md). Inspiration: Ishikawa, TPAMI 2003, doi:10.1109/TPAMI.2003.1233908; ordered-label submodular priors, doi:10.1016/j.dam.2009.02.026.

## Reduction

Let p and q be discrete depth labels 0,...,D-1 in two adjacent native CT columns. Let m be a measured monotone correspondence from p to q (the clipped rounded-median map of the DTW path). Introduce threshold indicators X_{p,k}=1[p>=k] for k=1,...,D-1, and analogously X_{q,t}. Define h(t)=min{j:m(j)>=t}, with h(t)=D for an empty set. Monotonicity gives 1[m(p)>=t]=1[p>=h(t)]. Therefore

    |q-m(p)| = sum_{t=1}^{D-1} |X_{q,t} - X_{p,h(t)}|.

For h(t)=0 the second indicator is identically 1 (source terminal); for h(t)=D it is identically 0 (sink terminal). Otherwise two opposite arcs of capacity w between the corresponding threshold vertices encode the weighted XOR exactly. No negative capacities or path enumeration are needed. Apply the same construction in the reverse direction with its own measured monotone map. The two maps need not be inverses. Their costs are both retained.

The unary difference encoding, threshold implications, slope bounds, minimum sheet separation and exact anchors remain those of 057/061. The sum of all finite arc capacities plus one bounds the penalty needed to enforce hard implications whenever the feasible set is nonempty. Decode the source-reachable residual set, then independently evaluate the integer objective. Reject any hard-constraint violation, residual source/sink path, or disagreement between direct objective and flow plus unary constant. For identity maps this penalty becomes an ordinary depth total-variation cost.

## Computational scope

For S sheets, N lateral columns, E undirected lateral adjacency edges and D depth states, threshold vertices are S*N*(D-1)+2. The extra directed correspondence terms contribute at most 4*S*E*(D-1) arcs before coalescing, rather than a dense D-by-D pairwise table per edge. The implementation caps threshold nodes and arcs; runtime is solver- and field-dependent. The DTW correspondences are computed once per directed edge, independently of sheet count. It assumes a single-valued ordered surface graph over the chosen depth axis; it cannot resolve arbitrary folded surfaces or unknown sheet counts.

## Verification and decision

062 independent exhaustive fixtures give energies 115 and 85 exactly; the identity case gives energy 62 and the same decoded optimum as TV. On the first actual CT cube, lateral transposition gives energy 389562 in both representations and zero decoded-depth differences. This is not a theorem that every numerical implementation or every tied optimum must return identical labels under permutation. The objective, rather than a chosen DTW spanning tree, is invariant to the consistent coordinate permutation.

Real efficacy did not pass. The fixed tree prior K outperformed the new all-edge G in the four-cube aggregate, and the sample contained only one multi-sheet cube with no eligible tight-gap pairs. Do not select a different weight after inspecting the test. The tested solver may be reused for future *independently validated* monotone maps, but an exact minimum of an inaccurate physical objective can still select the wrong sheet. The mathematical result is separable from the failed anatomical validation.
