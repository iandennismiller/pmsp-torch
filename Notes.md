# Notes

## 2019-11-25

What have we accomplished so far?

- We are reproducing the substantial findings of PSMP96 and KMP13, first with Lens and then with PyTorch.
- We have Lens running in a virtual machine - and the VM is portable to other hosts.
- We have the PMSP96 architecture implemented in Lens and PyTorch
- we are adapting the PMSP96/SM89 stimulus materials and experiment-specific training sets, partly to validate pytorch
- We have reproduced the mini-network warping findings of KMP13 with Lens.

Where are we going?

- as we validate this replicated framework, we want to take some incremental steps to push these models farther and to push pytorch farther.
- study 1: a comparison - the introduction of an irregular term with proportion N, versus the introduction of two new terms, matched by irregularity, each presented with proportion (1/2)N.  We propose to examine this for warping, as with KMP13, but within the pytorch framework.
- study 2: a semantics extension to eliminate O-P warping: using toy semantics, can we cause irregularities in the O-P mapping to migrate to distinct semantic representations?  One hypothesized outcome would be the reversal of O-P warping effects upon the introduction of semantically-distinct O-P irregularities.

## 2019-12-09

non-monotonic effect for ambiguous words; over extension of pronunciation that backs off, less than max...

- reset derivs, link specific rule, see whether that fixes it
- rerun sim in lens
- delta bar delta is in lens, run pmsp here, can reset link memory, set lr to 0 and train 1 epoch (zeroes hidden) or change LR algorithm, change back to dbd, train net, save, quit lens and reload - that resets link weights
- why does torch not fully replicate lens?
- can run 5-10 behavioral ppl from lab
- @blairarmstrong on github
- jay ruckle individual differences in simulations
