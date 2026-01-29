# Price Family Center for the Social Brain  
## Annual Progress Report

**Price Center Heads of Lab:** Daniel Kronauer, Vanessa Ruta  
**Project Title:** *From Scent to Society: Rewiring Olfactory Navigation for Social Behavior*


## Overview

This project examines how ancient sensory algorithms adapt to support social behavior across species. Our central hypothesis: **bilateral comparison of odor input (osmotropotaxis)** is a conserved navigational computation, but **species-specific sensory anatomy constrains its neural implementation**. We propose that contralateral olfactory projections reflect an evolutionary tradeoff between **decorrelating bilateral signals** and **preserving antennal position information**. Comparing *Drosophila* (small, immobile antennae) to *Ooceraea biroi* ants (large, mobile antennae used in social interactions) reveals how evolution shapes circuits for distinct ecological demands.

## Progress Toward Proposed Aims

### Aim 1: Bilateral odor asymmetry as a conserved navigational computation

We established that **bilateral odor asymmetry is a key driver of navigation in both ants and flies**, despite major differences in their ecology and sensory anatomy. In *O. biroi*, reanalysis of pheromone trail-following behavior revealed that **moment-to-moment differences in odor input across the two antennae predict turning direction**, with sensory asymmetries preceding motor output by ~400 ms. Importantly, ants dynamically reposition their antennae during navigation, suggesting that **antennal position itself is an integral component of odor-guided behavior**. In contrast, *Drosophila* typically encounters spatially structured odors only rarely. To test whether flies nonetheless possess a similar bilateral computation, we developed **FlyCLOPS (Fly Closed-Loop Optogenetic Projection System)**, which creates spatially structured “fictive” odor trails via optogenetic activation of olfactory receptor neurons. Using this system, we demonstrated that flies exhibit **robust trail-following–like behavior** that depends on intact bilateral antennae. Together, these results indicate that **bilateral odor comparison is a conserved navigational computation**, even in species that differ dramatically in how and why they use olfaction.

### Aim 2: Evolutionary divergence of circuit mechanisms for bilateral processing

A key anatomical difference between ants and flies lies in how early olfactory circuits integrate left–right information. In flies, olfactory receptor neurons project to both hemispheres of the brain, creating extensive **contralateral mixing** of sensory input, whereas in ants—and most other insects—olfactory projections remain strictly **ipsilateral**.
Using whole-brain connectomic analysis in *Drosophila*, we found that contralateral sensory inputs are preferentially routed through inhibitory interneurons, forming circuit motifs well suited to **decorrelating highly similar bilateral signals**. This architecture is particularly relevant given the fly’s small inter-antennal distance and limited antennal mobility, which naturally produce highly correlated sensory input.
Crucially, we propose that this solution is **not universally advantageous**. In ants, antennae are large, widely spaced, and actively moved during navigation and social interaction, inherently producing less-correlated bilateral input and enabling the animal to exploit **antenna position as an informative variable**. Early connectomic observations in ants support this view, as widespread left–right mixing is not observed until **fourth-order olfactory neurons (mushroom body output neurons)**.
Direct contralateral mixing at early sensory stages could therefore compromise positional information, making it more difficult to integrate odor cues with antennal movements and interactions with nestmates or brood. This reframes the absence of contralateral projections in ants not as a limitation, but as an **adaptive constraint** that preserves the coupling between sensory input and body configuration—an especially important requirement for social behavior.

To causally test this hypothesis, we identified a developmental strategy targeting **Neuroglian-expressing commissural neurons** in flies, allowing selective elimination of contralateral olfactory projections. This enables direct tests of how interhemispheric mixing benefits navigation when antennae are small and fixed, moving beyond descriptive anatomy toward **functional and evolutionary explanations of circuit motifs**. In parallel, we developed **ant transgenic lines expressing GCaMP7s in projection neurons and subsets of Kenyon cells**, enabling tracking of odor and antennal-position representations across multiple layers of olfactory processing.

We also developed **MultiBiOS (Multispecies Bilateral Odor-delivery System)**, a precision bilateral odor-delivery platform compatible with functional imaging in both ants and flies. Although rig development was slower than anticipated due to technical complexity, the system is now functional and imaging-capable and is currently undergoing validation, with experiments planned for early 2026.

### Aim 3: Neuromodulation and social context

**Computational modeling** showed that **simple bilateral steering rules with gain modulation generate complex collective behaviors** (recruitment, aggregation), linking circuit-level modulation to emergent social phenomena. This motivated focus on **monoaminergic neuromodulation** as a mechanism for contextual flexibility. In ants, we identified dopaminergic, octopaminergic, and serotonergic populations and generated transgenic lines expressing **PA-GFP and GCaMP8m under Vmat promoter**. Colonies are expanding and ready for histological validation in early 2026, enabling tests of how neuromodulators reshape bilateral computations during social behaviors.

## Setbacks and Adaptive Changes

Ants' preference for arena borders complicated tracking and trail-following assays. We addressed this with new arena geometries and **QR-code–like visual markers** for enhanced identity and pose tracking near edges. FlyCLOPS and MultiBiOS development was slower than expected due to technical complexity, but both systems are now functional and undergoing validation, positioning the project well for the next phase.

## New Avenues

We are developing a broader **evolutionary framework** examining how antennal size, spacing, and mobility across Diptera correlate with contralateral projections and Neuroglian expression. Mosquitoes (lacking contralateral ORN projections, with longer antennae) provide evolutionary comparison to test whether bilateral mixing is a derived feature linked to reduced antennal mobility.

## Broader Impact on Understanding the Social Brain

This work demonstrates that **social behavior emerges from modulation of conserved computations rather than novel circuitry**. Sensory anatomy constrains viable neural solutions: flies use contralateral projections to decorrelate input from small, fixed antennae; ants preserve ipsilateral processing to couple odor, antennal position, and social interaction. By integrating comparative anatomy, circuit manipulation, and modeling, we provide a framework for understanding how evolution balances sensory precision, bodily context, and social demands—likely extending across taxa.
