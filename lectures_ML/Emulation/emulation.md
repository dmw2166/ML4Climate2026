# Climate Model Emulation

```{admonition} Unit under development
:class: note
The lecture notes and tutorials for this unit are still being written. The
outline below describes the material this unit will cover.
```

Running a general circulation model is expensive. A single high-resolution
simulation can consume substantial supercomputer time, which puts a hard limit on
how many scenarios, parameter settings, or ensemble members anyone can afford to
explore.

**Emulation** addresses this by training a cheap statistical model on a limited set
of expensive simulations, then using that surrogate to predict what the full model
would have produced elsewhere in parameter space.

## Planned topics

- What an emulator is, and what it is not
- Choosing training runs: experimental design and parameter space sampling
- Emulating a simple energy balance model
- Emulators for spatial fields, not just global means
- Perturbed parameter ensembles and calibration against observations
- Where emulators fail: extrapolation beyond the training runs, and why this
  matters especially for future scenarios
