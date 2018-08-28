# ml_ms4alg_snippets

Electrophysiology tools
MountainLab processor library

Clone this repository into $CONDA_PREFIX/etc/mountainlab/packages directory

## Installation

Example: 
```
cd ~/conda/envs/mlab/etc/mountainlab/packages
git clone https://github.com/GeoffBarrett/ml_ms4alg_snippets.git
```

Check that the have been added to the processor list

```
ml-list-processors | grep ms4alg_snippets
```

If you do not see **ms4alg_snippets.sort** and **ms4alg_snippets.whiten** then it is possible that the .mp files do not have permissions. Execute the following.

```
cd ~/conda/envs/mlab/etc/mountainlab/packages/ml_ms4alg_snippets/ml_ms4alg_snippets
chmod a+x ms4alg_snippets_spec.py.mp
chmod a+x whiten_snippets_spec.py.mp
```
