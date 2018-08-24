#!/usr/bin/env python3

import sys

from mountainlab_pytools import processormanager as pm

import p_whiten_snippets

PM=pm.ProcessorManager()

PM.registerProcessor(p_whiten_snippets.whiten)

if not PM.run(sys.argv):
    exit(-1)