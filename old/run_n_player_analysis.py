#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""Compatibility wrapper that forwards to experiments.n_player_experiments."""

from experiments.n_player_experiments import run_default_n_player_analysis


def main():
    """Run the standard N-player analysis pipeline."""
    run_default_n_player_analysis()


if __name__ == "__main__":
    main()
