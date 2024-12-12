# Exercise 2 - Q-Learning for CartPole

This repository contains the implementation of Exercise 2 for Q-Learning applied to the CartPole environment.

## Overview

This exercise builds upon the concepts of the q-learning. It consists of four different way to complete the experiment, each using a methodology for exploration and Q-table initialization.

## Methodologies

The behavior of the Q-learning agent depends on the value of the SCHEDULE constant:

- CONSTANT:

   - Epsilon is fixed at 0.2.
   - Q-table is initialized with all zeros.

- GLIE:

   - Epsilon decreases according to a specific formula, reaching ￼ after 20,000 episodes.
   - Q-table is initialized with all zeros.

- ZERO:

   - Epsilon is set to 0 (no exploration).
   - Q-table is initialized with all zeros.

- ZERO_FIFTY:

   - Epsilon is set to 0.
   - Q-table is initialized with all values set to 50.

For each methodology, the agent can be either trained or tested by setting the value of the MODE constant.

## Repository Structure

<b>data/ </b>

Contains data generated or used during the experiment:

- <b>images/:</b> Contains images (sourced from the internet) used to explain concepts in the accompanying documentation.
- <b>log/:</b> CSV files for analyzing reward values across episodes.
- <b>model/:</b> Saved models for each phase, including:

   - Q-table snapshots before training, after one episode, halfway through training, and at the end of training.
   - Reward progression data for each phase.

- <b> plot/:</b> Contains plots generated during the exercise, including:

   - Reward progression over episodes for each phase.
   - Heatmaps of the Q-value function before training, after one episode, halfway through training, and at the end of training.

<b>qlearning.py</b>

The main script containing the implementation of the Q-learning algorithm and the logic for each phase.

## Instructions

Refer to the assignment document for detailed instructions on running and evaluating the experiment.

## Contact me

For inquiries or issues, please reach out to the maintainer:
Email: s329706@studenti.polito.it