# Entropy-Based Parking Optimization

Dynamic programming approach for optimal parking strategy using entropy model from statistical mechanics.

## Overview

This project applies concepts from statistical mechanics to solve the parking optimization problem. Using an entropy model with a single parameter (temperature), we predict parking occupancy distribution and develop an optimal parking policy through dynamic programming.

## Method

### Entropy Model
- Models parking spot distribution using canonical ensemble from statistical mechanics
- Single parameter (temperature) represents parking lot occupancy
- Energy function based on distance from entrance

### Dynamic Programming
- Optimal policy to minimize expected parking time
- Temperature-Informed Parking Policy (TIPP)
- Real-time decision making based on predicted occupancy

## Results

**25% reduction** in parking search time compared to baseline strategies.

### Policy Comparison
![Policy Comparison](results/policy_comparison.png)

TIPP consistently outperforms benchmark policies across different occupancy levels.

### Performance Metrics
![Parking Results](results/parking_results.png)

Detailed performance analysis showing search time, temperature evolution, and floor distribution.

## Technical Details

**Prediction:** Entropy model predicts spot availability per floor  
**Optimization:** Dynamic programming finds optimal floor selection  
**Simulation:** Python-based parking garage simulator

## Technologies

- Python
- NumPy
- Matplotlib
- Dynamic Programming

## Reference

Based on methodology from:
*"Entropy-Based Dynamic Programming for Efficient Vehicle Parking"*  
Lupien et al., 2024

## Course Project

Dynamic Programming Course, 2024  
Catholic University of Korea
