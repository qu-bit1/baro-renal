# Renal Function and Blood Pressure Regulation Model

A quantitative systems physiology model implemented in Python that simulates renal function and blood pressure regulation. This model captures the complex interactions between cardiovascular, renal, and hormonal systems.

## Overview

This model simulates key physiological processes involved in renal function and its role in maintaining sodium and water homeostasis. It incorporates:

- Systemic Hemodynamics
- Renal Vasculature
- Tubular Function
- Hormonal Regulation (RAAS System)
- Fluid-Electrolyte Balance

## Key Features

- **Comprehensive Physiology**: Models multiple physiological systems and their interactions
- **Dynamic Simulation**: Captures both short-term and long-term regulatory mechanisms
- **Detailed Modeling**: Includes glomerular filtration, tubular function, and hormonal regulation
- **Pathophysiological Insights**: Can simulate various disease states and their effects

## Installation

1. Clone the repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the simulation:
```bash
python run_renal_simulation.py
```

## Dependencies

- numpy
- scipy
- matplotlib
- dataclasses


## Model Components

### Core Model (`renal_model.py`)
The core model class (`RenalModel`) implements the main physiological systems and their interactions:

1. **Systemic Hemodynamics**
   - Blood pressure regulation through cardiac output and vascular resistance
   - Baroreceptor feedback mechanisms
   - Blood volume regulation
   - Tissue autoregulation

2. **Renal Vasculature**
   - Preafferent and afferent arteriole resistance
   - Glomerular filtration pressure
   - Renal blood flow regulation
   - Autoregulation of renal perfusion

3. **RAAS System Dynamics**
   - Renin secretion and regulation
   - Angiotensin I and II conversion
   - Aldosterone synthesis
   - ACE activity modulation

4. **Neural Control Integration**
   - Sympathetic and parasympathetic tone
   - Baroreceptor firing rate
   - Renal sympathetic nerve activity
   - Neural modulation of cardiovascular function

### Tubular Model (`renal_tubular.py`)
The tubular model class (`RenalTubular`) implements renal tubular function and hormonal regulation:

1. **Glomerular Filtration**
   - GFR calculation with circadian variation
   - Filtration fraction determination
   - Oncotic pressure effects

2. **Tubular Function**
   - Proximal tubule sodium and water reabsorption
   - Loop of Henle countercurrent multiplication
   - Distal tubule processing
   - Collecting duct regulation

3. **Hormonal Regulation**
   - ADH (vasopressin) synthesis and action
   - Aldosterone effects on sodium reabsorption
   - Renin-angiotensin system interactions
   - Circadian rhythm influences

4. **Neural-Hormonal Integration**
   - Sympathetic effects on tubular function
   - Neural modulation of hormone release
   - Integrated control of sodium balance

### Neural Control (`neural_mechanisms.py`)
The neural control module (`NeuralControl`) implements autonomic nervous system regulation:

1. **Autonomic Tone**
   - Sympathetic nervous system activity
   - Parasympathetic nervous system activity
   - Baroreceptor reflex integration
   - Neural feedback loops

2. **Cardiovascular Control**
   - Heart rate regulation
   - Stroke volume modulation
   - Cardiac contractility
   - Vascular tone adjustment

3. **Renal Neural Effects**
   - Renal sympathetic nerve activity
   - Neural modulation of renin release
   - Sympathetic effects on sodium handling
   - Neural control of renal blood flow

### Simulation Runner (`run_renal_simulation.py`)
The simulation runner orchestrates the model components and handles:

1. **Model Initialization**
   - Parameter setup
   - Initial state configuration
   - Component integration

2. **Time Integration**
   - ODE system solution
   - State variable updates
   - Component interaction handling

3. **Results Processing**
   - Data collection
   - Time series generation
   - Variable tracking

4. **Visualization**
   - Dynamic response plotting
   - Parameter sensitivity analysis
   - System behavior visualization

## Component Interactions

The model components interact through several key mechanisms:

1. **Hemodynamic-Tubular Coupling**
   - Blood pressure affects glomerular filtration
   - Renal blood flow influences tubular function
   - Tubular reabsorption affects blood volume

2. **Neural-Hormonal Integration**
   - Sympathetic activity modulates hormone release
   - Hormones influence neural tone
   - Integrated control of cardiovascular function

3. **Feedback Systems**
   - Pressure-natriuresis relationship
   - RAAS feedback loops
   - Neural reflex arcs
   - Tubuloglomerular feedback

4. **Time-Scale Integration**
   - Fast neural responses (seconds)
   - Intermediate hormonal changes (minutes-hours)
   - Long-term volume regulation (hours-days)


## Key Parameters

```python
nominal_map_setpoint = 93.0  # mmHg
CO_nom = 5.0  # L/min
blood_volume_nom = 5.0  # L
GFR_nom = 120.0  # ml/min
filtration_fraction_nom = 0.2  # 20%
```

## Simulation Results

The model generates dynamic responses of various physiological variables over time, including:

- Mean Arterial Pressure (MAP)
- Cardiac Output
- Blood Volume
- Plasma Sodium
- Renin-Angiotensin System components
- Aldosterone levels
- ADH and Osmolarity

## Physiological Insights

The model demonstrates:

1. **Multiple Timescales**
   - Immediate (seconds-minutes): Blood pressure, cardiac output
   - Intermediate (hours): RAAS system
   - Long-term (many hours): Aldosterone effects

2. **Feedback Systems**
   - Pressure-natriuresis
   - RAAS feedback
   - ADH-osmolarity control
   - Autoregulation of blood flow

3. **Homeostatic Control**
   - Multiple overlapping mechanisms
   - Different response times
   - Stable steady states
