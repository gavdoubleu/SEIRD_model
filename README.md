# Code for fitting a coupled SEIRD model to deaths per day data for multiple groups (e.g. age groups, sex).

Gavin Woolman 16/Sept/2025

## Theory

### Revision of the SEIRD model.

The SEIRD model (Susceptible, Exposed, Infectious, Recovered, Dead) is one of the canonical compartmental models of disease. Each compartment has a population, and a simple deterministic model describes the rate of change of the population of each compartment.

In the simplest form, the SEIRD model differential equations are as follows:

\begin{align}
\frac{\text{d}S}{\text{d}t} &= -\beta S\frac{I}{N - D},\\
\frac{\text{d}E}{\text{d}t} &= +\beta \frac{I}{N - D} - \frac{E}{l},\\
\frac{\text{d}I}{\text{d}t} &= \frac{E}{l} - \left(\gamma + \delta\right)I,\\
\frac{\text{d}R}{\text{d}t} &= \gamma I,\\
\frac{\text{d}D}{\text{d}t} &= \delta I,\\
\end{align}
where $\beta$ is the transmission coefficient, $l$ is the latent period between exposure and infectiousness, $\gamma$ is the recovery rate, and $\delta$ is the death rate.

### Coupled SEIRD model

This code creates a coupled-SEIRD model, imagining $N$ groups within the population that can meet and mix in one of $M$ mixing environments. Each mixing environment $m$ has its own transmission coefficient $\beta_m$. The distribution of population groups $n$ amongst mixing environments $m$ is given by an $M$ by $N$ matrix $\theta^m_n$. This makes the new differential equations:
\begin{align}
\frac{\text{d}S^g}{\text{d}t} &= \sum_m  -\beta_m \theta^{m}_{g} S^{g} \frac{\sum_g \theta^m_g I^g}{\sum_{g'} \theta^m_{g'}\left(N^{g'} - D^{g'}\right)},\\
\frac{\text{d}E^g}{\text{d}t} &= +\frac{\text{d}S^g}{\text{d}t} - \frac{E^g}{l_g},\\
\frac{\text{d}I^g}{\text{d}t} &= \frac{E^g}{l_g} - \left(\gamma_g + \delta_g\right)I^g,\\
\frac{\text{d}R^g}{\text{d}t} &= \gamma_g I^g,\\
\frac{\text{d}D^g}{\text{d}t} &= \delta_g I^g,\\
\end{align}
where $\beta_m$ is the transmission coefficient for mixing environment $m$, and $l_g$ is the latent period, $\gamma_g$ the recovery rate, and $\delta_g$ the death rate for group $g$.


## Code structure

The code is partitioned into several parts:


- The main script is given in the notebook `SEIRD model.ipynb`. This lays out examples for how to use the code.

- The differential equations that govern how the coupled SEIRD model works are laid out in `SEIRD_model.py`. This file shouldn't need changed unless something fundamental about the model is being altered.

- A class used to manage fitting the model to data is set out in `SEIRD_fitting.py`. Again, this class shouldn't really need changed unless something fundamental about the fitting process is being altered.

- The main class that is to be changed is `ModelHandler`, defined in the jupyter notebook itself. This defines how the fitting parameters are put into the model, and sets initial conditions (e.g. the compartment populations on the first day of infection, and the date of the first day of infection).

