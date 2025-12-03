# Step 4: Ecological Analysis

## Overview

After labeling mini-scenes with behaviors, you can perform various ecological analyses to generate insights from your behavioral data. The KABR tools framework supports multiple types of analysis including time budgets, social interactions, and behavioral transitions.

## Time Budget Analysis

Time budget analysis shows how animals allocate their time across different behaviors during observation periods.

### Example Analysis

<div style="display: flex; justify-content: space-between;">
  <img src="https://github.com/Imageomics/kabr-tools/blob/main/docs/images/01_18_session_7_flightpath.png" alt="drone_telemetry" style="width: 48%;">
  <img src="https://github.com/Imageomics/kabr-tools/blob/main/docs/images/grevys.png" alt="grevys" style="width: 48%;">
</div>

**Figure 1:** Example flight path and video clip from KABR dataset: two male Grevy's zebras observed for 10 minutes on 01/18/23.

![Time Budget](../images/timebudget.png)

**Figure 2:** Overall time budget for duration of 10 minute observation.

![Timeline 1](../images/timeline0.png)
![Timeline 2](../images/timeline1.png)

**Figure 3:** Gantt chart for each zebra (3 minute duration).

### Implementation

See the [time budgets notebook](../case_studies/0_time_budget/time_budget.ipynb) for the code to create these visualizations.

## Social Interactions

Social interaction analysis examines how different species and individuals interact within the same environment, including proximity patterns and mixed-species associations.

### Implementation

See the [social interactions notebook](../case_studies/3_mixed_species_social/mixed_species_overlap.ipynb) for the code to create these visualizations.

## Behavior Transitions

Behavioral transition analysis reveals patterns in how animals move between different behavioral states over time, providing insights into behavioral sequences and decision-making processes.

### Implementation

See the [behavior transitions notebook](../case_studies/2_zebra_transition/behaviortransitionsheatmap.ipynb) for the code to create these visualizations.

## Case Studies

Explore real-world applications in the case studies directory:

- **[Grevy's Landscape Analysis](../case_studies/1_grevys_landscape/grevys_landscape_graphs.ipynb)** - Landscape-scale behavioral analysis.
- **[Mixed Species Social](../case_studies/3_mixed_species_social/mixed_species_overlap.ipynb)** - Multi-species interaction analysis.
- **[Zebra Behavior Transitions](../case_studies/2_zebra_transition/behaviortransitionsheatmap.ipynb)** - Behavioral transition patterns.

## Key Metrics Generated

The analysis pipeline can generate several key ecological metrics:

1. **Time Budgets** - Proportion of time spent in each behavioral category.
2. **Behavioral Transitions** - Probability matrices of behavior changes.
3. **Social Interactions** - Proximity and interaction frequency between individuals.
4. **Habitat Associations** - Relationship between behaviors and spatial locations.
5. **Group Composition Dynamics** - Changes in group structure over time.

## Visualization Tools

The framework includes tools for creating publication-ready visualizations:

- Gantt charts for individual behavioral timelines.
- Heat maps for behavioral transition probabilities.
- Spatial plots for habitat use patterns.
- Social network diagrams for interaction patterns.
- Time series plots for behavioral trends.

## Next Steps

For additional customization and advanced features, see [Optional Steps](optional-steps.md) which covers fine-tuning YOLO models and additional utility tools.