# Step 4: Ecological Analysis

## Overview

After labeling mini-scenes with behaviors, you can perform various ecological analyses to generate insights from your behavioral data. The Etho tools framework supports multiple types of analysis including time budgets, social interactions, and behavioral transitions.
- **Time budget analysis** shows how animals allocate their time across different behaviors during observation periods. See the [worked example](worked-example.md) for code to create time budget visualizations.
- **Social interaction** analysis examines how different species and individuals interact within the same environment, including proximity patterns and mixed-species associations. See the [social interactions notebook](../case_studies/3_mixed_species_social/mixed_species_overlap.ipynb) for the code to create these visualizations.
- **Behavioral transition** analysis reveals patterns in how animals move between different behavioral states over time, providing insights into behavioral sequences and decision-making processes. See the [behavior transitions notebook](../case_studies/2_zebra_transition/behaviortransitionsheatmap.ipynb) for the code to create these visualizations.

## Case Studies

Explore real-world applications in the case studies directory:

- **[Grevy's Landscape Analysis](../case_studies/overview.md#case-study-1-grevys-landscape-of-fear)** - Landscape-scale behavioral analysis.
- **[Zebra Behavior Transitions](../case_studies/overview.md#case-study-2-zebra-state-transitions)** - Behavioral transition patterns.
- **[Mixed Species Social](../case_studies/overview.md#case-study-3-mixed-species-social-interactions)** - Multi-species interaction analysis.
- **[African Wild Dog Individual Monitoring](../case_studies/overview.md#case-study-4-african-wild-dog-individual-behavioral-monitoring)** - Individual-level behavioral monitoring from camera trap videos.

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