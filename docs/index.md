<!-- The main content of this page is generated from the repository README.md
     via pymdownx.snippets (the --8<-- line below). Edit README.md at the repo
     root to change it. The "Getting Started" and "Additional Resources" sections
     below are docs-site-only navigation and are maintained here. -->
--8<-- "README.md"

## Getting Started

To get started with Etho tools, follow the pipeline steps in order:

1. **[Data Collection](pipeline/data-collection.md)** - Collect drone video data following best practices
2. **[Pre-processing](pipeline/preprocessing.md)** - Use CVAT to create mini-scenes from your videos  
3. **[Behavior Labeling](pipeline/behavior-labeling.md)** - Apply machine learning models to classify behaviors
4. **[Analysis](pipeline/analysis.md)** - Generate ecological insights and visualizations

## Additional Resources

- [KABR Project Page](https://imageomics.github.io/KABR/) for additional details on the dataset and original paper.
- [KABR Mini-Scene Dataset on Hugging Face](https://huggingface.co/datasets/imageomics/KABR)
- [Pre-trained Model](https://huggingface.co/imageomics/x3d-kabr-kinetics)
- [KABR Collection on Hugging Face](https://huggingface.co/collections/imageomics/kabr-664dff304d29e6cd7b8e1a00): All datasets and models associated to the KABR Project.
- See also the legacy notebooks (`datasets`, `processing`, and `train_yolo`, the last of which is documented in [Pipeline - Optional Steps](pipeline/optional-steps.md)) used for the original KABR mini-scene dataset creation.
