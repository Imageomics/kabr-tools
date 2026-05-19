---
license: cc0-1.0
language:
- en
pretty_name: "kabr_testing"
task_categories: ["object-detection", "video-classification"]
tags:
- biology
- image
- animals
- CV
size_categories: 100K<n<1M
---

<!--

NOTE: Add more tags (your particular animal, type of model and use-case, etc.).

As with your GitHub Project repo, it is important to choose an appropriate license for your dataset. The default license is [CC0](https://creativecommons.org/publicdomain/zero/1.0/) (public domain dedication, see [Dryad's explanation of why to use CC0](https://blog.datadryad.org/2023/05/30/good-data-practices-removing-barriers-to-data-reuse-with-cc0-licensing/)). Alongside the appropriate stakeholders (eg., your PI, co-authors), select a license that is [Open Source Initiative](https://opensource.org/licenses) (OSI) compliant.
For more information on how to choose a license and why it matters, see [Choose A License](https://choosealicense.com) and [A Quick Guide to Software Licensing for the Scientist-Programmer](https://doi.org/10.1371/journal.pcbi.1002598) by A. Morin, et al.
See the [Imageomics policy for licensing](https://docs.google.com/document/d/1SlITG-r7kdJB6C8f4FCJ9Z7o7ccwldZoSRJKjhRAWVA/edit#heading=h.c1sxg0wsiqru) for more information.

See more options for the above information by clicking "edit dataset card" on your repo.

Fill in as much information as you can at each location that says "More information needed".
-->

<!--
Image with caption (jpg or png):
|![Figure #](https://huggingface.co/datasets/imageomics/<data-repo>/resolve/main/<filepath>)|
|:--|
|**Figure #.** [Image of <>](https://huggingface.co/datasets/imageomics/<data-repo>/raw/main/<filepath>) <caption description>.|
-->

<!--
Notes on styling:

To render LaTex in your README, wrap the code in `\\(` and `\\)`. Example: \\(\frac{1}{2}\\)

Escape underscores ("_") with a "\". Example: image\_RGB
-->

# Dataset Card for kabr_testing

<!-- Provide a quick summary of what the dataset is or can be used for. --> 
This dataset is for testing the kabr_tools.

## Dataset Details

### Dataset Description

- **Curated by:** list curators (authors for _data_ citation, moved up)
- **Language(s) (NLP):** [More Information Needed]
<!-- Provide the basic links for the dataset. These will show up on the sidebar to the right of your dataset card ("Curated by" too). -->
- **Homepage:** 
- **Repository:** [related project repo]
- **Paper:** 


<!-- Provide a longer summary of what this dataset is. -->
[More Information Needed]

<!--This dataset card aims to be a base template for new datasets. It has been generated using [this raw template](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/datasetcard_template.md?plain=1), and further altered to suit Imageomics Institute needs.-->


### Supported Tasks and Leaderboards
[More Information Needed]

<!-- Provide benchmarking results -->


## Dataset Structure

<!-- This section provides a description of the dataset fields, and additional information about the dataset structure such as criteria used to create the splits, relationships between data points, etc. -->

<!-- Provide format of the dataset, ex:

```​
/dataset/
    <species_1>/
        <img_id 1>.png
        <img_id 2>.png
        ...
        <img_id n>.png
    <species_2>/
        <img_id 1>.png
        <img_id 2>.png
        ...
        <img_id n>.png
    ...
    <species_N>/
        <img_id 1>.png
        <img_id 2>.png
        ...
        <img_id n>.png
    metadata.csv
```​

-->

### Data Instances
[More Information Needed]

<!--
Describe data files

Ex: All images are named <img_id>.png, each within a folder named for the species. They are 1024 x 1024, and the color has been standardized using <link to color standardization package>.
-->

### Data Fields
[More Information Needed]
<!--
Describe the types of the data files or the columns in a CSV with metadata.

Ex: 
**metadata.csv**:
  - `img_id`: Unique identifier for the dataset. 
  - `specimen_id`: ID of specimen in the image, provided by museum data source. There are multiple images of a single specimen.
  - `species`: Species of the specimen in the image. There are N different species of <genus> of <animal>.
  - `view`: View of the specimen in the image (e.g., `ventral` or `dorsal` OR `top` or `bottom`, etc.; specify options where reasonable).
  - `file_name`: Relative path to image from the root of the directory (`<species>/<img_id>.png`); allows for image to be displayed in the dataset viewer alongside its associated metadata.
-->

### Data Splits
[More Information Needed]
<!--
Give your train-test splits for benchmarking; could be as simple as "split is indicated by the `split` column in the metadata file: `train`, `val`, or `test`." Or perhaps this is just the training dataset and other datasets were used for testing (you may indicate which were used).
-->

## Dataset Creation

### Curation Rationale
[More Information Needed]
<!-- Motivation for the creation of this dataset. For instance, what you intended to study and why that required curation of a new dataset (or if it's newly collected data and why the data was collected (intended use)), etc. -->

### Source Data

<!-- This section describes the source data (e.g., news text and headlines, social media posts, translated sentences, ...). As well as an original source it was created from (e.g., sampling from Zenodo records, compiling images from different aggregators, etc.) -->

#### Data Collection and Processing
[More Information Needed]
<!-- This section describes the data collection and processing process such as data selection criteria, filtering and normalization methods, re-sizing of images, tools and libraries used, etc. 
This is what _you_ did to it following collection from the original source; it will be overall processing if you collected the data initially.
-->

#### Who are the source data producers?
[More Information Needed]
<!-- This section describes the people or systems who originally created the data.

Ex: This dataset is a collection of images taken of the butterfly collection housed at the Ohio State University Museum of Biological Diversity. The associated labels and metadata are the information provided with the collection from biologists that study butterflies and supplied the specimens to the museum.
 -->


### Annotations
<!-- 
If the dataset contains annotations which are not part of the initial data collection, use this section to describe them. 

Ex: We standardized the taxonomic labels provided by the various data sources to conform to a uniform 7-rank Linnean structure. (Then, under annotation process, describe how this was done: Our sources used different names for the same kingdom (both _Animalia_ and _Metazoa_), so we chose one for all (_Animalia_). -->

#### Annotation process
[More Information Needed]
<!-- This section describes the annotation process such as annotation tools used, the amount of data annotated, annotation guidelines provided to the annotators, interannotator statistics, annotation validation, etc. -->

#### Who are the annotators?
[More Information Needed]
<!-- This section describes the people or systems who created the annotations. -->

### Personal and Sensitive Information
[More Information Needed]
<!-- 
For instance, if your data includes people or endangered species. -->


## Considerations for Using the Data
[More Information Needed]
<!--
Things to consider while working with the dataset. For instance, maybe there are hybrids and they are labeled in the `hybrid_stat` column, so to get a subset without hybrids, subset to all instances in the metadata file such that `hybrid_stat` is _not_ "hybrid".
-->

### Bias, Risks, and Limitations
[More Information Needed]
<!-- This section is meant to convey both technical and sociotechnical limitations. Could also address misuse, malicious use, and uses that the dataset will not work well for.-->

<!-- For instance, if your data exhibits a long-tailed distribution (and why). -->

### Recommendations
[More Information Needed]
<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

## Licensing Information
[More Information Needed]

<!-- See notes at top of file about selecting a license. 
If you choose CC0: This dataset is dedicated to the public domain for the benefit of scientific pursuits. We ask that you cite the dataset and journal paper using the below citations if you make use of it in your research.

Be sure to note different licensing of images if they have a different license from the compilation.
ex: 
The data (images and text) contain a variety of licensing restrictions mostly within the CC family. Each image and text in this dataset is provided under the least restrictive terms allowed by its licensing requirements as provided to us (i.e, we impose no additional restrictions past those specified by licenses in the license file).

EOL images contain a variety of licenses ranging from [CC0](https://creativecommons.org/publicdomain/zero/1.0/) to [CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/).
For license and citation information by image, see our [license file](https://huggingface.co/datasets/imageomics/treeoflife-10m/blob/main/metadata/licenses.csv).

This dataset (the compilation) has been marked as dedicated to the public domain by applying the [CC0 Public Domain Waiver](https://creativecommons.org/publicdomain/zero/1.0/). However, images may be licensed under different terms (as noted above).
-->

## Citation
[More Information Needed]

**BibTeX:**
<!--
If you want to include BibTex, replace "<>"s with your info 

**Data**
```​
@misc{<ref_code>,
  author = {<author1 and author2>},
  title = {<title>},
  year = {<year>},
  url = {https://huggingface.co/datasets/imageomics/<dataset_name>},
  doi = {<doi once generated>},
  publisher = {Hugging Face}
}
```​

-for an associated paper:
**Paper**
```​
@article{<ref_code>,
  title    = {<title>},
  author   = {<author1 and author2>},
  journal  = {<journal_name>},
  year     =  <year>,
  url      = {<DOI_URL>},
  doi      = {<DOI>}
}
```​
-->

<!---
If the data is modified from another source, add the following. 

Please be sure to also cite the original data source(s):
<citation>
-->


## Acknowledgements

This work was supported by the [Imageomics Institute](https://imageomics.org), which is funded by the US National Science Foundation's Harnessing the Data Revolution (HDR) program under [Award #2118240](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2118240) (Imageomics: A New Frontier of Biological Information Powered by Knowledge-Guided Machine Learning). Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.

<!-- You may also want to credit the source of your data, i.e., if you went to a museum or nature preserve to collect it. -->

## Glossary 

<!-- [optional] If relevant, include terms and calculations in this section that can help readers understand the dataset or dataset card. -->

## More Information 

<!-- [optional] Any other relevant information that doesn't fit elsewhere. -->

## Dataset Card Authors 

[More Information Needed]

## Dataset Card Contact

[More Information Needed--optional]
<!-- Could include who to contact with questions, but this is also what the "Discussions" tab is for. -->