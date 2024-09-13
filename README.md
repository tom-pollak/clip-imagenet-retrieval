# CLIP Index

> Evaluating Approximate Nearest Neighbors (ANN) with CLIP.

## [Report](https://tom-pollak.github.io/clip-index/assets/Enhancing%20Image%20Retrieval%20in%20Natural%20Language%20Processing%20Applications.pdf)

## Summary

The increasing amount of images and digital media available today
has led to a growing demand for efficient and general methods to
retrieve relevant images from large heterogeneous image databases.

This dissertation proposes a novel image retrieval approach
combining natural language processing (NLP) and computer vision,
aiming to deliver a powerful image retrieval system capable of
searching image databases using any natural language query. I
leverage OpenAI’s Contrastive Language-Image Pretraining (CLIP)
multimodal model to predict the similarity between an image and
text query, and integrate this with an Approximate k-nearest
neighbour (ANN) index to scale to tens of thousands of images. This
approach aims to provide an intuitive and powerful search
experience, not bound by the constraints of a predetermined set of
labels or categories.

This research will assess the effectiveness of the image retrieval
system through the merits and tradeoffs of different model
architectures and ANN index algorithms, considering factors such as
inference speed, model size, accuracy, recall, build time and query
time. The significance of these factors may vary depending on the
objectives of the production system. By exploring these aspects,
this study aims to provide an image retrieval guide for different use
cases.
