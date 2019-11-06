First, keep in mind that jupyter notebooks are a great tool to do quick analysis of data or to build/debug prototypes. Once you have a piece of code working, you probably want to move it into a plain python file that integrates in your project. As an example, you are not going to launch an experiment that takes a training time of 5 hours in a jupyter notebook, but you can understand the architecture you are using by running a forward pass on a batch and inspecting the result in a notebook, for instance.

I believe right now our priority is to build a **Fully-Operative Training/Validation loop**, with suitable hyperparameters exposed to the command line (I wil tell you about this next week). For this you really need to start refactoring your code once it is functional. Please have a look at the list below. It will grow over time, but for now you can start with this:

### Short-term goals
- [ ] Read [this article](https://medium.com/miccai-educational-initiative/project-roadmap-for-the-medical-imaging-student-working-with-deep-learning-351add6066cf) on how to structure and develop machine learning projects for medical image analysis (you can probably ignore the "write tests" part).
- [ ] The code as it is now is quite dirty. Start refactoring by moving pieces of your implementation to python files. For instance, if you have several architectures you want to test, you probably want to create a `models` or an `architectures` folder where you store the model definitions. For handling data (dataloader, datasets, transformations...) you may want to add those to a folder called `utils` or so, while for performance evaluation you may want to create an `evaluation` folder. The training code is typically stored in a `train.py` file in the root folder of the project, while for building predictions on data not used for training, you can create a `test.py` file. Raw data (the images) should go into a `data` folder (I already created that).
- [ ] Understand correctly how this task should be evaluated. For that, carefully study and complete/solve [this notebook](nbs/evaluation.ipynb). Then we can discuss what to use as early-stopping criteria.
- [ ] Train a small U-Net on this problem. Find relevant hyperparameters rigorously by grid search on the validation set. 
- [ ] Evaluate this model on DRIVE's test set where you should hopefully get results consistent with the state of the art.
- [ ] Evaluate the same model on other datasets, find out if performance dicreases. This is what we are expecting, and what will lead to the second part of the project.

Note, you probably will want to add another architecture aside of a standard U-Net as you have now. We will discuss when you have an operative experimental pipeline. In general, think that you want to code such pipeline in such a way that changing architectures should be a simple one-line replacement of the code. For a state-of-the-art review of the different options, please have a look at [this paper](https://arxiv.org/abs/1910.07655), but do not spend much time on it, it's just a big picture of deep learning and segmentation as of 2019.

### Mid-term goals
- [ ] Implement No-Reference Segmentation Quality Metric.
- [ ] Implement Reverse-Classifier Accuracy.
- [ ] Use both to perform Domain Adaptation and improve performance on other datasets.

### End goal
- [ ]  Write and submit paper.


