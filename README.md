# COMP4102 Term Project
# Automated Background Retrieval

**Michael Kuang 101000485**

**Kevin Sun 101000157**

**Gordon Brown 101002159**

**Maxim Kuzmenko 101002578**

# Summary
The goal of this project will be to develop an application that will identify and remove detected object(s) from an image, while filling in the background that was in place of said object(s) naturally. For example, if there is a person and a dog at the beach but the beach image is not sufficient, the application will isolate then remove the person and the dog from the image. The white void left behind by the removal of the images will then be filled with the appropriate background so that it would appear the image never contained the objects in the first place.

# Background

We are aware of a publication at  Stanford where they used Viola-Jones face detection to find humans, and then used Exemplar Based Inpainting (EBP) followed by Non-Local Means (NLM) filtering to clean up the inpainting. There are also papers published by Microsoft exploring similar techniques using EBP which we will also explore time permitting. 

We will be exploring the watershed formation, hierarchical segmentation, and the P algorithm for our object detection by comparing their performance.

Exemplar Based Inpainting
https://stanford.edu/class/ee367/Winter2018/li_li_ee367_win18_report.pdf
https://tschumperle.users.greyc.fr/publications/tschumperle_tip15.pdf
https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/criminisi_cvpr2003.pdf
https://www.irisa.fr/vista/Papers/2004_ip_criminisi.pdf

Watershed formation: https://www.mdpi.com/2313-433X/4/10/123/pdf
Hierarchical Segmentation: http://www.sci.utah.edu/publications/Liu2016a/07515198.pdf
P Algorithm: https://hal-mines-paristech.archives-ouvertes.fr/hal-00835019/document


# Challenge
The main challenges present with this project are object segmentation, and background interpolation. For object segmentation we need to detect objects within the imagine as well as crop it out of the image precisely. There already exists various forms of object detection, such as YOLO object detection,  watershed algorithm, and using R-CNNS for instance segmentation. We will explore the various techniques in object detection to determine one that would best fit our application and possibly improve the algorithm to suit our needs.

![Example of Project Procedure](https://github.com/Krusso/COMP4102TermProject/blob/master/Project.png)

Figure 1: Example of project procedure


Background interpolation will be a much more difficult problem given its twofold nature - we must correctly interpolate differing regions of the image as well as convincingly filter the interpolated area to ensure continuity. It is expected that this area will cause the most trouble/challenges as apart from EBP there doesn’t seem to be any efficient ways to interpolate images.


# Goals and Deliverables
The goals that we plan to achieve in this project are:
Detect objects within an image (able to box or lasso objects of interest)
Remove the selected objects
Fill the selected objects based on the surrounding background


A successful project is when the application is able to remove reasonably complex objects (such as a human or a car) from a non-trivial image (i.e house from a forest, not a person in front of a green screen) with no obvious breaks in continuity within the image.

A couple of extra goals that we hope to achieve if our project is going ahead of schedule would be:
Remove objects from noisy backgrounds
Allow user interaction to remove objects

The entire procedure will be that given an image, the program will find the most distinct objects in the image, remove them, and then fill the void left behind with an appropriate background to make the entire image complete and natural. In order to measure how accurate and successful the project is, the final image will be compared to the actual image with the objects removed. The comparison can be done quantitatively using residual errors and other functions to determine if the images have a large visual difference between them. 

# Schedule

**Michael**

Feb 3: Explore object detection algorithms (OD)

Feb 10: Implement 1-2 OD algorithms

Feb 17: Implement 1-2 OD algorithms

Feb 24: Analyse, compare and improve OD algorithms

Mar 2: Analyse, compare and improve OD algorithms and select an algorithm that best segments image.

Mar 9: Integration with rest of group

Mar 16: Explore other methods of object detection or object segmentation

Mar 23: Explore other methods of object detection or object segmentation

Mar 30: Write report and and prepare for presentation

Apr 6: Write report and and prepare for presentation

**Gordon**

Feb 3: Explore interpolation algorithms

Feb 10: Implement EBP 

Feb 17: Implement EBP 

Feb 24: Finalize EBP 

Mar 2: Finalize EBP 

Mar 9: Integration with rest of group

Mar 16: Explore other methods of background interpolation

Mar 23: Explore other methods of background interpolation

Mar 30: Write report and and prepare for presentation

Apr 6: Write report and and prepare for presentation

**Maxim**

Feb 3: Explore OD algorithms

Feb 10: Implement 1-2 OD algorithms

Feb 17: Implement 1-2 OD algorithms

Feb 24: Analyse, compare and improve OD algorithms

Mar 2: Analyse, compare and improve OD algorithms and select an algorithm that best segments image.

Mar 9: Integration with rest of group

Mar 16: Explore other methods of object detection or object segmentation

Mar 23: Explore other methods of object detection or object segmentation

Mar 30: Write report and and prepare for presentation

Apr 6: Write report and and prepare for presentation

**Kevin**

Feb 3: Explore interpolation algorithms

Feb 10: Implement EBP 

Feb 17: Implement EBP 

Feb 24: Finalize EBP 

Mar 2: Finalize EBP 

Mar 9: Integration with rest of group

Mar 16: Explore other methods of background interpolation

Mar 23: Explore other methods of background interpolation

Mar 30: Write report and and prepare for presentation

Apr 6: Write report and and prepare for presentation


