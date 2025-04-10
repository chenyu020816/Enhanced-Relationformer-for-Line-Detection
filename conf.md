---
authors: ["Chen-Yu(Erioe) Liu", "Yuanhao Shen"]
title: Redefined Relationformer with Improved Matching for Line Detection
# paper_url: /static_files/projects/yang_building_insights.pdf
# video_url: "https://mymedia.bu.edu/media/t/1_rdtih9o5"
# slides_url: /static_files/projects/yang_preso.pdf
tags: ["GeoCV", "Transforme", "DETR"]
categories: ["Algorithm", "Application"]
---

Line detection or Graph generaCtion plays is crucial in various computer vision tasks. Although Relationformer has made significant progress in modeling inter-object relationships, it still faces challenges in endpoint prediction and matching stability. To address these issues, we aim to reproduce and validate Relationformer and its variants.
We further propose a distributed-based approach that replaces the tradictional Dirac delta-based box prediction to improve endpoint localization accuracy. Additionally, inspired by General Focal Loss and D-EIM, we design a loss function combined with data augmentation techniques to reduce the impact of negative samples and enhance model robustness.