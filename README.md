# PTMs

预训练模型相关资料
补充领域介绍文档
每篇论文入选理由，1-3句话介绍该论文亮点。
随时更新

##  论文清单

### 综述

1.  论文：Pre-trained Models for Natural Language Processing: A Survey</br>地址:https://arxiv.org/pdf/2003.08271.pdf</br>复旦大学邱锡鹏等学者发布了的的自然语言处理处理中预训练模型PTMs的综述大全，共25页pdf205篇参考文献，从背景知识到当前代表性PTM模型和应用研究挑战等，是绝好的预训练语言模型的文献。

### 领域必读（10篇）

1. **context2vec: Learning Generic Context Embedding with Bidirectional LSTM**. *Oren Melamud, Jacob Goldberger, Ido Dagan*. CoNLL 2016. [[pdf](https://www.aclweb.org/anthology/K16-1006.pdf)] [[project](http://u.cs.biu.ac.il/~nlp/resources/downloads/context2vec/)] (**context2vec**).

   提出了一种无监督模型，借助双向LSTM从大型语料库中有效学习通用句子上下文表征。

2. Attention is all you need.[1] Vaswani, A. ,  Shazeer, N. et al. (2017).

   最强特征提取器,超越RNN的模型之作,BERT模型的重要组成部分.

3. Deep contextualized word representations.*Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee and Luke Zettlemoyer*. NAACL 2018. [[pdf](https://arxiv.org/pdf/1802.05365.pdf)] [[project](https://allennlp.org/elmo)] (**ELMo**).

   解决一词多义的动态词向量模型. 

4. **Improving Language Understanding by Generative Pre-Training**. *Alec Radford, Karthik Narasimhan, Tim Salimans and Ilya Sutskever*. Preprint. [[pdf](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)] [[project](https://openai.com/blog/language-unsupervised/)] (**GPT**).

   在这篇论文中，探索出了一种对自然语言理解任务的半监督方法，融合了无监督的预训练(pre-training)和有监督的微调(fine-tuning)过程。

5. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova*. NAACL 2019. [[pdf](https://arxiv.org/pdf/1810.04805.pdf)] [[code & model](https://github.com/google-research/bert)]. 

   划时代意义的预训练模型.

6. **ERNIE: Enhanced Language Representation with Informative Entities**. *Zhengyan Zhang, Xu Han, Zhiyuan Liu, Xin Jiang, Maosong Sun and Qun Liu*. ACL 2019. [[pdf](https://www.aclweb.org/anthology/P19-1139)] [[code & model](https://github.com/thunlp/ERNIE)] (**ERNIE (Tsinghua)** ).

   利用了大规模的语料信息和知识图谱，去训练一个增强的语言表示模型，它能够同时利用词汇、语义和知识信息。

7. **Defending Against Neural Fake News**. *Rowan Zellers, Ari Holtzman, Hannah Rashkin, Yonatan Bisk, Ali Farhadi, Franziska Roesner, Yejin Choi*. NeurIPS 2019. [[pdf](https://arxiv.org/pdf/1905.12616.pdf)] [[project](https://rowanzellers.com/grover/)] (**Grover**).

   本文讨论了不同的自然语言处理方法，以开发出对神经假新闻的强大防御，包括使用GPT-2检测器模型和Grover（AllenNLP）

8. Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks. *Suchin Gururangan, Ana Marasović, Swabha Swayamdipta, Kyle Lo, Iz Beltagy, Doug Downey, Noah A. Smith*. ACL 2020. [[pdf](https://www.aclweb.org/anthology/2020.acl-main.740.pdf)]. 

   ACL2020最佳论文,预训练训练与微调使用trick.

9. **ALBERT: A Lite BERT for Self-supervised Learning of Language Representations**.  *Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut*. ICLR 2020. [[pdf](https://openreview.net/pdf?id=H1eA7AEtvS)].

   一种针对Bert的模型压缩模型.

10. **Language Models are Unsupervised Multitask Learners**. *Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei and Ilya Sutskever*. Preprint. [[pdf](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)] [[code](https://github.com/openai/gpt-2)] (**GPT-2**).

    多任务预训练+超大数据集+超大规模模型
### 较新重要（2-3年内，20篇)
1. **TinyBERT: Distilling BERT for Natural Language Understanding**. *Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang Wang, Qun Liu*. Preprint. [[pdf](https://arxiv.org/pdf/1909.10351v2.pdf)] [[code & model](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT)]
2. **Distilling Task-Specific Knowledge from BERT into Simple Neural Networks**. *Raphael Tang, Yao Lu, Linqing Liu, Lili Mou, Olga Vechtomova, Jimmy Lin*. Preprint. [[pdf](https://arxiv.org/pdf/1903.12136.pdf)]
3. **Patient Knowledge Distillation for BERT Model Compression**. *Siqi Sun, Yu Cheng, Zhe Gan, Jingjing Liu*. EMNLP 2019. [[pdf](https://arxiv.org/pdf/1908.09355.pdf)] [[code](https://github.com/intersun/PKD-for-BERT-Model-Compression)]
4. **Model Compression with Multi-Task Knowledge Distillation for Web-scale Question Answering System**. *Ze Yang, Linjun Shou, Ming Gong, Wutao Lin, Daxin Jiang*. Preprint. [[pdf](https://arxiv.org/pdf/1904.09636.pdf)]
5. **PANLP at MEDIQA 2019: Pre-trained Language Models, Transfer Learning and Knowledge Distillation**. *Wei Zhu, Xiaofeng Zhou, Keqiang Wang, Xun Luo, Xiepeng Li, Yuan Ni, Guotong Xie*. The 18th BioNLP workshop. [[pdf](https://www.aclweb.org/anthology/W19-5040)]
6. **Improving Multi-Task Deep Neural Networks via Knowledge Distillation for Natural Language Understanding**. *Xiaodong Liu, Pengcheng He, Weizhu Chen, Jianfeng Gao*. Preprint. [[pdf](https://arxiv.org/pdf/1904.09482.pdf)] [[code & model](https://github.com/namisan/mt-dnn)]
7. **Well-Read Students Learn Better: The Impact of Student Initialization on Knowledge Distillation**. *Iulia Turc, Ming-Wei Chang, Kenton Lee, Kristina Toutanova*. Preprint. [[pdf](https://arxiv.org/pdf/1908.08962.pdf)]
8. **Small and Practical BERT Models for Sequence Labeling**. *Henry Tsai, Jason Riesa, Melvin Johnson, Naveen Arivazhagan, Xin Li, Amelia Archer*. EMNLP 2019. [[pdf](https://arxiv.org/pdf/1909.00100.pdf)]
9. **Q-BERT: Hessian Based Ultra Low Precision Quantization of BERT**. *Sheng Shen, Zhen Dong, Jiayu Ye, Linjian Ma, Zhewei Yao, Amir Gholami, Michael W. Mahoney, Kurt Keutzer*. Preprint. [[pdf](https://arxiv.org/pdf/1909.05840.pdf)]
10. **ALBERT: A Lite BERT for Self-supervised Learning of Language Representations**.  *Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut*. ICLR 2020. [[pdf](https://openreview.net/pdf?id=H1eA7AEtvS)]
11. **Extreme Language Model Compression with Optimal Subwords and Shared Projections**. *Sanqiang Zhao, Raghav Gupta, Yang Song, Denny Zhou*. Preprint. [[pdf](https://arxiv.org/pdf/1909.11687)]
12. **DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter**. *Victor Sanh, Lysandre Debut, Julien Chaumond, Thomas Wolf*. Preprint. [[pdf](https://arxiv.org/pdf/1910.01108)]
13. **Reducing Transformer Depth on Demand with Structured Dropout**.  *Angela Fan, Edouard Grave, Armand Joulin*. ICLR 2020. [[pdf](https://openreview.net/pdf?id=SylO2yStDr)]
14. **Thieves on Sesame Street! Model Extraction of BERT-based APIs**. *Kalpesh Krishna, Gaurav Singh Tomar, Ankur P. Parikh, Nicolas Papernot, Mohit Iyyer*. ICLR 2020. [[pdf](https://openreview.net/pdf?id=Byl5NREFDr)]
15. **DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference**. *Ji Xin, Raphael Tang, Jaejun Lee, Yaoliang Yu, Jimmy Lin*. ACL 2020. [[pdf](https://www.aclweb.org/anthology/2020.acl-main.204.pdf)]
16. **Contrastive Distillation on Intermediate Representations for Language Model Compression**. *Siqi Sun, Zhe Gan, Yuwei Fang, Yu Cheng, Shuohang Wang, Jingjing Liu*. EMNLP 2020. [[pdf](https://www.aclweb.org/anthology/2020.emnlp-main.36)]
17. **BERT-of-Theseus: Compressing BERT by Progressive Module Replacing**. *Canwen Xu, Wangchunshu Zhou, Tao Ge, Furu Wei, Ming Zhou*. EMNLP 2020. [[pdf](https://www.aclweb.org/anthology/2020.emnlp-main.633.pdf)]
18. **TernaryBERT: Distillation-aware Ultra-low Bit BERT**. *Wei Zhang, Lu Hou, Yichun Yin, Lifeng Shang, Xiao Chen, Xin Jiang, Qun Liu*. EMNLP 2020. [[pdf](https://www.aclweb.org/anthology/2020.emnlp-main.37)]
19. **When BERT Plays the Lottery, All Tickets Are Winning**. *Sai Prasanna, Anna Rogers, Anna Rumshisky*. EMNLP 2020. [[pdf](https://www.aclweb.org/anthology/2020.emnlp-main.259.pdf)]
20. **Funnel-Transformer: Filtering out Sequential Redundancy for Efficient Language Processing**. *Zihang Dai, Guokun Lai, Yiming Yang, Quoc Le*. NeurIPS 2020. [[pdf](https://papers.nips.cc/paper/2020/file/2cd2915e69546904e4e5d4a2ac9e1652-Paper.pdf)]
21. **DynaBERT: Dynamic BERT with Adaptive Width and Depth**. *Lu Hou, Zhiqi Huang, Lifeng Shang, Xin Jiang, Xiao Chen, Qun Liu*. NeurIPS 2020. [[pdf](https://papers.nips.cc/paper/2020/file/6f5216f8d89b086c18298e043bfe48ed-Paper.pdf)]
22. **BERT Loses Patience: Fast and Robust Inference with Early Exit**. *Wangchunshu Zhou, Canwen Xu, Tao Ge, Julian McAuley, Ke Xu, Furu Wei*. NeurIPS 2020. [[pdf](https://papers.nips.cc/paper/2020/file/d4dd111a4fd973394238aca5c05bebe3-Paper.pdf)]
### 最新可读（1年内，无篇数上限）
1. **Revealing the Dark Secrets of BERT**. *Olga Kovaleva, Alexey Romanov, Anna Rogers, Anna Rumshisky*. EMNLP 2019. [[pdf](https://arxiv.org/abs/1908.08593)] 
2. **How Does BERT Answer Questions? A Layer-Wise Analysis of Transformer Representations**. *Betty van Aken, Benjamin Winter, Alexander Löser, Felix A. Gers*. CIKM 2019. [[pdf](https://arxiv.org/pdf/1909.04925.pdf)]
3. **Are Sixteen Heads Really Better than One?**. *Paul Michel, Omer Levy, Graham Neubig*. Preprint. [[pdf](https://arxiv.org/pdf/1905.10650.pdf)] [[code](https://github.com/pmichel31415/are-16-heads-really-better-than-1)]
4. **Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment**. *Di Jin, Zhijing Jin, Joey Tianyi Zhou, Peter Szolovits*. Preprint. [[pdf](https://arxiv.org/pdf/1907.11932.pdf)] [[code](https://github.com/jind11/TextFooler)]
5. **BERT has a Mouth, and It Must Speak: BERT as a Markov Random Field Language Model**. *Alex Wang, Kyunghyun Cho*. NeuralGen 2019. [[pdf](https://arxiv.org/pdf/1902.04094.pdf)] [[code](https://github.com/nyu-dl/bert-gen)]
6. **Linguistic Knowledge and Transferability of Contextual Representations**. *Nelson F. Liu, Matt Gardner, Yonatan Belinkov, Matthew E. Peters, Noah A. Smith*. NAACL 2019. [[pdf](https://www.aclweb.org/anthology/N19-1112)]
7. **What Does BERT Look At? An Analysis of BERT's Attention**. *Kevin Clark, Urvashi Khandelwal, Omer Levy, Christopher D. Manning*. BlackBoxNLP 2019. [[pdf](https://arxiv.org/pdf/1906.04341.pdf)] [[code](https://github.com/clarkkev/attention-analysis)]
8. **Open Sesame: Getting Inside BERT's Linguistic Knowledge**. *Yongjie Lin, Yi Chern Tan, Robert Frank*. BlackBoxNLP 2019. [[pdf](https://arxiv.org/pdf/1906.01698.pdf)] [[code](https://github.com/yongjie-lin/bert-opensesame)]
9. **Analyzing the Structure of Attention in a Transformer Language Model**. *Jesse Vig, Yonatan Belinkov*. BlackBoxNLP 2019. [[pdf](https://arxiv.org/pdf/1906.04284.pdf)]
10. **Blackbox meets blackbox: Representational Similarity and Stability Analysis of Neural Language Models and Brains**. *Samira Abnar, Lisa Beinborn, Rochelle Choenni, Willem Zuidema*. BlackBoxNLP 2019. [[pdf](https://arxiv.org/pdf/1906.01539.pdf)]
11. **BERT Rediscovers the Classical NLP Pipeline**. *Ian Tenney, Dipanjan Das, Ellie Pavlick*. ACL 2019. [[pdf](https://www.aclweb.org/anthology/P19-1452)]
12. **How multilingual is Multilingual BERT?**. *Telmo Pires, Eva Schlinger, Dan Garrette*. ACL 2019. [[pdf](https://www.aclweb.org/anthology/P19-1493)]
13. **What Does BERT Learn about the Structure of Language?**. *Ganesh Jawahar, Benoît Sagot, Djamé Seddah*. ACL 2019. [[pdf](https://www.aclweb.org/anthology/P19-1356)]
14. **Beto, Bentz, Becas: The Surprising Cross-Lingual Effectiveness of BERT**. *Shijie Wu, Mark Dredze*. EMNLP 2019. [[pdf](https://arxiv.org/pdf/1904.09077.pdf)]
15. **How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings**. *Kawin Ethayarajh*. EMNLP 2019. [[pdf](https://arxiv.org/pdf/1909.00512.pdf)]
16. **Probing Neural Network Comprehension of Natural Language Arguments**. *Timothy Niven, Hung-Yu Kao*. ACL 2019. [[pdf](https://www.aclweb.org/anthology/P19-1459)] [[code](https://github.com/IKMLab/arct2)]
17. **Universal Adversarial Triggers for Attacking and Analyzing NLP**. *Eric Wallace, Shi Feng, Nikhil Kandpal, Matt Gardner, Sameer Singh*. EMNLP 2019. [[pdf](https://arxiv.org/pdf/1908.07125.pdf)] [[code](https://github.com/Eric-Wallace/universal-triggers)]
18. **The Bottom-up Evolution of Representations in the Transformer: A Study with Machine Translation and Language Modeling Objectives**. *Elena Voita, Rico Sennrich, Ivan Titov*. EMNLP 2019. [[pdf](https://arxiv.org/pdf/1909.01380.pdf)]
19. **Do NLP Models Know Numbers? Probing Numeracy in Embeddings**. *Eric Wallace, Yizhong Wang, Sujian Li, Sameer Singh, Matt Gardner*. EMNLP 2019. [[pdf](https://arxiv.org/pdf/1909.07940.pdf)]
20. **Investigating BERT's Knowledge of Language: Five Analysis Methods with NPIs**. *Alex Warstadt, Yu Cao, Ioana Grosu, Wei Peng, Hagen Blix, Yining Nie, Anna Alsop, Shikha Bordia, Haokun Liu, Alicia Parrish, Sheng-Fu Wang, Jason Phang, Anhad Mohananey, Phu Mon Htut, Paloma Jeretič, Samuel R. Bowman*. EMNLP 2019. [[pdf](https://arxiv.org/pdf/1909.02597.pdf)] [[code](https://github.com/alexwarstadt/data_generation)]
21. **Visualizing and Understanding the Effectiveness of BERT**. *Yaru Hao, Li Dong, Furu Wei, Ke Xu*. EMNLP 2019. [[pdf](https://arxiv.org/pdf/1908.05620.pdf)]
22. **Visualizing and Measuring the Geometry of BERT**. *Andy Coenen, Emily Reif, Ann Yuan, Been Kim, Adam Pearce, Fernanda Viégas, Martin Wattenberg*. NeurIPS 2019. [[pdf](https://arxiv.org/pdf/1906.02715.pdf)]
23. **On the Validity of Self-Attention as Explanation in Transformer Models**. *Gino Brunner, Yang Liu, Damián Pascual, Oliver Richter, Roger Wattenhofer*. Preprint. [[pdf](https://arxiv.org/pdf/1908.04211.pdf)]
24. **Transformer Dissection: An Unified Understanding for Transformer's Attention via the Lens of Kernel**. *Yao-Hung Hubert Tsai, Shaojie Bai, Makoto Yamada, Louis-Philippe Morency, Ruslan Salakhutdinov*. EMNLP 2019. [[pdf](https://arxiv.org/pdf/1908.11775.pdf)]
25. **Language Models as Knowledge Bases?** *Fabio Petroni, Tim Rocktäschel, Patrick Lewis, Anton Bakhtin, Yuxiang Wu, Alexander H. Miller, Sebastian Riedel*. EMNLP 2019, [[pdf](https://arxiv.org/pdf/1909.01066.pdf)] [[code](https://github.com/facebookresearch/LAMA)]
26. **To Tune or Not to Tune? Adapting Pretrained Representations to Diverse Tasks**. *Matthew E. Peters, Sebastian Ruder, Noah A. Smith*. RepL4NLP 2019, [[pdf](https://www.aclweb.org/anthology/W19-4302.pdf)]
27. **On the Cross-lingual Transferability of Monolingual Representations**. *Mikel Artetxe, Sebastian Ruder, Dani Yogatama*. Preprint, [[pdf](https://arxiv.org/pdf/1910.11856.pdf)] [[dataset](https://github.com/deepmind/XQuAD)]
28. **A Structural Probe for Finding Syntax in Word Representations**. *John Hewitt, Christopher D. Manning*. NAACL 2019. [[pdf](https://www.aclweb.org/anthology/N19-1419.pdf)]
29. **Assessing BERT’s Syntactic Abilities**. *Yoav Goldberg*. Technical Report. [[pdf](https://arxiv.org/pdf/1901.05287.pdf)]
30. **What do you learn from context? Probing for sentence structure in contextualized word representations**. *Ian Tenney, Patrick Xia, Berlin Chen, Alex Wang, Adam Poliak, R. Thomas McCoy, Najoung Kim, Benjamin Van Durme, Samuel R. Bowman,
    Dipanjan Das, and Ellie Pavlick*. ICLR 2019. [[pdf](https://arxiv.org/pdf/1905.06316.pdf)]
31. **Can You Tell Me How to Get Past Sesame Street? Sentence-Level Pretraining Beyond Language Modeling**. *Alex Wang, Jan Hula, Patrick Xia, Raghavendra Pappagari, R. Thomas McCoy, Roma Patel, Najoung Kim, Ian Tenney, Yinghui Huang, Katherin Yu, Shuning Jin, Berlin Chen, Benjamin Van Durme, Edouard Grave, Ellie Pavlick, Samuel R. Bowman*. ACL 2019. [[pdf](https://arxiv.org/pdf/1812.10860.pdf)]
32. **BERT is Not an Interlingua and the Bias of Tokenization**. *Jasdeep Singh, Bryan McCann, Richard Socher, and Caiming Xiong*. DeepLo 2019. [[pdf](https://www.aclweb.org/anthology/D19-6106.pdf)] [[dataset](https://github.com/salesforce/xnli_extension)]
33. **What BERT is not: Lessons from a new suite of psycholinguistic diagnostics for language models**. *Allyson Ettinger*. Preprint. [[pdf](https://arxiv.org/pdf/1907.13528)] [[code](https://github.com/aetting/lm-diagnostics)]
34. **How Language-Neutral is Multilingual BERT?**. *Jindřich Libovický, Rudolf Rosa, and Alexander Fraser*. Preprint. [[pdf](https://arxiv.org/pdf/1911.03310)]
35. **Cross-Lingual Ability of Multilingual BERT: An Empirical Study**. *Karthikeyan K, Zihan Wang, Stephen Mayhew, Dan Roth*. ICLR 2020. [[pdf](https://openreview.net/pdf?id=HJeT3yrtDr)]
36. **Finding Universal Grammatical Relations in Multilingual BERT**. *Ethan A. Chi, John Hewitt, Christopher D. Manning*. ACL 2020. [[pdf](https://www.aclweb.org/anthology/2020.acl-main.493.pdf)]
37. **Negated and Misprimed Probes for Pretrained Language Models: Birds Can Talk, But Cannot Fly**. *Nora Kassner, Hinrich Schütze*. ACL 2020. [[pdf](https://www.aclweb.org/anthology/2020.acl-main.698.pdf)]
38. **Perturbed Masking: Parameter-free Probing for Analyzing and Interpreting BERT**. *Zhiyong Wu, Yun Chen, Ben Kao, Qun Liu*. ACL 2020. [[pdf](https://www.aclweb.org/anthology/2020.acl-main.383.pdf)]
39. **Birds have four legs?! NumerSense: Probing Numerical Commonsense Knowledge of Pre-trained Language Models**. *Bill Yuchen Lin, Seyeon Lee, Rahul Khanna and Xiang Ren*. EMNLP 2020. [[pdf](https://www.aclweb.org/anthology/2020.emnlp-main.557)]
40. **Identifying Elements Essential for BERT’s Multilinguality**. *Philipp Dufter, Hinrich Schütze*. EMNLP 2020. [[pdf](https://www.aclweb.org/anthology/2020.emnlp-main.358.pdf)]
41. **AUTOPROMPT: Eliciting Knowledge from Language Models with Automatically Generated Prompts**. *Taylor Shin, Yasaman Razeghi, Robert L Logan IV, Eric Wallace, Sameer Singh*. EMNLP 2020. [[pdf](https://www.aclweb.org/anthology/2020.emnlp-main.346.pdf)]
42. **The Lottery Ticket Hypothesis for Pre-trained BERT Networks**. *Tianlong Chen, Jonathan Frankle, Shiyu Chang, Sijia Liu, Yang Zhang, Zhangyang Wang, Michael Carbin*. NeurIPS 2020. [[pdf](https://papers.nips.cc/paper/2020/file/b6af2c9703f203a2794be03d443af2e3-Paper.pdf)]
