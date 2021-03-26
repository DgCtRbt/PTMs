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

10.**XLNet: Generalized Autoregressive Pretraining for Language Understanding**. *Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le*. NeurIPS 2019. [[pdf](https://arxiv.org/pdf/1906.08237.pdf)] [[code & model](https://github.com/zihangdai/xlnet)]
   自回归预训练方法,克服了BERT的局限性。
11.  **Language Models are Unsupervised Multitask Learners**. *Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei and Ilya Sutskever*. Preprint. [[pdf](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)] [[code](https://github.com/openai/gpt-2)] (**GPT-2**).
   
   多任务预训练+超大数据集+超大规模模型
### 较新重要（2-3年内，20篇)
1. **RoBERTa: A Robustly Optimized BERT Pretraining Approach**. *Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov*. Preprint. [[pdf](https://arxiv.org/pdf/1907.11692.pdf)] [[code & model](https://github.com/pytorch/fairseq)]
2. **Knowledge Enhanced Contextual Word Representations**. *Matthew E. Peters, Mark Neumann, Robert L. Logan IV, Roy Schwartz, Vidur Joshi, Sameer Singh, Noah A. Smith*. EMNLP 2019. [[pdf](https://arxiv.org/pdf/1909.04164.pdf)] (**KnowBert**) 
3. **Contrastive Bidirectional Transformer for Temporal Representation Learning**. *Chen Sun, Fabien Baradel, Kevin Murphy, Cordelia Schmid*. Preprint. [[pdf](https://arxiv.org/pdf/1906.05743.pdf)] (**CBT**)
4. **ERNIE 2.0: A Continual Pre-training Framework for Language Understanding**. *Yu Sun, Shuohuan Wang, Yukun Li, Shikun Feng, Hao Tian, Hua Wu, Haifeng Wang*. Preprint. [[pdf](https://arxiv.org/pdf/1907.12412v1.pdf)] [[code](https://github.com/PaddlePaddle/ERNIE/blob/develop/README.md)] 
5. **BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension**. *Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov, Luke Zettlemoyer*. ACL 2020. [[pdf](https://arxiv.org/pdf/1910.13461.pdf)]
6. **FreeLB: Enhanced Adversarial Training for Language Understanding**. *Chen Zhu, Yu Cheng, Zhe Gan, Siqi Sun, Thomas Goldstein, Jingjing Liu*. ICLR 2020. [[pdf](https://openreview.net/pdf?id=BygzbyHFvB)]
7. **On the Sentence Embeddings from Pre-trained Language Models**. *Bohan Li, Hao Zhou, Junxian He, Mingxuan Wang, Yiming Yang, Lei Li*. EMNLP 2020. [[pdf](https://www.aclweb.org/anthology/2020.emnlp-main.733)]
8. **Pre-training via Paraphrasing**. *Mike Lewis, Marjan Ghazvininejad, Gargi Ghosh, Armen Aghajanyan, Sida Wang, Luke Zettlemoyer*. NeurIPS 2020. [[pdf](https://papers.nips.cc/paper/2020/file/d6f1dd034aabde7657e6680444ceff62-Paper.pdf)]
9. **PANLP at MEDIQA 2019: Pre-trained Language Models, Transfer Learning and Knowledge Distillation**. *Wei Zhu, Xiaofeng Zhou, Keqiang Wang, Xun Luo, Xiepeng Li, Yuan Ni, Guotong Xie*. The 18th BioNLP workshop. [[pdf](https://www.aclweb.org/anthology/W19-5040)]
10. **Reducing Transformer Depth on Demand with Structured Dropout**.  *Angela Fan, Edouard Grave, Armand Joulin*. ICLR 2020. [[pdf](https://openreview.net/pdf?id=SylO2yStDr)]
11. **Contrastive Distillation on Intermediate Representations for Language Model Compression**. *Siqi Sun, Zhe Gan, Yuwei Fang, Yu Cheng, Shuohang Wang, Jingjing Liu*. EMNLP 2020. [[pdf](https://www.aclweb.org/anthology/2020.emnlp-main.36)]
12. **When BERT Plays the Lottery, All Tickets Are Winning**. *Sai Prasanna, Anna Rogers, Anna Rumshisky*. EMNLP 2020. [[pdf](https://www.aclweb.org/anthology/2020.emnlp-main.259.pdf)]
13. **BERT Loses Patience: Fast and Robust Inference with Early Exit**. *Wangchunshu Zhou, Canwen Xu, Tao Ge, Julian McAuley, Ke Xu, Furu Wei*. NeurIPS 2020. [[pdf](https://papers.nips.cc/paper/2020/file/d4dd111a4fd973394238aca5c05bebe3-Paper.pdf)]
14. **Revealing the Dark Secrets of BERT**. *Olga Kovaleva, Alexey Romanov, Anna Rogers, Anna Rumshisky*. EMNLP 2019. [[pdf](https://arxiv.org/abs/1908.08593)] 
15. **How Does BERT Answer Questions? A Layer-Wise Analysis of Transformer Representations**. *Betty van Aken, Benjamin Winter, Alexander Löser, Felix A. Gers*. CIKM 2019. [[pdf](https://arxiv.org/pdf/1909.04925.pdf)]
16. **BERT has a Mouth, and It Must Speak: BERT as a Markov Random Field Language Model**. *Alex Wang, Kyunghyun Cho*. NeuralGen 2019. [[pdf](https://arxiv.org/pdf/1902.04094.pdf)] [[code](https://github.com/nyu-dl/bert-gen)]
17. **What Does BERT Look At? An Analysis of BERT's Attention**. *Kevin Clark, Urvashi Khandelwal, Omer Levy, Christopher D. Manning*. BlackBoxNLP 2019. [[pdf](https://arxiv.org/pdf/1906.04341.pdf)] [[code](https://github.com/clarkkev/attention-analysis)]
18. **BERT Rediscovers the Classical NLP Pipeline**. *Ian Tenney, Dipanjan Das, Ellie Pavlick*. ACL 2019. [[pdf](https://www.aclweb.org/anthology/P19-1452)]
19. **Language Models as Knowledge Bases?** *Fabio Petroni, Tim Rocktäschel, Patrick Lewis, Anton Bakhtin, Yuxiang Wu, Alexander H. Miller, Sebastian Riedel*. EMNLP 2019, [[pdf](https://arxiv.org/pdf/1909.01066.pdf)] [[code](https://github.com/facebookresearch/LAMA)]
20. **What do you learn from context? Probing for sentence structure in contextualized word representations**. *Ian Tenney, Patrick Xia, Berlin Chen, Alex Wang, Adam Poliak, R. Thomas McCoy, Najoung Kim, Benjamin Van Durme, Samuel R. Bowman,
Dipanjan Das, and Ellie Pavlick*. ICLR 2019. [[pdf](https://arxiv.org/pdf/1905.06316.pdf)]
### 最新可读（1年内，无篇数上限）
1. **Cross-Lingual Ability of Multilingual BERT: An Empirical Study**. *Karthikeyan K, Zihan Wang, Stephen Mayhew, Dan Roth*. ICLR 2020. [[pdf](https://openreview.net/pdf?id=HJeT3yrtDr)]
2. **Finding Universal Grammatical Relations in Multilingual BERT**. *Ethan A. Chi, John Hewitt, Christopher D. Manning*. ACL 2020. [[pdf](https://www.aclweb.org/anthology/2020.acl-main.493.pdf)]
3. **Negated and Misprimed Probes for Pretrained Language Models: Birds Can Talk, But Cannot Fly**. *Nora Kassner, Hinrich Schütze*. ACL 2020. [[pdf](https://www.aclweb.org/anthology/2020.acl-main.698.pdf)]
4. **Perturbed Masking: Parameter-free Probing for Analyzing and Interpreting BERT**. *Zhiyong Wu, Yun Chen, Ben Kao, Qun Liu*. ACL 2020. [[pdf](https://www.aclweb.org/anthology/2020.acl-main.383.pdf)]
5. **Birds have four legs?! NumerSense: Probing Numerical Commonsense Knowledge of Pre-trained Language Models**. *Bill Yuchen Lin, Seyeon Lee, Rahul Khanna and Xiang Ren*. EMNLP 2020. [[pdf](https://www.aclweb.org/anthology/2020.emnlp-main.557)]
6. **Identifying Elements Essential for BERT’s Multilinguality**. *Philipp Dufter, Hinrich Schütze*. EMNLP 2020. [[pdf](https://www.aclweb.org/anthology/2020.emnlp-main.358.pdf)]
7. **AUTOPROMPT: Eliciting Knowledge from Language Models with Automatically Generated Prompts**. *Taylor Shin, Yasaman Razeghi, Robert L Logan IV, Eric Wallace, Sameer Singh*. EMNLP 2020. [[pdf](https://www.aclweb.org/anthology/2020.emnlp-main.346.pdf)]
8. **The Lottery Ticket Hypothesis for Pre-trained BERT Networks**. *Tianlong Chen, Jonathan Frankle, Shiyu Chang, Sijia Liu, Yang Zhang, Zhangyang Wang, Michael Carbin*. NeurIPS 2020. [[pdf](https://papers.nips.cc/paper/2020/file/b6af2c9703f203a2794be03d443af2e3-Paper.pdf)]
9. **SMART: Robust and Efficient Fine-Tuning for Pre-trained Natural Language Models through Principled Regularized Optimization**. *Haoming Jiang, Pengcheng He, Weizhu Chen, Xiaodong Liu, Jianfeng Gao, Tuo Zhao*. ACL 2020. [[pdf](https://www.aclweb.org/anthology/2020.acl-main.197.pdf)]
10. **Do you have the right scissors? Tailoring Pre-trained Language Models via Monte-Carlo Methods**. *Ning Miao, Yuxuan Song, Hao Zhou, Lei Li*. ACL 2020. [[pdf](https://www.aclweb.org/anthology/2020.acl-main.314.pdf)]
11. **ExpBERT: Representation Engineering with Natural Language Explanations**. *Shikhar Murty, Pang Wei Koh, Percy Liang*. ACL 2020. [[pdf](https://www.aclweb.org/anthology/2020.acl-main.190.pdf)]
12. **Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks**. *Suchin Gururangan, Ana Marasović, Swabha Swayamdipta, Kyle Lo, Iz Beltagy, Doug Downey, Noah A. Smith*. ACL 2020. [[pdf](https://www.aclweb.org/anthology/2020.acl-main.740.pdf)]
13. **Recall and Learn: Fine-tuning Deep Pretrained Language Models with Less Forgetting**. *Sanyuan Chen, Yutai Hou, Yiming Cui, Wanxiang Che, Ting Liu, Xiangzhan Yu*. EMNLP 2020. [[pdf](https://www.aclweb.org/anthology/2020.emnlp-main.634.pdf)]
14. **Masking as an Efficient Alternative to Finetuning for Pretrained Language Models**. *Mengjie Zhao, Tao Lin, Fei Mi, Martin Jaggi, Hinrich Schütze*. EMNLP 2020. [[pdf](https://www.aclweb.org/anthology/2020.emnlp-main.174.pdf)]
15. **CogLTX: Applying BERT to Long Texts**. *Ming Ding, Chang Zhou, Hongxia Yang, Jie Tang*. NeurIPS 2020. [[pdf](https://papers.nips.cc/paper/2020/file/96671501524948bc3937b4b30d0e57b9-Paper.pdf)]

参考资料:

1. https://github.com/thunlp/PLMpapers
2. https://my.oschina.net/u/4246997/blog/4480524
