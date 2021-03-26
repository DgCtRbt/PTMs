# PTMs

预训练模型相关资料
补充领域介绍文档
每篇论文入选理由，1-3句话介绍该论文亮点。
随时更新

##  论文清单

### 综述

1.  论文：Pre-trained Models for Natural Language Processing: A Survey</br>地址:https://arxiv.org/pdf/2003.08271.pdf</br>复旦大学邱锡鹏等学者发布了的的自然语言处理处理中预训练模型PTMs的综述大全，共25页pdf205篇参考文献，从背景知识到当前代表性PTM模型和应用研究挑战等，是绝好的预训练语言模型的文献。

### 领域必读（10篇）

1. **context2vec: Learning Generic Context Embedding with Bidirectional LSTM**. *Oren Melamud, Jacob Goldberger, Ido Dagan*. CoNLL 2016. [[pdf](https://www.aclweb.org/anthology/K16-1006.pdf)] [[project](http://u.cs.biu.ac.il/~nlp/resources/downloads/context2vec/)] (**context2vec**).提出了一种无监督模型，借助双向LSTM从大型语料库中有效学习通用句子上下文表征。
2. Attention is all you need.[1] Vaswani, A. ,  Shazeer, N. et al. (2017).最强特征提取器,超越RNN的模型之作,BERT模型的重要组成部分.
3. Deep contextualized word representations.*Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee and Luke Zettlemoyer*. NAACL 2018. [[pdf](https://arxiv.org/pdf/1802.05365.pdf)] [[project](https://allennlp.org/elmo)] (**ELMo**).解决一词多义的动态词向量模型. 
4. **Improving Language Understanding by Generative Pre-Training**. *Alec Radford, Karthik Narasimhan, Tim Salimans and Ilya Sutskever*. Preprint. [[pdf](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)] [[project](https://openai.com/blog/language-unsupervised/)] (**GPT**).在这篇论文中，探索出了一种对自然语言理解任务的半监督方法，融合了无监督的预训练(pre-training)和有监督的微调(fine-tuning)过程。
5. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova*. NAACL 2019. [[pdf](https://arxiv.org/pdf/1810.04805.pdf)] [[code & model](https://github.com/google-research/bert)]. 划时代意义的预训练模型.
6. Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks. *Suchin Gururangan, Ana Marasović, Swabha Swayamdipta, Kyle Lo, Iz Beltagy, Doug Downey, Noah A. Smith*. ACL 2020. [[pdf](https://www.aclweb.org/anthology/2020.acl-main.740.pdf)]. ACL2020最佳论文,预训练训练与微调使用trick.
7. **ALBERT: A Lite BERT for Self-supervised Learning of Language Representations**.  *Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut*. ICLR 2020. [[pdf](https://openreview.net/pdf?id=H1eA7AEtvS)].一种针对Bert的模型压缩模型.
8. **Language Models are Unsupervised Multitask Learners**. *Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei and Ilya Sutskever*. Preprint. [[pdf](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)] [[code](https://github.com/openai/gpt-2)] (**GPT-2**).多任务预训练+超大数据集+超大规模模型
9. **ERNIE: Enhanced Language Representation with Informative Entities**. *Zhengyan Zhang, Xu Han, Zhiyuan Liu, Xin Jiang, Maosong Sun and Qun Liu*. ACL 2019. [[pdf](https://www.aclweb.org/anthology/P19-1139)] [[code & model](https://github.com/thunlp/ERNIE)] (**ERNIE (Tsinghua)** ).利用了大规模的语料信息和知识图谱，去训练一个增强的语言表示模型，它能够同时利用词汇、语义和知识信息。
10. **Defending Against Neural Fake News**. *Rowan Zellers, Ari Holtzman, Hannah Rashkin, Yonatan Bisk, Ali Farhadi, Franziska Roesner, Yejin Choi*. NeurIPS 2019. [[pdf](https://arxiv.org/pdf/1905.12616.pdf)] [[project](https://rowanzellers.com/grover/)] (**Grover**).本文讨论了不同的自然语言处理方法，以开发出对神经假新闻的强大防御，包括使用GPT-2检测器模型和Grover（AllenNLP）

### 较新重要（2-3年内，20篇)

### 最新可读（1年内，无篇数上限）
  
