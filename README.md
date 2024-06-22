# AM_DeepONet
Application of two different DeepONet models in multi-physics solidification and additive manufacturing applications. Two DeepONet models were employed:
1. A sequential DeepONet model with a GRU network. Used to capture time-dependent input functions in the solidification problem.
2. A ResUNet-based DeepONet model with a ResUNet in the trunk network. Used to capture the changing designs in the LDED prediction problem.

The DeepONet implementation and training is based on DeepXDE:
@article{lu2021deepxde,
  title={DeepXDE: A deep learning library for solving differential equations},
  author={Lu, Lu and Meng, Xuhui and Mao, Zhiping and Karniadakis, George Em},
  journal={SIAM review},
  volume={63},
  number={1},
  pages={208--228},
  year={2021},
  publisher={SIAM}
}

The implementation of the ResUNet is adapted from Jan Palase: https://github.com/JanPalasek/resunet-tensorflow

If you find our model helpful in your specific applications and researches, please cite this article as: 
@article{kushwaha2024advanced,
  title={Advanced deep operator networks to predict multiphysics solution fields in materials processing and additive manufacturing},
  author={Kushwaha, Shashank and Park, Jaewan and Koric, Seid and He, Junyan and Jasiuk, Iwona and Abueidda, Diab},
  journal={Additive Manufacturing},
  pages={104266},
  year={2024},
  publisher={Elsevier}
}

The training data is large in size and can be downloaded through the following UIUC Box link: 
https://uofi.box.com/s/m91ux8n3aiygpwu7dliaebw3p866dt0p
All three models described in the paper are provided.
