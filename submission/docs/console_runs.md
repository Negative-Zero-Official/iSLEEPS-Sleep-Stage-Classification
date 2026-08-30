# Console Runs

```
python train_gbdt.py --cache cache --folds 5 --save-proba

93,937 epochs x 426 features | 99 recordings | 86 patients
class balance: {'Wake': 25921, 'N1': 9360, 'N2': 39198, 'N3': 8220, 'REM': 11238}
class weights: {'Wake': 0.851, 'N1': 1.417, 'N2': 0.692, 'N3': 1.512, 'REM': 1.293}
run fingerprint: 4f1eb453af6462ae

  fold 1/5 [LightGBM]: 74,982 train / 18,955 test  acc 0.7622  trees 400  (25s)
  fold 2/5 [LightGBM]: 75,018 train / 18,919 test  acc 0.7646  trees 400  (24s)
  fold 3/5 [LightGBM]: 75,486 train / 18,451 test  acc 0.7741  trees 400  (26s)
  fold 4/5 [LightGBM]: 75,103 train / 18,834 test  acc 0.7826  trees 400  (25s)
  fold 5/5 [LightGBM]: 75,159 train / 18,778 test  acc 0.7724  trees 400  (25s)

=== Gradient boosting -- 5-fold, grouped by patient ===
  accuracy   0.7712
  macro F1   0.7070
  Cohen kappa 0.6768

  stage      prec  recall      F1  support
  Wake      0.820   0.852   0.835   25,921
  N1        0.407   0.338   0.370    9,360
  N2        0.796   0.846   0.820   39,198
  N3        0.748   0.715   0.731    8,220
  REM       0.840   0.725   0.779   11,238

  confusion (row = truth, %)
               Wake      N1      N2      N3     REM
  Wake       85.2     8.5     5.5     0.1     0.7
  N1         30.5    33.8    30.4     0.4     4.9
  N2          3.7     4.6    84.6     4.9     2.3
  N3          0.6     0.0    27.6    71.5     0.2
  REM         4.4     5.5    17.3     0.2    72.5

  top 20 features
    epoch_position                     0.952%
    hours_from_start                   0.931%
    EMG_log_iqr                        0.814%
    EOG_corr_z_sm15                    0.676%
    EMG_skew_z_sm15                    0.603%
    EMG_kurtosis_z_sm15                0.564%
    C4_kurtosis_z_sm15                 0.552%
    E2_skew_z_sm15                     0.544%
    EOG_corr_z_sm5                     0.530%
    EOG_corr                           0.526%
    E1_kurtosis_z_sm15                 0.520%
    EMG_log_rms                        0.515%
    C3_log_sigma_theta_z_sm15          0.513%
    C4_log_sigma_theta_z_sm15          0.500%
    C3_kurtosis_z_sm15                 0.498%
    O2_skew_z_sm15                     0.492%
    C4_log_iqr                         0.490%
    C4_log_alpha_beta                  0.480%
    C3_skew_z_sm15                     0.480%
    E2_kurtosis_z_sm15                 0.480%

saved -> results/gbdt_oof.npz
```

```
python train_gbdt.py --cache cache --folds 5 --save-proba --n-estimators 1200 --out results_1200
PS C:\Users\pgait\Documents\Coding Projects and Repos\iSLEEPS Sleep Classification> python train_gbdt.py --cache cache --folds 5 --save-proba --n-estimators 1200 --out results_1200
93,937 epochs x 426 features | 99 recordings | 86 patients
class balance: {'Wake': 25921, 'N1': 9360, 'N2': 39198, 'N3': 8220, 'REM': 11238}
class weights: {'Wake': 0.851, 'N1': 1.417, 'N2': 0.692, 'N3': 1.512, 'REM': 1.293}
run fingerprint: a77e403236957bc0

  fold 1/5 [LightGBM]: 74,982 train / 18,955 test  acc 0.7678  trees 1200  (71s)
  fold 2/5 [LightGBM]: 75,018 train / 18,919 test  acc 0.7682  trees 1200  (71s)
  fold 3/5 [LightGBM]: 75,486 train / 18,451 test  acc 0.7752  trees 1200  (72s)
  fold 4/5 [LightGBM]: 75,103 train / 18,834 test  acc 0.7847  trees 1200  (71s)
  fold 5/5 [LightGBM]: 75,159 train / 18,778 test  acc 0.7745  trees 1200  (71s)

=== Gradient boosting -- 5-fold, grouped by patient ===
  accuracy   0.7741
  macro F1   0.7060
  Cohen kappa 0.6794

  stage      prec  recall      F1  support
  Wake      0.819   0.857   0.838   25,921
  N1        0.423   0.312   0.359    9,360
  N2        0.790   0.856   0.822   39,198
  N3        0.754   0.706   0.729    8,220
  REM       0.842   0.731   0.783   11,238

  confusion (row = truth, %)
               Wake      N1      N2      N3     REM
  Wake       85.7     7.4     6.1     0.1     0.8
  N1         31.3    31.2    32.1     0.5     4.9
  N2          3.7     3.9    85.6     4.6     2.2
  N3          0.6     0.0    28.6    70.6     0.2
  REM         4.3     4.9    17.5     0.2    73.1

  top 20 features
    hours_from_start                   0.763%
    epoch_position                     0.752%
    EMG_log_iqr                        0.644%
    EMG_skew_z_sm15                    0.586%
    EOG_corr_z_sm15                    0.554%
    EMG_kurtosis_z_sm15                0.546%
    E2_skew_z_sm15                     0.529%
    O2_skew_z_sm15                     0.503%
    EOG_corr_z_sm5                     0.496%
    C4_kurtosis_z_sm15                 0.496%
    E1_skew_z_sm15                     0.484%
    C4_skew_z_sm15                     0.483%
    C3_skew_z_sm15                     0.477%
    EOG_corr                           0.475%
    E1_kurtosis_z_sm15                 0.470%
    C3_kurtosis_z_sm15                 0.467%
    E2_kurtosis_z_sm15                 0.467%
    O1_skew_z_sm15                     0.461%
    O1_log_alpha_beta_z_sm15           0.453%
    O2_log_alpha_beta_z_sm15           0.442%

saved -> results_1200/gbdt_oof.npz
```

```
python train_cnn.py  --cache cache --folds 5 --save-proba
PS C:\Users\pgait\Documents\Coding Projects and Repos\iSLEEPS Sleep Classification> python train_cnn.py  --cache cache --folds 5 --save-proba                                       
93,937 epochs | 99 recordings | 86 patients | device=cuda
run fingerprint: d8fdcde8f826f4ee

  fold 1/5: 79 train / 20 test recordings, 18,955 test epochs
    epoch  1/60  loss 0.9177  val kappa 0.4956  *  (7s)
    epoch  2/60  loss 0.7420  val kappa 0.5972  *  (5s)
    epoch  3/60  loss 0.6744  val kappa 0.6092  *  (5s)
    epoch  4/60  loss 0.6374  val kappa 0.5952  (5s)
    epoch  5/60  loss 0.6291  val kappa 0.6680  *  (5s)
    epoch  6/60  loss 0.5983  val kappa 0.5997  (5s)
    epoch  7/60  loss 0.5806  val kappa 0.6039  (5s)
    epoch  8/60  loss 0.5550  val kappa 0.6144  (5s)
    epoch  9/60  loss 0.5544  val kappa 0.6355  (5s)
    epoch 10/60  loss 0.5239  val kappa 0.6587  (5s)
    epoch 11/60  loss 0.4935  val kappa 0.6242  (5s)
    epoch 12/60  loss 0.4802  val kappa 0.6407  (5s)
    epoch 13/60  loss 0.4740  val kappa 0.6471  (5s)
    epoch 14/60  loss 0.4618  val kappa 0.6655  (5s)
    epoch 15/60  loss 0.4370  val kappa 0.6422  (5s)
    early stop at epoch 15; best val kappa 0.6680
    fold accuracy 0.6868

  fold 2/5: 79 train / 20 test recordings, 18,919 test epochs
    epoch  1/60  loss 0.9351  val kappa 0.5504  *  (5s)
    epoch  2/60  loss 0.7529  val kappa 0.4869  (5s)
    epoch  3/60  loss 0.6954  val kappa 0.6136  *  (5s)
    epoch  4/60  loss 0.6902  val kappa 0.6040  (5s)
    epoch  5/60  loss 0.6348  val kappa 0.6344  *  (5s)
    epoch  6/60  loss 0.6040  val kappa 0.6506  *  (5s)
    epoch  7/60  loss 0.5903  val kappa 0.6166  (5s)
    epoch  8/60  loss 0.5689  val kappa 0.6132  (5s)
    epoch  9/60  loss 0.5515  val kappa 0.6068  (5s)
    epoch 10/60  loss 0.5196  val kappa 0.6444  (5s)
    epoch 11/60  loss 0.5276  val kappa 0.6254  (5s)
    epoch 12/60  loss 0.4984  val kappa 0.6050  (5s)
    epoch 13/60  loss 0.5044  val kappa 0.5752  (5s)
    epoch 14/60  loss 0.4908  val kappa 0.6173  (5s)
    epoch 15/60  loss 0.4953  val kappa 0.5776  (5s)
    epoch 16/60  loss 0.4562  val kappa 0.6115  (5s)
    early stop at epoch 16; best val kappa 0.6506
    fold accuracy 0.7293

  fold 3/5: 80 train / 19 test recordings, 18,451 test epochs
    epoch  1/60  loss 0.8960  val kappa 0.5690  *  (5s)
    epoch  2/60  loss 0.7125  val kappa 0.5612  (5s)
    epoch  3/60  loss 0.6667  val kappa 0.5414  (5s)
    epoch  4/60  loss 0.6553  val kappa 0.5998  *  (5s)
    epoch  5/60  loss 0.6152  val kappa 0.5803  (5s)
    epoch  6/60  loss 0.5844  val kappa 0.5987  (5s)
    epoch  7/60  loss 0.5527  val kappa 0.5703  (5s)
    epoch  8/60  loss 0.5490  val kappa 0.4670  (5s)
    epoch  9/60  loss 0.5274  val kappa 0.5933  (5s)
    epoch 10/60  loss 0.5115  val kappa 0.5967  (5s)
    epoch 11/60  loss 0.4856  val kappa 0.5433  (5s)
    epoch 12/60  loss 0.4651  val kappa 0.5769  (5s)
    epoch 13/60  loss 0.4502  val kappa 0.5454  (5s)
    epoch 14/60  loss 0.4444  val kappa 0.5968  (5s)
    early stop at epoch 14; best val kappa 0.5998
    fold accuracy 0.7307

  fold 4/5: 79 train / 20 test recordings, 18,834 test epochs
    epoch  1/60  loss 0.9839  val kappa 0.5788  *  (5s)
    epoch  2/60  loss 0.7612  val kappa 0.5956  *  (5s)
    epoch  3/60  loss 0.6959  val kappa 0.6172  *  (5s)
    epoch  4/60  loss 0.6727  val kappa 0.6015  (5s)
    epoch  5/60  loss 0.6276  val kappa 0.5879  (5s)
    epoch  6/60  loss 0.5973  val kappa 0.6155  (5s)
    epoch  7/60  loss 0.5743  val kappa 0.5869  (5s)
    epoch  8/60  loss 0.5549  val kappa 0.5970  (5s)
    epoch  9/60  loss 0.5480  val kappa 0.6297  *  (5s)
    epoch 10/60  loss 0.5127  val kappa 0.5648  (5s)
    epoch 11/60  loss 0.5068  val kappa 0.6013  (5s)
    epoch 12/60  loss 0.4925  val kappa 0.6095  (5s)
    epoch 13/60  loss 0.4745  val kappa 0.6054  (5s)
    epoch 14/60  loss 0.4646  val kappa 0.6055  (5s)
    epoch 15/60  loss 0.4499  val kappa 0.6093  (5s)
    epoch 16/60  loss 0.4298  val kappa 0.5787  (5s)
    epoch 17/60  loss 0.4160  val kappa 0.5934  (5s)
    epoch 18/60  loss 0.4015  val kappa 0.6441  *  (5s)
    epoch 19/60  loss 0.3918  val kappa 0.6029  (5s)
    epoch 20/60  loss 0.3922  val kappa 0.6354  (5s)
    epoch 21/60  loss 0.3796  val kappa 0.5622  (5s)
    epoch 22/60  loss 0.3734  val kappa 0.6033  (5s)
    epoch 23/60  loss 0.3584  val kappa 0.6202  (5s)
    epoch 24/60  loss 0.3635  val kappa 0.6104  (5s)
    epoch 25/60  loss 0.3300  val kappa 0.6206  (5s)
    epoch 26/60  loss 0.3204  val kappa 0.5888  (5s)
    epoch 27/60  loss 0.3300  val kappa 0.6245  (5s)
    epoch 28/60  loss 0.3206  val kappa 0.6220  (5s)
    early stop at epoch 28; best val kappa 0.6441
    fold accuracy 0.7414

  fold 5/5: 79 train / 20 test recordings, 18,778 test epochs
    epoch  1/60  loss 0.9515  val kappa 0.6066  *  (5s)
    epoch  2/60  loss 0.7533  val kappa 0.6564  *  (5s)
    epoch  3/60  loss 0.7211  val kappa 0.6426  (5s)
    epoch  4/60  loss 0.6666  val kappa 0.6323  (5s)
    epoch  5/60  loss 0.6433  val kappa 0.6406  (5s)
    epoch  6/60  loss 0.5960  val kappa 0.6015  (5s)
    epoch  7/60  loss 0.5982  val kappa 0.6299  (5s)
    epoch  8/60  loss 0.5565  val kappa 0.6156  (5s)
    epoch  9/60  loss 0.5517  val kappa 0.6497  (5s)
    epoch 10/60  loss 0.5332  val kappa 0.6399  (5s)
    epoch 11/60  loss 0.5265  val kappa 0.6689  *  (5s)
    epoch 12/60  loss 0.4892  val kappa 0.6047  (5s)
    epoch 13/60  loss 0.4777  val kappa 0.6453  (5s)
    epoch 14/60  loss 0.4562  val kappa 0.5971  (5s)
    epoch 15/60  loss 0.4629  val kappa 0.6553  (5s)
    epoch 16/60  loss 0.4329  val kappa 0.6733  *  (5s)
    epoch 17/60  loss 0.4279  val kappa 0.6490  (5s)
    epoch 18/60  loss 0.4135  val kappa 0.6677  (5s)
    epoch 19/60  loss 0.4017  val kappa 0.6725  (5s)
    epoch 20/60  loss 0.4025  val kappa 0.6425  (5s)
    epoch 21/60  loss 0.3880  val kappa 0.6432  (5s)
    epoch 22/60  loss 0.3631  val kappa 0.6641  (5s)
    epoch 23/60  loss 0.3623  val kappa 0.6375  (5s)
    epoch 24/60  loss 0.3554  val kappa 0.6278  (5s)
    epoch 25/60  loss 0.3454  val kappa 0.6422  (5s)
    epoch 26/60  loss 0.3363  val kappa 0.6414  (5s)
    early stop at epoch 26; best val kappa 0.6733
    fold accuracy 0.7216

=== CNN + BiGRU -- 5 of 5 folds, grouped by patient ===
  accuracy   0.7219
  macro F1   0.6602
  Cohen kappa 0.6144

  stage      prec  recall      F1  support
  Wake      0.821   0.774   0.797   25,921
  N1        0.332   0.363   0.347    9,360
  N2        0.787   0.785   0.786   39,198
  N3        0.641   0.737   0.686    8,220
  REM       0.704   0.669   0.686   11,238

  confusion (row = truth, %)
               Wake      N1      N2      N3     REM
  Wake       77.4    14.1     5.2     0.2     3.1
  N1         26.1    36.3    29.9     1.2     6.5
  N2          3.2     6.0    78.5     8.1     4.2
  N3          0.9     0.8    23.5    73.7     1.1
  REM         5.6     6.8    20.2     0.5    66.9

saved -> results/cnn_oof.npz
```

```
python compare.py
PS C:\Users\pgait\Documents\Coding Projects and Repos\iSLEEPS Sleep Classification> python compare.py                                          
alignment verified: both models scored on the same 93,937 epochs, same folds

                        accuracy  macro F1    kappa
  gradient boosting       0.7712    0.7070   0.6768
  CNN + BiGRU             0.7219    0.6602   0.6144

  per-stage F1              GBDT     CNN    delta
  Wake                     0.835   0.797   -0.039
  N1                       0.370   0.347   -0.023
  N2                       0.820   0.786   -0.034
  N3                       0.731   0.686   -0.046
  REM                      0.779   0.686   -0.093

  agreement between models: 76.5%
  McNemar (epoch-level): GBDT-only-right 11,695  CNN-only-right 7,066  chi2 1141.6  p 2.94e-250
    (epochs within a night are correlated, so treat this as anti-conservative)

  per-recording paired test (n=99 recordings, the independent unit):
    GBDT better on 74/99   mean delta +0.0523   Wilcoxon p 2.35e-07

  oracle upper bound (either model right): 0.8464 vs 0.7712 best single -> +0.0752 headroom from combining
```

```
python stack.py
PS C:\Users\pgait\Documents\Coding Projects and Repos\iSLEEPS Sleep Classification> python stack.py                                                                                                                     
empirical stage self-transition probabilities (why smoothing helps):
    Wake   stays 0.908
    N1     stays 0.711
    N2     stays 0.930
    N3     stays 0.913
    REM    stays 0.952

choosing the transition weight per fold on training recordings only:
      fold alpha=0.15 (train kappa 0.6845)
      fold alpha=0.15 (train kappa 0.6821)
      fold alpha=0.15 (train kappa 0.6806)
      fold alpha=0.6 (train kappa 0.6753)
      fold alpha=0.6 (train kappa 0.6819)
      fold alpha=0.15 (train kappa 0.6302)
      fold alpha=0.15 (train kappa 0.6103)
      fold alpha=0.15 (train kappa 0.6119)
      fold alpha=0.15 (train kappa 0.6061)
      fold alpha=0.15 (train kappa 0.6176)
      fold alpha=0.15 (train kappa 0.6967)
      fold alpha=0.15 (train kappa 0.6867)
      fold alpha=0.15 (train kappa 0.6903)
      fold alpha=0.15 (train kappa 0.6833)
      fold alpha=0.15 (train kappa 0.6894)

                                  accuracy  macro F1    kappa
  gradient boosting                 0.7712    0.7070   0.6768
  gradient boosting + Viterbi       0.7741    0.7058   0.6795
  CNN + BiGRU                       0.7219    0.6602   0.6144
  CNN + Viterbi                     0.7233    0.6593   0.6153
  ensemble (equal weight)           0.7794    0.7137   0.6885
  ensemble + Viterbi                0.7805    0.7125   0.6893

=== best variant: ensemble + Viterbi ===
  accuracy   0.7805
  macro F1   0.7125
  Cohen kappa 0.6893

  stage      prec  recall      F1  support
  Wake      0.832   0.860   0.846   25,921
  N1        0.433   0.325   0.372    9,360
  N2        0.800   0.861   0.830   39,198
  N3        0.735   0.735   0.735    8,220
  REM       0.842   0.727   0.780   11,238

  confusion (row = truth, %)
               Wake      N1      N2      N3     REM
  Wake       86.0     7.9     5.1     0.1     0.9
  N1         30.9    32.5    31.0     0.8     4.7
  N2          2.8     3.6    86.1     5.2     2.2
  N3          0.4     0.0    26.1    73.5     0.0
  REM         4.0     4.7    18.4     0.2    72.7

saved -> results/stacked.npz (ensemble + Viterbi)
```