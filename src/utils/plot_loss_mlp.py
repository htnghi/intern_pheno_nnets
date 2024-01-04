import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# plot train loss and val loss
train_loss = [0.12486829236149788, 0.03976386017166078, 0.0332864869851619, 0.021346784429624677, 0.014772219466976821, 0.017045073211193085, 0.013408330967649817, 0.01140540954656899, 0.009696920198621228, 0.007763704692479223, 0.007433379942085594, 0.006554103689268231, 0.005458769330289215, 0.00696427549701184, 0.00691379769705236, 0.006835183186922222, 0.005829962814459577, 0.005527727218577638, 0.004934976750519127, 0.005558771517826244, 0.004689226814662106, 0.004833677405258641, 0.004919243248878047, 0.005390586593421176, 0.0049812967772595584]
val_loss = [0.051838893443346024, 0.05959298647940159, 0.03249427489936352, 0.02883371990174055, 0.033468982204794884, 0.005434238584712148, 0.0056473256554454565, 0.006526528857648373, 0.005968909477815032, 0.007122889393940568, 0.002719505166169256, 0.003907444537617266, 0.0040409856010228395, 0.0071568775456398726, 0.0048921683337539434, 0.003317152033559978, 0.0031672873301431537, 0.004475079826079309, 0.003563197096809745, 0.00406513677444309, 0.0021293668542057276, 0.0023568723117932677, 0.0025815321132540703, 0.0028875345597043633, 0.003566128551028669]
x = np.arange(len(train_loss)) + 1 # epoch array

# finding minimum validation loss for annotation
valid_min = min(val_loss)
valid_ind = val_loss.index(valid_min)
x_min = x[valid_ind]
annot_text = "Min Valid Loss\n"
annot_text += "Loss: {:.4f}\n".format(valid_min)
# annot_text += "Epoch: {}".format(x_min)

# Plot
plt.subplots(figsize=(10, 6))
plt.plot(x, train_loss, color="blue", lw=2, ls='-', marker='h', label = "Train Loss")
plt.plot(x, val_loss, color="red",  lw=2, ls='-', marker='d', label = "Validation Loss")
plt.annotate(annot_text, xy=(x_min, valid_min), xytext=(x_min-2, valid_min+0.05),
            arrowprops=dict(arrowstyle="fancy"))
# Add in a title and axes labels
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.savefig('./MLP_loss.svg', bbox_inches='tight')
plt.show()

