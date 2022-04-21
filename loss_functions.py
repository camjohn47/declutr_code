from tensorflow.keras.losses import Loss
from tensorflow.keras.losses import categorical_crossentropy

import sys

class MaskedMethodLoss(Loss):
    def __init__(self):
        super().__init__()

    def get_config(self):
        return super().get_config()

    def call(self, outputs, labels):
        mlm_loss = categorical_crossentropy(outputs, labels)
        return mlm_loss

class ContrastiveLoss(Loss):
    def __init__(self):
        super().__init__()

    def get_config(self):
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, outputs, labels):
        contrastive_loss = categorical_crossentropy(outputs, labels)
        return contrastive_loss

class DeclutrLoss(Loss):
    OBJECTIVE_OPTIONS = ['contrastive', 'masked_method', 'contrastive_and_masked_method']
    OBJECTIVE_LOSSES = dict(contrastive=ContrastiveLoss, masked_method=MaskedMethodLoss,
                            contrastive_and_masked_method=lambda x,y: ContrastiveLoss(x,y) + MaskedMethodLoss(x,y))

    def __init__(self, objective="contrastive_and_masked_method"):
        super().__init__()
        if objective not in self.OBJECTIVE_OPTIONS:
            print(f'ERROR: Requested DeClutr loss objective = {objective} not in options: {self.OBJECTIVE_OPTIONS}')
            sys.exit(1)

        self.objective = objective
        self.loss = self.OBJECTIVE_LOSSES[self.objective]()

    def get_config(self):
        config = super().get_config()
        config['objective'] = self.objective
        return config

    def contrastive_loss(self, outputs, labels):
        contrastive_outputs = outputs['contrastive']
        contrastive_labels = labels['contrastive']
        contrastive_loss = categorical_crossentropy(contrastive_outputs, contrastive_labels)
        return contrastive_loss

    def call(self, declutr_outputs, declutr_labels):
        '''
        Revised version of the DeClutr loss function shown in 3.3 of the original paper:

        loss = contrastive loss + masked method loss,

        where

        contrastive loss = Cross Entropy(contrastive outputs, contrastive labels),

        and

        masked method loss = cross entropy(masked method predictions, masked method labels).
        '''

        declutr_loss = self.loss(declutr_outputs, declutr_labels)
        return declutr_loss
